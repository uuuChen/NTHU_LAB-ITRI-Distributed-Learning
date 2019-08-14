# System Imports
from __future__ import print_function
from torch.autograd import Variable
import argparse
from socket import *
import pickle

# Model Imports
from model.LeNet import *

# DataSet Imports
from dataSet.MNIST_dataSet import *
from data.data_args import *  # import data arguments

os.chdir('../')
# server socket setting
ip_port = ('localhost', 8080)
back_log = 5
buffer_size = 4096

tcp_server = socket(AF_INET, SOCK_STREAM)
tcp_server.bind(ip_port)
tcp_server.listen(back_log)

# training settings
parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 512)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 30)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

train_args = parser.parse_args(args=[])

test_dataSet = MNIST_DataSet(data_args=MNIST_TEST_ARGS,
                             shuffle=True)

model_server = Server_LeNet()

def train_epoch(epoch,conn):
    # get data_nums
    while 1:
        try:
            data = conn.recv(buffer_size)
            data_nums = pickle.loads(data)
            print("get data_nums:",data_nums)
            break
        except Exception:
            break

    batches = (data_nums - 1) // train_args.batch_size + 1
    datas_tratned_num = 0
    for batch_idx in range(1, batches + 1):
        model_server.train()
        optimizer_server.zero_grad()

        # get gent_output
        while 1:
            try:
                data = conn.recv(1000000)
                agent_output = pickle.loads(data)
                print("get agent_output")
                break
            except Exception:
                break
        # get target
        while 1:
            try:
                data = conn.recv(1000000)
                target = pickle.loads(data)
                print("get target")
                break
            except Exception:
                break

        agent_output_clone = Variable(agent_output).float()
        agent_output_clone.requires_grad_()
        if train_args.cuda:
            agent_output_clone = agent_output_clone.cuda()
            target = target.cuda()

        #server forward
        output = model_server(agent_output_clone)

        #server backward
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer_server.step()

        #agent backward

        # send agent_output_clone
        print("send agent_output_clone_gradient")
        msg = pickle.dumps(agent_output_clone.grad.data)
        while 1:
            try:
                conn.send(msg)
                break
            except Exception:
                break

        datas_tratned_num += len(target)
        if batch_idx % train_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, datas_tratned_num, data_nums,
                       100. * batch_idx / batches, loss.item()))


def test_epoch(conn):

    model_server.eval()
    # get model_agent
    print("get model_agent")
    while 1:
        try:
            data = conn.recv(10000000)
            model_agent = pickle.loads(data)
            break
        except Exception:
            break
    model_agent.eval()

    test_loss = 0

    correct = 0

    data_nums = test_dataSet.get_data_nums_from_database()

    batches = (data_nums - 1) // train_args.batch_size + 1

    datas, targets = test_dataSet.get_data_and_labels(batch_size=train_args.batch_size,
                                                       one_hot=False)

    for batch_idx in range(1, batches + 1):

        data = datas[(batch_idx-1) * train_args.batch_size:batch_idx * train_args.batch_size]
        target = targets[(batch_idx-1) * train_args.batch_size:batch_idx * train_args.batch_size]

        if train_args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data).float(), Variable(target).long()

        agent_output = model_agent(data)
        output = model_server(agent_output)

        test_loss += F.cross_entropy(output, target).item()

        pred = output.data.max(1)[1]

        correct += pred.eq(target.data).cpu().sum()

    test_loss /= batches

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data_nums,
        100. * correct / data_nums))

if __name__ == '__main__':
    msg = pickle.dumps(train_args)
    while True:
        try:
            print('服務端開始運行了')
            conn, addr = tcp_server.accept()  # 服務端阻塞
            print('雙向連結是', conn)
            print('客戶端地址', addr)
            # send train_args
            print("send train_args")
            conn.send(msg)
            break
        except Exception:
            break

    # train setup
    train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                        # deterministic
    if train_args.cuda:
        torch.cuda.manual_seed(train_args.seed)  # set a random seed for the current GPU
        model_server.cuda()  # move all model parameters to the GPU

    optimizer_server = optim.SGD(model_server.parameters(),
                          lr=train_args.lr,
                          momentum=train_args.momentum)

    for epoch in range(1, train_args.epochs + 1):
        train_epoch(epoch=epoch,conn=conn)
        test_epoch(conn=conn)


    # close socket
    conn.close()
    tcp_server.close()

