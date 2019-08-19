# System Imports
from __future__ import print_function
from torch.autograd import Variable
import argparse

# Model Imports
from model.LeNet import *

# DataSet Imports
from dataSet.MNIST_dataSet import *

# Socket Imports
from socket_.socket_ import *
from socket_.socket_args import *

# training settings
parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=512, metavar='N',
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

model_server = Server_LeNet()

# server socket setting
server_sock = Socket(SERVER_SOCKET_ARGS)

def train_epoch(epoch):

    model_server.train()
    data_nums = server_sock.recv('data_nums')
    batches = (data_nums - 1) // train_args.batch_size + 1

    trained_data_num = 0
    for batch_idx in range(1, batches + 1):

        optimizer_server.zero_grad()

        # get agent_output and target from agent
        agent_output = server_sock.recv('agent_output')
        target = server_sock.recv('target')

        # store gradient in agent_output_clone
        agent_output_clone = Variable(agent_output).float()
        agent_output_clone.requires_grad_()

        if train_args.cuda:
            agent_output_clone = agent_output_clone.cuda()
            target = target.cuda()

        # server forward
        server_output = model_server(agent_output_clone)
        loss = F.cross_entropy(server_output, target)

        # server backward
        loss.backward()
        optimizer_server.step()

        # send gradient to agent
        server_sock.send(agent_output_clone.grad.data, 'agent_output_clone_grad')

        trained_data_num += len(target)
        if batch_idx % train_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, trained_data_num, data_nums,
                       100. * batch_idx / batches, loss.item()))

def test_epoch():

    model_server.eval()

    test_loss = 0
    correct = 0

    data_nums = server_sock.recv('data_nums')
    batches = (data_nums - 1) // train_args.batch_size + 1

    for batch_idx in range(1, batches + 1):

        # get agent_output and target from agent
        agent_output = server_sock.recv('agent_output')
        target = server_sock.recv('target')

        agent_output_clone = Variable(agent_output).float()

        if train_args.cuda:
            agent_output_clone = agent_output_clone.cuda()
            target = target.cuda()

        # server forward
        server_output = model_server(agent_output_clone)
        loss = F.cross_entropy(server_output, target)

        test_loss += loss

        pred = server_output.data.max(1)[1]

        correct += pred.eq(target.data).cpu().sum()

    test_loss /= batches

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data_nums,
        100. * correct / data_nums))

if __name__ == '__main__':

    # wait for agent connect
    server_sock.accept()

    # get train args from agent
    server_sock.send(train_args, 'train_args')

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
        train_epoch(epoch=epoch)
        test_epoch()

