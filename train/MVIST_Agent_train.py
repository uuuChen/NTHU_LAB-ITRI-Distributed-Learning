# System Imports
from __future__ import print_function
from torch.autograd import Variable
import argparse

# Model Imports
from model.LeNet import *

# DataSet Imports
from dataSet.MNIST_dataSet import *
from data.data_args import *  # import data arguments

# Socket Imports
from socket_.socket_ import *
from socket_.socket_args import *

os.chdir('../')

# agent socket_ setting
agent_sock = Socket(AGENT_SOCKET_ARGS)

# training settings
train_dataSet = MNIST_DataSet(data_args=MNIST_TRAIN_ARGS,
                              shuffle=True)

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

test_dataSet = MNIST_DataSet(data_args=MNIST_TEST_ARGS,
                             shuffle=True)

model_agent = Agent_LeNet()

def train_epoch(epoch):

    model_agent.train()
    data_nums = train_dataSet.get_data_nums_from_database()
    batches = (data_nums - 1) // train_args.batch_size + 1

    for batch_idx in range(1, batches + 1):

        data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size)

        if train_args.cuda:
            data = data.cuda()

        data, target = Variable(data).float(), Variable(target).long()

        #agent forward
        optimizer_agent.zero_grad()
        agent_output = model_agent(data)

        # send things to server
        agent_sock.send(agent_output, 'agent_output')  # send agent_output
        agent_sock.send(target, 'target')  # send target
        agent_output_grad = agent_sock.recv('agent_output_clone_grad')  # get agent_output_clone

        # receive loss value from server
        loss = agent_sock.recv('loss')

        #agent backward
        agent_output.backward(gradient=agent_output_grad)
        optimizer_agent.step()

        if batch_idx % train_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train_args.batch_size, data_nums,
                       100. * batch_idx / batches, loss.item()))

if __name__ == '__main__':

    # sent train_args to server
    agent_sock.send(train_args, 'train_args')

    train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                        # deterministic

    if train_args.cuda:
        torch.cuda.manual_seed(train_args.seed)  # set a random seed for the current GPU
        model_agent.cuda()  # move all model parameters to the GPU

    optimizer_agent = optim.SGD(model_agent.parameters(),
                          lr=train_args.lr,
                          momentum=train_args.momentum)

    for epoch in range(1, train_args.epochs + 1):
        train_epoch(epoch=epoch)

        # test_epoch()



