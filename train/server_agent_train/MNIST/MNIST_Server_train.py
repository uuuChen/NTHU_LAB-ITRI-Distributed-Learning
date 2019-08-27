# System Imports
from __future__ import print_function
from torch.autograd import Variable
import argparse
import torch

# Model Imports
from model.LeNet import *

# DataSet Imports
from dataSet.MNIST_dataSet import *

# Socket Imports
from socket_.socket_ import *
import socket

# training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--agent-nums', type=int, default=2, metavar='N',
                    help='input agents number (default: 2)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
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

# initial the model
server_model = Server_LeNet()

# ==================================
# LocalHost testing
# ==================================
# server socket setting
server_port_begin = 8080
server_socks = []

# agent host port list for testing
agents_attrs = []
agent_port_begin = 2048

# stored data from agent
train_data_nums = [0, 0, 0, 0]
all_train_data_nums = 0
test_data_nums = [0, 0, 0, 0]
all_test_data_nums = 0

# training setting
is_first_training = True

# ==================================
# LAN testing
# ==================================
# localIP = socket.gethostbyname(socket.gethostname())
# server_sock = Socket((localIP, 8080), True)
# # agent host port list for testing
# cur_agent_idx = 1  # in this case, cur_agent_idx's range is 1 ~ 4
# agents_attrs = [
#     {
#
#     },
#     {
#         'name': 'agent_1',
#         'host_port': ('172.20.10.3', 2048)
#     },
#     {
#         'name': 'agent_2',
#         'host_port': ('172.20.10.3', 2049)
#     },
#     {
#         'name': 'agent_3',
#         'host_port': ('172.20.10.3', 2050)
#     },
#     {
#         'name': 'agent_4',
#         'host_port': ('172.20.10.3', 2051)
#     }
# ]
# agent_nums = len(agents_attrs) - 1
def conn_to_agents():

    for i in range(train_args.agent_nums):
        server_sock = Socket(('localhost', server_port_begin + i), True)
        server_socks.append(server_sock)

    for i in range(train_args.agent_nums):
        server_socks[i].accept()
        agents_attr = {
            'name': 'agent_' + str(i+1),
            'host_port': (server_socks[i].addr[0], agent_port_begin + i)
        }
        agents_attrs.append(agents_attr)

def get_prev_next_agent(agent_idx):

    prev_agent_idx = agent_idx - 1
    next_agent_idx = agent_idx + 1

    if prev_agent_idx == -1:
        prev_agent_idx = train_args.agent_nums - 1
    if next_agent_idx == train_args.agent_nums:
        next_agent_idx = 0

    prev_agent = agents_attrs[prev_agent_idx]
    next_agent = agents_attrs[next_agent_idx]

    return prev_agent, next_agent

def get_data_nums():
    global all_train_data_nums
    global all_test_data_nums

    for i in range(train_args.agent_nums):

        train_data_nums[i] += server_socks[i].recv('data_nums')
        all_train_data_nums += train_data_nums[i]

        test_data_nums[i] += server_socks[i].recv('data_nums')
        all_test_data_nums += test_data_nums[i]

def send_train_args():

    for i in range(train_args.agent_nums):
        # send train args to agent
        server_socks[i].send(train_args, 'train_args')

def is_training_done(flag):

    for i in range(train_args.agent_nums):
        server_socks[i].send(flag, 'is_training_done')

def train_with_cur_agent(agent_idx, epoch, trained_data_num):

    data_nums = train_data_nums[agent_idx]
    batches = (data_nums - 1) // train_args.batch_size + 1

    for batch_idx in range(1, batches + 1):
        optimizer_server.zero_grad()

        # get agent_output and target from agent
        agent_output = server_socks[agent_idx].recv('agent_output')
        target = server_socks[agent_idx].recv('target')

        # store gradient in agent_output_clone
        agent_output_clone = Variable(agent_output).float()
        if train_args.cuda:
            agent_output_clone = agent_output_clone.cuda()
            target = target.cuda()
        agent_output_clone.requires_grad_()

        # server forward
        server_output = server_model(agent_output_clone)
        loss = F.cross_entropy(server_output, target)

        # server backward
        loss.backward()
        optimizer_server.step()

        # send gradient to agent
        server_socks[agent_idx].send(agent_output_clone.grad.data, 'agent_output_clone_grad')

        trained_data_num += len(target)
        if batch_idx % train_args.log_interval == 0:
            print('Train Epoch: {} at {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, agents_attrs[agent_idx], trained_data_num, all_train_data_nums,
                100. * batch_idx / batches, loss.item()))

    return trained_data_num

def train_epoch(epoch):

    global is_first_training

    server_model.train()

    trained_data_num = 0
    for i in range(train_args.agent_nums):

        # get previous and next agent attributes
        prev_agent_attrs, next_agent_attrs = get_prev_next_agent(i)

        # send prev_agent_attrs, next_agent_attrs to agent
        if is_first_training:
            prev_agent_attrs = None
        server_socks[i].send((prev_agent_attrs, next_agent_attrs), 'prev_next_agent_attrs')

        # VERY IMPORTANT !!! server is waiting for previos agent sending model snapshot to current agent
        if not is_first_training:
            server_socks[i].sleep()

        is_first_training = False

        print('starting training with' + str(agents_attrs[i]))
        trained_data_num = train_with_cur_agent(i, epoch, trained_data_num)


def test_with_cur_agent(agent_idx, test_loss, correct):

    data_nums = test_data_nums[agent_idx]
    batches = (data_nums - 1) // train_args.batch_size + 1
    for batch_idx in range(1, batches + 1):

        # get agent_output and target from agent
        agent_output = server_socks[agent_idx].recv('agent_output')
        target = server_socks[agent_idx].recv('target')

        agent_output_clone = Variable(agent_output).float()

        if train_args.cuda:
            agent_output_clone = agent_output_clone.cuda()
            target = target.cuda()

        # server forward
        server_output = server_model(agent_output_clone)
        loss = F.cross_entropy(server_output, target)

        test_loss += loss

        pred = server_output.data.max(1)[1]

        correct += pred.eq(target.data).cpu().sum()

    return test_loss, correct


def test_epoch():

    server_model.eval()

    test_loss = 0
    correct = 0
    batches = 0

    for i in range(train_args.agent_nums):

        # get previous and next agent attributes
        prev_agent_attrs, next_agent_attrs = get_prev_next_agent(i)

        # send prev_agent_attrs, next_agent_attrs to agent
        server_socks[i].send((prev_agent_attrs, next_agent_attrs), 'prev_next_agent_attrs')
        server_socks[i].sleep()

        test_loss, correct = test_with_cur_agent(i, test_loss, correct)
        batches += (test_data_nums[i] - 1) // train_args.batch_size + 1

        if epoch is train_args.epochs - 1:
            server_socks[i].send(True, 'is_training_done')
        else:
            server_socks[i].send(False, 'is_training_done')

    test_loss /= batches

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, all_test_data_nums,
        100. * correct / all_test_data_nums))



if __name__ == '__main__':

    # training setup
    train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                        # deterministic
    if train_args.cuda:
        torch.cuda.manual_seed(train_args.seed)  # set a random seed for the current GPU
        server_model.cuda()  # move all model parameters to the GPU
    optimizer_server = optim.SGD(server_model.parameters(),
                                 lr=train_args.lr,
                                 momentum=train_args.momentum)

    conn_to_agents()
    get_data_nums()
    send_train_args()

    for epoch in range(train_args.epochs):

        # start training and testing
        train_epoch(epoch=epoch)
        test_epoch()






