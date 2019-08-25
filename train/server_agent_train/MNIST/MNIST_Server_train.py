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
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
server_sock = Socket(('localhost', 8080), True)

# agent host port list for testing
cur_agent_idx = 1  # in this case, cur_agent_idx's range is 1 ~ 4
agents_attrs = [
    {

    },
    {
        'name': 'agent_1',
        'host_port': ('localhost', 2048)
    },
    {
        'name': 'agent_2',
        'host_port': ('localhost', 2049)
    },
    {
        'name': 'agent_3',
        'host_port': ('localhost', 2050)
    },
    {
        'name': 'agent_4',
        'host_port': ('localhost', 2051)
    }
]
agent_nums = len(agents_attrs) - 1

# ==================================
# LAN testing
# ==================================
# localIP = socket.gethostbyname(socket.gethostname())
# server_sock = Socket((localIP, 8080), True)
#
# # agent host port list for testing
# cur_agent_idx = 1  # in this case, cur_agent_idx's range is 1 ~ 4
# agents_attrs = [
#     {
#
#     },
#     {
#         'name': 'agent_1',
#         'host_port': ('10.1.1.11', 2048)
#     },
#     {
#         'name': 'agent_2',
#         'host_port': ('10.1.1.11', 2049)
#     },
#     {
#         'name': 'agent_3',
#         'host_port': ('10.1.1.11', 2050)
#     },
#     {
#         'name': 'agent_4',
#         'host_port': ('10.1.1.11', 2051)
#     }
# ]
# agent_nums = len(agents_attrs) - 1


def trans_to_next_agent_idx():

    global cur_agent_idx

    cur_agent_idx += 1
    if cur_agent_idx == agent_nums + 1:
        cur_agent_idx = 1


def get_prev_next_agent():

    global cur_agent_idx

    prev_agent_idx = cur_agent_idx - 1
    next_agent_idx = cur_agent_idx + 1

    if prev_agent_idx == 0:
        prev_agent_idx = agent_nums
    if next_agent_idx == agent_nums + 1:
        next_agent_idx = 1

    prev_agent = agents_attrs[prev_agent_idx]
    next_agent = agents_attrs[next_agent_idx]

    return prev_agent, next_agent


def get_cur_agent_name():
    return agents_attrs[cur_agent_idx]['name']


def train_epoch(epoch):

    server_model.train()
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
        server_sock.send(agent_output_clone.grad.data, 'agent_output_clone_grad')

        trained_data_num += len(target)
        if batch_idx % train_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, trained_data_num, data_nums,
                       100. * batch_idx / batches, loss.item()))

        # break


def test_epoch():

    server_model.eval()

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
        server_output = server_model(agent_output_clone)
        loss = F.cross_entropy(server_output, target)

        test_loss += loss

        pred = server_output.data.max(1)[1]

        correct += pred.eq(target.data).cpu().sum()

    test_loss /= batches

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data_nums,
        100. * correct / data_nums))


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

    epoch = 1
    is_first_training = True
    is_training_done = False
    socket_closed_agents_nums = 0

    while True:

        # wait for current training agent connect. Keep waiting if it's not connected by current training agent
        server_sock.accept()
        if not server_sock.is_right_conn(client_name=get_cur_agent_name()):
            continue

        if not is_training_done:
            # tell agents training is not done
            server_sock.send(False, 'is_training_done')

            # get previous and next agent attributes
            prev_agent_attrs, next_agent_attrs = get_prev_next_agent()

            # send prev_agent_attrs, next_agent_attrs to agent
            if is_first_training:
                prev_agent_attrs = None
            server_sock.send((prev_agent_attrs, next_agent_attrs), 'prev_next_agent_attrs')

            # VERY IMPORTANT !!! server is waiting for previos agent sending model snapshot to current agent
            if not is_first_training:
                server_sock.sleep()

            # send train args to agent
            server_sock.send(train_args, 'train_args')

            # start training and testing
            train_epoch(epoch=epoch)
            test_epoch()

            # set some training attributes and send current epoch
            epoch += 1
            is_first_training = False

            # determine whether training is done
            if epoch == (train_args.epochs + 1):
                is_training_done = True  # it will close current agent's socket and tell it not to send model snapshot
                socket_closed_agents_nums += 1

        else:
            socket_closed_agents_nums += 1
            if socket_closed_agents_nums == agent_nums:
                server_sock.close()
                break

        # transfer to next training agent index
        trans_to_next_agent_idx()

        # tell current agent whether training is done, and does it need to send model snapshot to next agent or not
        server_sock.send(is_training_done, 'is_training_done')

        print('trans to agent ' + str(get_cur_agent_name()))





