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

model_server = Server_LeNet()

# server socket setting
server_sock = Socket(SERVER_SOCKET_ARGS)

def iter_once():

    # get things from agent
    is_training = server_sock.recv('is_training')
    agent_output = server_sock.recv('agent_output')  # get agent_output
    target = server_sock.recv('target')  # get target

    # server model setup
    if is_training:
        model_server.train()
    else:
        model_server.eval()
    optimizer_server.zero_grad()
    agent_output_clone = Variable(agent_output).float()
    agent_output_clone.requires_grad_()
    if train_args.cuda:
        agent_output_clone = agent_output_clone.cuda()
        target = target.cuda()

    #server forward
    server_output = model_server(agent_output_clone)
    loss = F.cross_entropy(server_output, target)

    #server backward
    if is_training:
        loss.backward()
        optimizer_server.step()

    # send things to agent
    if is_training:
        server_sock.send(agent_output_clone.grad.data, 'agent_output_clone_grad')
    else:
        server_sock.send(server_output, 'server_output')
    server_sock.send(loss, 'loss')

if __name__ == '__main__':

    # wait for agent connect
    server_sock.accept()

    # get train args from agent
    train_args = server_sock.recv('train_args')

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

    while True:
        iter_once()

