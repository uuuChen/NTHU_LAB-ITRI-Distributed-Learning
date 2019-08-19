# System Imports
from __future__ import print_function
from torch.autograd import Variable

# Model Imports
from model.LeNet import *

# DataSet Imports
from dataSet.MNIST_dataSet import *
from data.data_args import *  # import data arguments

# Socket Imports
from socket_.socket_ import *
from socket_.socket_args import *

os.chdir('../../../')

# agent socket_ setting
from_agent_sock = Socket(('localhost', 80), False)
agent_server_sock = Socket(('localhost', 8080), False)

# training settings
train_dataSet = MNIST_DataSet(data_args=MNIST_TRAIN_ARGS,
                              shuffle=True)

test_dataSet = MNIST_DataSet(data_args=MNIST_TEST_ARGS,
                             shuffle=True)

model_agent = Agent_LeNet()

def train_epoch():
    model_agent.train()
    data_nums = train_dataSet.get_data_nums_from_database()
    agent_server_sock.send(data_nums, 'data_nums')
    batches = (data_nums - 1) // train_args.batch_size + 1

    for batch_idx in range(1, batches + 1):

        data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size)

        if train_args.cuda:
            data = data.cuda()

        data, target = Variable(data).float(), Variable(target).long()

        # agent forward
        optimizer_agent.zero_grad()
        agent_output = model_agent(data)

        # send agent_output and target to server
        agent_server_sock.send(agent_output, 'agent_output')  # send agent_output
        agent_server_sock.send(target, 'target')  # send target

        # receive gradient from server
        agent_output_grad = agent_server_sock.recv('agent_output_clone_grad')  # get agent_output_clone

        # agent backward
        agent_output.backward(gradient=agent_output_grad)
        optimizer_agent.step()


def test_epoch():
    model_agent.eval()

    data_nums = test_dataSet.get_data_nums_from_database()
    agent_server_sock.send(data_nums, 'data_nums')
    batches = (data_nums - 1) // train_args.batch_size + 1

    for batch_idx in range(1, batches + 1):

        data, target = test_dataSet.get_data_and_labels(batch_size=train_args.batch_size)

        if train_args.cuda:
            data = data.cuda()

        data, target = Variable(data).float(), Variable(target).long()

        agent_output = model_agent(data)

        # send agent_output and target to server
        agent_server_sock.send(agent_output, 'agent_output')  # send agent_output
        agent_server_sock.send(target, 'target')  # send target


if __name__ == '__main__':

    # connect to last training agent
    from_agent_sock.connect()

    model_agent = from_agent_sock.recv('model_agent')

    from_agent_sock.close()

    # connect to server
    agent_server_sock.connect()

    # receive train_args from server
    train_args = agent_server_sock.recv('train_args')

    train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(train_args.seed)  # seeding the CPU for generating random numbers so that the results are
    # deterministic

    if train_args.cuda:
        torch.cuda.manual_seed(train_args.seed)  # set a random seed for the current GPU
        model_agent.cuda()  # move all model parameters to the GPU

    optimizer_agent = optim.SGD(model_agent.parameters(),
                                lr=train_args.lr,
                                momentum=train_args.momentum)

    # for epoch in range(1, train_args.epochs + 1):
    train_epoch()
    test_epoch()

    agent_server_sock.close()








