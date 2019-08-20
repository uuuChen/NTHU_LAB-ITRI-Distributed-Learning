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

os.chdir('../../../')

# training settings
train_dataSet = MNIST_DataSet(data_args=MNIST_TRAIN_ARGS,
                              shuffle=True)
test_dataSet = MNIST_DataSet(data_args=MNIST_TEST_ARGS,
                             shuffle=True)
model_agent = Agent_LeNet()

cur_host_port = ('localhost', 2051)

# build current agent server. it's used to send model snapshot
to_agent_sock = Socket(cur_host_port, True)

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

        break


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

    while True:

        # connect to server, agent and server socket setting
        agent_server_sock = Socket(('localhost', 8080), False)
        agent_server_sock.connect()

        if agent_server_sock.is_right_conn(client_name='agent_4'):

            # receive previous, next agents from server
            prev_agent_attrs, next_agent_attrs = agent_server_sock.recv('prev_next_agent_attrs')

            # connect to last training agent and get model snapshot. prev_agent_attrs is None when the first training
            if prev_agent_attrs is not None:
                from_agent_sock = Socket(prev_agent_attrs['host_port'], False)
                from_agent_sock.connect()
                model_agent = from_agent_sock.recv('model_agent')
                from_agent_sock.close()

                # VERY IMPORTANT !!! awake server after current agent receiving model snapshot from previous agent
                agent_server_sock.awake()

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

            # train an epoch with server
            train_epoch()
            test_epoch()
            agent_server_sock.close()

            # send model to next agent
            to_agent_sock.accept()
            to_agent_sock.send(model_agent, 'model_agent')
            to_agent_sock.close()

            print('agent_4 done')








