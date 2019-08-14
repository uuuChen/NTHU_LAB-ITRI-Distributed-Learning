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
# agent socket setting
ip_port=('localhost',8080)
back_log=5
buffer_size=4096

tcp_client=socket(AF_INET,SOCK_STREAM)
tcp_client.connect(ip_port)

# training settings
parser = argparse.ArgumentParser()
train_args = parser.parse_args(args=[])

train_dataSet = MNIST_DataSet(data_args=MNIST_TRAIN_ARGS,
                              shuffle=True)

model_agent = Agent_LeNet()

def train_epoch(epoch):

    model_agent.train()
    data_nums = train_dataSet.get_data_nums_from_database()

    # send data_nums to server
    msg = pickle.dumps(data_nums)
    while 1:
        try:
            tcp_client.send(msg)
            print("send data_nums:",data_nums)
            break
        except Exception:
            break

    batches = (data_nums - 1) // train_args.batch_size + 1

    datas, targets = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size,
                                                   one_hot=False)

    for batch_idx in range(1, batches + 1):
        if batch_idx * train_args.batch_size < len(datas):
            data = datas[(batch_idx-1) * train_args.batch_size:batch_idx * train_args.batch_size]
            target = targets[(batch_idx-1) * train_args.batch_size:batch_idx * train_args.batch_size]
        else:
            data = datas[(batch_idx-1) * train_args.batch_size:len(datas)]
            target = targets[(batch_idx-1) * train_args.batch_size:len(datas)]

        if train_args.cuda:
            data = data.cuda()

        data, target = Variable(data).float(), Variable(target).long()

        #agent forward
        optimizer_agent.zero_grad()
        agent_output = model_agent(data)

        # send agent_output
        msg = pickle.dumps(agent_output)
        while 1:
            try:
                tcp_client.send(msg)
                print("send agent_output")
                break
            except Exception:
                break

        # send target
        msg = pickle.dumps(target)
        while 1:
            try:
                tcp_client.send(msg)
                print("send target")
                break
            except Exception:
                break

        # get agent_output_clone
        while 1:
            try:
                data = tcp_client.recv(1000000)
                agent_output_clone_gradient = pickle.loads(data)
                print("get agent_output_clone_gradient:")
                break
            except Exception:
                break

        #agent backward
        agent_output.backward(gradient=agent_output_clone_gradient)
        optimizer_agent.step()

if __name__ == '__main__':
    # get train_args from server
    while 1:
        try:
            data = tcp_client.recv(buffer_size)
            train_args = pickle.loads(data)
            print("get train_args")
            break
        except Exception:
            break

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
        # send model_agent
        msg = pickle.dumps(model_agent)
        while 1:
            try:
                tcp_client.send(msg)
                print("send model_agent")
                break
            except Exception:
                break



