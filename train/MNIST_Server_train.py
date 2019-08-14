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

os.chdir('../')

model_server = Server_LeNet()

# server socket setting
server_sock = Socket(SERVER_SOCKET_ARGS)

def train_once():

    model_server.train()
    optimizer_server.zero_grad()

    # data_nums = server_sock.recv('data_nums')  # get data_nums
    agent_output = server_sock.recv('agent_output')  # get agent_output
    target = server_sock.recv('target')  # get target
    # batch_idx = server_sock.recv('batch_idx')  # get batch index

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

    # send agent_output's gradient
    server_sock.send(agent_output_clone.grad.data, 'agent_output_clone_grad')
    server_sock.send(loss, 'loss')

# def test_epoch(conn):
#
#     model_server.eval()
#     model_agent.eval()
#
#     test_loss = 0
#
#     correct = 0
#
#     data_nums = test_dataSet.get_data_nums_from_database()
#
#     batches = (data_nums - 1) // train_args.batch_size + 1
#
#     datas, targets = test_dataSet.get_data_and_labels(batch_size=train_args.batch_size,
#                                                        one_hot=False)
#
#     for batch_idx in range(1, batches + 1):
#
#         data = datas[(batch_idx-1) * train_args.batch_size:batch_idx * train_args.batch_size]
#         target = targets[(batch_idx-1) * train_args.batch_size:batch_idx * train_args.batch_size]
#
#         if train_args.cuda:
#             data, target = data.cuda(), target.cuda()
#
#         data, target = Variable(data).float(), Variable(target).long()
#
#         agent_output = model_agent(data)
#         output = model_server(agent_output)
#
#         test_loss += F.cross_entropy(output, target).item()
#
#         pred = output.data.max(1)[1]
#
#         correct += pred.eq(target.data).cpu().sum()
#
#     test_loss /= batches
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, data_nums,
#         100. * correct / data_nums))

if __name__ == '__main__':

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
        train_once()

