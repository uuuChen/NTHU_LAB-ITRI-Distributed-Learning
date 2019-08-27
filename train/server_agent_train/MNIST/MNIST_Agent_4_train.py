# Model Imports
from model.LeNet import *

# DataSet Imports
from dataSet.MNIST_dataSet import *
from data.data_args import *  # import data arguments

# Socket Imports
from train.server_agent_train.agent import Agent

os.chdir('../../../')

# training settings
train_dataSet = MNIST_DataSet(data_args=MNIST_TRAIN_ARGS,
                              shuffle=True)
test_dataSet = MNIST_DataSet(data_args=MNIST_TEST_ARGS,
                             shuffle=True)
model_agent = Agent_LeNet()

cur_agent_name = 'agent_4'

# ==================================
# LocalHost testing
# ==================================
server_host_port = ('localhost', 8083)
cur_host_port = ('localhost', 2051)

# ==================================
# LAN testing
# ==================================
# server_host_port = ('172.20.10.2', 8080)
# cur_host_port = ('172.20.10.3', 8083)


if __name__ == '__main__':

    agent = Agent(model_agent, train_dataSet, test_dataSet, server_host_port, cur_host_port, cur_agent_name)
    agent.start_training()







