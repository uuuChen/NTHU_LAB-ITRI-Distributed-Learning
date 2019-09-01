# Model Imports
from model.LeNet import *

# DataSet Imports
# from data.data_args import *  # import data arguments

from train.server_agent_train.agent import Agent

import os

os.chdir('../../../')

# training settings
model_agent = Agent_LeNet()

cur_agent_name = 'agent_2'

# ==================================
# LocalHost testing
# ==================================
server_host_port = ('localhost', 8081)

# ==================================
# LAN testing
# ==================================
# server_host_port = ('172.20.10.2', 8081)


if __name__ == '__main__':

    agent = Agent(model_agent, server_host_port, cur_agent_name)
    agent.start_training()







