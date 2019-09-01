# Model Imports
from model.LeNet import *

from train.agent import Agent

import os

os.chdir('../../../')

# training settings
model_agent = Agent_LeNet()

cur_agent_name = 'agent_4'

# ==================================
# LocalHost testing
# ==================================
server_host_port = ('localhost', 8083)

# ==================================
# LAN testing
# ==================================
# server_host_port = ('172.20.10.2', 8083)


if __name__ == '__main__':

    agent = Agent(server_host_port, cur_agent_name, model=model_agent)
    agent.start_training()







