
from train.agent import Agent

import os

# training settings

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

    os.chdir('../../../')

    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()







