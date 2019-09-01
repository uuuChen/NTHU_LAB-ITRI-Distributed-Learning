
from train.agent import Agent

import os


cur_agent_name = 'agent_3'

server_host_port = ('localhost', 8082)
# server_host_port = ('172.20.10.2', 8082)


if __name__ == '__main__':

    os.chdir('../../../')

    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()







