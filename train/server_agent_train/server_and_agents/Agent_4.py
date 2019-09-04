
from train.server_agent_train.agent import Agent

import os


cur_agent_name = 'agent_4'

server_host_port = ('localhost', 8083)
# server_host_port = ('10.1.1.13', 8083)


if __name__ == '__main__':

    os.chdir('../../../')

    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()







