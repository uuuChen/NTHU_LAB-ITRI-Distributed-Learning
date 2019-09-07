
from train.distributed_train.agent import Agent

import os

cur_agent_name = 'agent_2'

server_host_port = ('localhost', 8081)
# server_host_port = ('10.1.1.13', 8081)


if __name__ == '__main__':

    os.chdir('../../../')

    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()







