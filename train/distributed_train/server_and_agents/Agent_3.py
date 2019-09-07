
from train.distributed_train.agent import Agent

import os


cur_agent_name = 'agent_3'

server_host_port = ('localhost', 8082)
# server_host_port = ('10.1.1.13', 8082)


if __name__ == '__main__':

    os.chdir('../../../')

    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()







