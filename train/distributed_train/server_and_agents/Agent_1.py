
from train.distributed_train.agent import Agent

import os

cur_agent_name = 'agent_1'

server_host_port = ('localhost', 8080)
server_host_port = ('192.168.0.189', 8080)


if __name__ == '__main__':

    os.chdir('../../../')

    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()







