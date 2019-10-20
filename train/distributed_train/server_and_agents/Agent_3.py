
from train.distributed_train.agent import Agent

import os


cur_agent_name = 'agent_3'

# server_host_port = ('localhost', 8082)
server_host_port = ('192.168.0.189', 8082)
save_path = "record/agent/10_20/"


if __name__ == '__main__':

    os.chdir('../../../')

    agent = Agent(server_host_port, cur_agent_name, save_path)
    agent.start_training()







