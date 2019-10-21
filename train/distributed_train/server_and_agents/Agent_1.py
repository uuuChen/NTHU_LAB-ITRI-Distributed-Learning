
from train.distributed_train.agent import Agent
import sys
import os
sys.path.insert(0, os.getcwd())

# get args
cur_agent_num = sys.argv[1]
server_host = sys.argv[2]
cur_agent_name = 'agent_'+cur_agent_num

# server_host_port = ('localhost', 8080)
server_host_port = (server_host, 8080+int(cur_agent_num)-1)


if __name__ == '__main__':

    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()







