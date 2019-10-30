import sys
import os
sys.path.insert(0, os.getcwd())

from train.distributed_train.agent import Agent
import time

# get args
cur_agent_num = sys.argv[1]
server_host = sys.argv[2]
cur_agent_name = 'agent_'+cur_agent_num

# server_host_port = ('localhost', 8080+int(cur_agent_num)-1)
server_host_port = (server_host, 8080+int(cur_agent_num)-1)
dataSet_num = 4

if __name__ == '__main__':

    # for i in range(1, dataSet_num + 1):
    agent = Agent(server_host_port, cur_agent_name)
    agent.start_training()
        # if i != dataSet_num:
        #     time.sleep(100)








