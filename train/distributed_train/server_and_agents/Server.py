
# Server Imports
from train.distributed_train.server import *
import sys
import os
sys.path.insert(0, os.getcwd())

# get args
data_name = sys.argv[1]
agent_nums = int(sys.argv[2])

if __name__ == '__main__':

    os.chdir('../../../')
    server = Server(data_name=data_name, agent_nums=agent_nums, use_localhost=False)
    server.start_training()






