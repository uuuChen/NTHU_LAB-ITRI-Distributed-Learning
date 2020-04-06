import sys
import os
sys.path.insert(0, os.getcwd())

# Server Imports
from train.distributed_train.server import *

# get args
data_name = sys.argv[1]
agent_nums = int(sys.argv[2])
use_localhost = False

if __name__ == '__main__':
    server = Server(data_name, agent_nums, use_localhost=use_localhost)
    server.start_training()








