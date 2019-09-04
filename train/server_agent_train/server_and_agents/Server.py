
# Server Imports
from train.server_agent_train.server import *
import os

data_name = 'MNIST'
# data_name = 'ECG'
# data_name = 'DRD'

if __name__ == '__main__':

    os.chdir('../../../')
    server = Server(data_name=data_name, use_localhost=True)
    server.start_training()






