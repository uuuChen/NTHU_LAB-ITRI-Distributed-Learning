
# Server Imports
from train.server import *
import os

data_name = 'MNIST'

if __name__ == '__main__':

    os.chdir('../../../')

    server = Server(data_name=data_name)
    server.start_training()






