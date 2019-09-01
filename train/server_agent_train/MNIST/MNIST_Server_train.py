
# Server Imports
from train.server import *
import os

os.chdir('../../../')

if __name__ == '__main__':

    server = Server(data_name='MNIST')
    server.start_training()






