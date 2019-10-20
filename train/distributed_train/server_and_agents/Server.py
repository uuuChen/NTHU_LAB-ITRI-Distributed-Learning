
# Server Imports
from train.distributed_train.server import *
import os

# data_name = 'MNIST'
data_name = 'ECG'
# data_name = 'DRD'
# data_name = 'Xray'
# data_name = 'OCT'
save_path = "record/server/10_20/"

if __name__ == '__main__':

    os.chdir('../../../')
    server = Server(data_name=data_name, save_path=save_path, use_localhost=False)
    server.start_training()






