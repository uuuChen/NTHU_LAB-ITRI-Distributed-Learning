# System Imports
import argparse

# Server Imports
from train.server_agent_train.server import *

# training settings
parser = argparse.ArgumentParser()

parser.add_argument('--is-simulate', type=bool, default=True, metavar='N',
                    help='does the project use for accuracy simulation or not (actual hospitals usage) (default: True)')

parser.add_argument('--train-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')

parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')

parser.add_argument('--model', type=str, default='MLP', metavar='N',
                    help='training model (default: MLP)')

parser.add_argument('--agent-nums', type=int, default=2, metavar='N',
                    help='input agents number (default: 2)')

parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')

parser.add_argument('--dataSet', type=str, default='ECG', metavar='N',
                    help='training dataSet (default: ECG)')

parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                    help='learning rate (default: 1e-1)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

train_args = parser.parse_args(args=[])



if __name__ == '__main__':

    server = Server(train_args)
    server.start_training()






