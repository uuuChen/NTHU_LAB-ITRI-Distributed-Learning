
import argparse

# --------------------------------
#  MNIST
# --------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--is-simulate', type=bool, default=True, metavar='N',
                    help='does the project use for accuracy simulation or not (actual hospitals usage) (default: True)')
parser.add_argument('--train-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--agent-nums', type=int, default=2, metavar='N',
                    help='input agents number (default: 2)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
MNIST_TRAINING_ARGS = parser.parse_args(args=[])


# --------------------------------
#  ECG
# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--is-simulate', type=bool, default=True, metavar='N',
                    help='does the project use for accuracy simulation or not (actual hospitals usage) (default: True)')
parser.add_argument('--train-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--agent-nums', type=int, default=2, metavar='N',
                    help='input agents number (default: 2)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
ECG_TRAINING_ARGS = parser.parse_args(args=[])


# --------------------------------
# OCT
# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--is-simulate', type=bool, default=True, metavar='N',
                    help='does the project use for accuracy simulation or not (actual hospitals usage) (default: True)')
parser.add_argument('--train-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--agent-nums', type=int, default=4, metavar='N',
                    help='input agents number (default: 2)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--image-size', type=int, default=(256, 256), metavar='N',
                    help='image size (width, height) for training and testing (default: (256, 256))', nargs='+')
OCT_TRAINING_ARGS = parser.parse_args(args=[])

# --------------------------------
#  MC
# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--is-simulate', type=bool, default=True, metavar='N',
                    help='does the project use for accuracy simulation or not (actual hospitals usage) (default: True)')
parser.add_argument('--train-batch-size', type=int, default=384, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=384, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--image-size', type=int, default=(100, 100), metavar='N',
                    help='image size (width, height) for training and testing (default: (100, 100))', nargs='+')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--agent-nums', type=int, default=2, metavar='N',
                    help='input agents number (default: 2)')
MC_TRAINING_ARGS = parser.parse_args(args=[])