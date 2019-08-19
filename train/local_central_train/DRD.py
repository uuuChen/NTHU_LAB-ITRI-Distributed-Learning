
# System Imports
from __future__ import print_function
import argparse
import torch.optim as optim
from torch.autograd import Variable

# Model Imports
from model.AlexNet import *

# DataSet Imports
from dataSet.DRD_dataSet import *
from data.data_args import *  # import data arguments

os.chdir('../../')

# training settings
parser = argparse.ArgumentParser()

parser.add_argument('--train-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')

parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')

parser.add_argument('--image-size', type=int, default=(100, 100), metavar='N',
                    help='image size (width, height) for training and testing (default: (100, 100))', nargs='+')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

train_args = parser.parse_args(args=[])

# -------------------------------------
# use all of the data to train
# -------------------------------------
# train_dataSet = DRD_DataSet(data_args=DRD_TRAIN_ARGS,
#                             shuffle=True)
#
# test_dataSet = DRD_DataSet(data_args=DRD_TEST_ARGS,
#                            shuffle=True)

# -------------------------------------
# use a small amount of data to train
# -------------------------------------
train_dataSet = DRD_DataSet(data_args=DRD_TESTING_ARGS,
                            shuffle=False)

test_dataSet = DRD_DataSet(data_args=DRD_TESTING_ARGS,
                           shuffle=False)

model = AlexNet()

optimizer = optim.SGD(model.parameters(),
                      lr=train_args.lr,
                      momentum=train_args.momentum)

def train_epoch(epoch):

    model.train()

    data_nums = train_dataSet.get_data_nums_from_database()

    train_batch_size = train_args.train_batch_size

    batches = (data_nums - 1) // train_batch_size + 1

    for batch_idx in range(1, batches + 1):

        data, target = train_dataSet.get_data_and_labels(batch_size=train_batch_size,
                                                         image_size=train_args.image_size)

        if train_args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data).float(), Variable(target).long()

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % train_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train_batch_size, batches * train_batch_size, 100. * batch_idx / batches,
                loss.item()))

def test_epoch():

    model.eval()

    test_batch_size = train_args.test_batch_size

    data, target = test_dataSet.get_data_and_labels(batch_size=test_batch_size,
                                                    image_size=train_args.image_size)

    if train_args.cuda:
        data, target = data.cuda(), target.cuda()

    data, target = Variable(data).float(), Variable(target).long()

    output = model(data)

    test_loss = F.nll_loss(output, target).item()

    pred = output.data.max(1)[1]

    correct = pred.eq(target.data).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_batch_size,
        100. * correct / test_batch_size))

if __name__ == '__main__':

    train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                        # deterministic

    if train_args.cuda:
        torch.cuda.manual_seed(train_args.seed)  # set a random seed for the current GPU
        model.cuda()  # move all model parameters to the GPU

    for epoch in range(1, train_args.epochs + 1):

        train_epoch(epoch=epoch)

        test_epoch()


