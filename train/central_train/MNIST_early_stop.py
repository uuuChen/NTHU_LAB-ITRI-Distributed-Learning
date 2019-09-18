from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from model.LeNet import *
import argparse
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import time
import os

os.chdir('../../')

# train_args
parser = argparse.ArgumentParser()

parser.add_argument('--train-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')

parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')

parser.add_argument('--epochs', type=int, default=50, metavar='N',
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

train_args = parser.parse_args(args=[])

# model, dataSet
model = LeNet()
data_train = datasets.MNIST(root="./data/",
                                    transform=transforms.ToTensor(),
                                    train=True,
                                    download=True)
data_test = datasets.MNIST(root="./data/",
                                   transform=transforms.ToTensor(),
                                   train=False)

# record
save_acc = open("record/MNIST_early_stopping_acc.txt", "w")
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# cuda setting
train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()
if train_args.cuda:
    torch.cuda.manual_seed(train_args.seed)  # set a random seed for the current GPU
    model.cuda()  # move all model parameters to the GPU
    optimizer = optim.SGD(model.parameters(), lr=train_args.lr,  momentum=train_args.momentum)

# early stopping
check_correct = 0
check_count = 0
model_save_dir = 'record/'

def _iter_epoch(epoch, is_training):
    if is_training:
        model.train()
        images = data_train.data.tolist()
        labels = data_train.targets.tolist()
        batch_size = train_args.train_batch_size
    else:
        model.eval()
        images = data_test.data.tolist()
        labels = data_test.targets.tolist()
        batch_size = train_args.test_batch_size

    data_nums = len(images)

    trained_data_num = 0
    total_loss = 0
    correct = 0
    batches = (data_nums - 1) // batch_size + 1
    for batch_idx in range(1, batches + 1):
        if batch_idx != batches + 1:
            data = images[(batch_idx-1)*batch_size:batch_idx*batch_size]
            target = labels[(batch_idx-1)*batch_size:batch_idx*batch_size]
        else:
            data = images[(batch_idx-1)*batch_size:]
            target = labels[(batch_idx-1)*batch_size:]

        data, target = np.array(data), np.array(target)
        datas = []
        for data_ in data:
            data_ = data_.reshape(1, 28, 28)
            datas.append(data_)
        data = np.array(datas)
        target = np.array(target)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data, target = Variable(data).float(), Variable(target).long()
        if train_args.cuda:
            data, target = data.cuda(), target.cuda()
        if is_training:
            optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        if is_training:
            loss.backward()
            optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        total_loss += loss.item()

        if is_training:
            trained_data_num += data.shape[0]
            if batch_idx % train_args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, trained_data_num, data_nums, 100. * batch_idx / batches, loss.item()))

    total_loss /= batches
    if is_training:
        train_acc.append(100. * correct / data_nums)
        train_loss.append(total_loss)
        save_acc.write('Epoch {} \r\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r\n'.format(
            epoch, total_loss, correct, data_nums, 100. * correct / data_nums))
    else:
        test_acc.append(100. * correct / data_nums)
        test_loss.append(total_loss)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            total_loss, correct, data_nums, 100. * correct / data_nums))
        save_acc.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r\n\n'.format(
            total_loss, correct, data_nums, 100. * correct / data_nums))
        return correct



def record_time(hint):
    localtime = time.asctime( time.localtime(time.time()) )
    save_acc.write(hint + localtime + '\r\n\n')

def plot_acc_loss(end_epoch):
    x = np.arange(1, end_epoch + 1)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(linestyle=":")
    plt.plot(x, np.array(train_loss), label='train')
    plt.plot(x, np.array(test_loss), label='test')
    plt.legend()
    plt.savefig("record/early_stopping_loss.png", dpi=300, format="png")

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Accuracy")
    plt.grid(linestyle=":")
    plt.plot(x, np.array(train_acc), label='train')
    plt.plot(x, np.array(test_acc), label='test')
    plt.legend()
    plt.savefig("record/early_stopping_acc.png", dpi=300, format="png")

if __name__ == '__main__':
    record_time('開始時間 : ')
    end_epoch = 0
    for epoch in range(1,  train_args.epochs + 1):
        epoch = epoch
        _iter_epoch(epoch, is_training=True)
        correct = _iter_epoch(epoch, is_training=False)

        if correct > check_correct:
            check_correct = correct
            check_count = 0
        else:
            check_count += 1

        if check_count > 5:
            print('\nEarly stop at epoch {}\n'.format(epoch))
            end_epoch = epoch
            break

    record_time('結束時間 : ')
    save_acc.close()
    plot_acc_loss(end_epoch)









