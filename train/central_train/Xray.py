# System Imports
from __future__ import print_function
from torch.autograd import Variable
import argparse
import torch.optim as optim
import numpy as np
import os
import io
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image
import time

# Model Imports
from model.VGGNet import *

# DataSet Imports
from dataSet.data_proc import data_processor

os.chdir('../../')

# training settings
parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 512)')

parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 30)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

train_args = parser.parse_args(args=[])


#plot
save_acc = open("record/Xray_acc.txt", "w")
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# train args
model = VGG('VGG19', 15)
train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()
torch.manual_seed(train_args.seed)
if train_args.cuda:
    torch.cuda.manual_seed(train_args.seed)  # set a random seed for the current GPU
    model.cuda()  # move all model parameters to the GPU

optim = optim.Adam(model.parameters(), lr=train_args.lr)

# dataSet
Xray_class_id = {
    'Hernia': 0,
    'Pneumonia': 1,
    'Fibrosis': 2,
    'Edema': 3,
    'Emphysema': 4,
    'Cardiomegaly': 5,
    'Pleural_Thickening': 6,
    'Consolidation': 7,
    'Pneumothorax': 8,
    'Mass': 9,
    'Nodule': 10,
    'Atelectasis': 11,
    'Effusion': 12,
    'Infiltration': 13,
    'No Finding': 14,
}

summary = [
    ['Hernia', 0],
    ['Pneumonia', 0],
    ['Fibrosis', 0],
    ['Edema', 0],
    ['Emphysema', 0],
    ['Cardiomegaly', 0],
    ['Pleural_Thickening', 0],
    ['Consolidation', 0],
    ['Pneumothorax', 0],
    ['Mass', 0],
    ['Nodule', 0],
    ['Atelectasis', 0],
    ['Effusion', 0],
    ['Infiltration', 0],
    ['No Finding', 0],
]
def get_dataSet(from_path):
    sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])

    image_file_names = os.listdir(from_path)
    image_file_names.sort(key=sort_key)
    image_file_nums = len(image_file_names)
    image_labels = []
    id = 0
    with open('data/Xray/labels.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if id == image_file_nums:
                break
            image_csv_name = row[0].split('_')
            image_file_name = image_file_names[id].split('_')
            while image_csv_name[0] == image_file_name[0] and image_csv_name[1].split('.')[0] == image_file_name[1].split('.')[0]:
                label = row[1]
                image_labels.append(label)
                id += 1
                if id == image_file_nums:
                    break
                image_file_name = image_file_names[id].split('_')

    dataSet = list(zip(image_file_names, image_labels))
    random.seed(3)
    random.shuffle(dataSet)
    list(zip(*dataSet))

    return dataSet


dataSet_id = 0
def get_data_label(dataSet, batch_size):
    global dataSet_id
    if dataSet_id+batch_size < len(dataSet):
        data = dataSet[dataSet_id:dataSet_id+batch_size]
        dataSet_id += batch_size
    else:
        data = dataSet[dataSet_id:]
        dataSet_id = 0

    return data

def _iter_epoch(is_training, epoch):

    batch_size = train_args.batch_size
    if is_training:
        from_path = 'data/Xray/sample_3'
        model.train()
    else:
        from_path = 'data/Xray/images_12'
        model.eval()
    dataSet = get_dataSet(from_path)

    data_nums = len(dataSet)
    trained_data_num = 0
    total_loss = 0
    correct = 0
    batches = (data_nums - 1) // batch_size + 1
    for batch_idx in range(1, batches + 1):
        data = get_data_label(dataSet, batch_size)

        datas = []
        targets = []
        image_size = (256, 256)
        for data_, target_ in data:
            data_ = os.path.join(from_path, data_)
            data_ = np.array(Image.open(data_).resize(image_size)) / 255
            data_ = data_.reshape(1, image_size[0], image_size[1])
            datas.append(data_)
            targets.append(Xray_class_id[target_])
        data, target = np.array(datas), np.array(targets)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data, target = Variable(data).float(), Variable(target).long()

        if train_args.cuda:
            data, target = data.cuda(), target.cuda()

        if is_training:
            optim.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)

        if is_training:
            loss.backward()
            optim.step()

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
    x = np.arange(1,  end_epoch)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(linestyle=":")
    plt.plot(x, np.array(train_loss), label='train')
    plt.plot(x, np.array(test_loss), label='test')
    plt.legend()
    plt.savefig("record/Xray_loss.png", dpi=300, format="png")

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Accuracy")
    plt.grid(linestyle=":")
    plt.plot(x, np.array(train_acc), label='train')
    plt.plot(x, np.array(test_acc), label='test')
    plt.legend()
    plt.savefig("record/Xray_acc.png", dpi=300, format="png")



record_time('開始時間 : ')

end_epoch = train_args.epochs + 1
best_correct = 0
check_count = 0
for epoch in range(1,  train_args.epochs + 1):
    print('{}/{}'.format(epoch, train_args.epochs))
    epoch = epoch

    _iter_epoch(is_training=True, epoch=epoch)
    correct = _iter_epoch(is_training=False, epoch=epoch)

    # early stopping
    if correct > best_correct:
        best_correct = correct
        check_count = 0

    else:
        check_count += 1

    if check_count > 10:
        print('\nEarly stop at epoch {}\n'.format(epoch))
        end_epoch = epoch + 1
        break

record_time('結束時間 : ')
save_acc.close()
plot_acc_loss(end_epoch)
