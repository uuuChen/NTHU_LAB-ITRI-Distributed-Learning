from train.switch import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import csv
import random
from PIL import Image


class Central_Train:

    def __init__(self):
        pass

    def _build(self, data_name):
        self.save_acc = open("record/" + data_name + "_central_acc.txt", "w")

        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        self.data_name = data_name
        self.switch = Switch(data_name=data_name)
        self.train_args = self.switch.get_train_args()
        self.train_data_paths_and_labels = self.get_data_paths_and_labels('data/DRD_data/train',
                                                                          'data/DRD_data/trainLabels.csv')
        self.test_data_paths_and_labels = self.get_data_paths_and_labels('data/DRD_data/sample',
                                                                         'data/DRD_data/trainLabels.csv')
        self.train_data_pointer = 0
        self.test_data_pointer = 0
        self.image_size = (100, 100)
        self.model = self.switch.get_model(is_central=True)
        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                                 # deterministic

        if self.train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.model.cuda()  # move all model parameters to the GPU

        self.optim = optim.Adam(self.model.parameters(), lr=self.train_args.lr)

    def get_data_paths_and_labels(self, images_dir_path, csv_file_path):
        image_file_names = os.listdir(images_dir_path)
        sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])
        image_file_names.sort(key=sort_key)
        image_file_nums = len(image_file_names)
        image_labels = []
        read_file_nums = 0
        with open(csv_file_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if read_file_nums == image_file_nums:
                    break
                image_csv_name = row[0].split('_')
                image_file_name = image_file_names[read_file_nums].split('_')
                while image_csv_name[0] == image_file_name[0] and image_csv_name[1].split('.')[0] == \
                    image_file_name[1].split('.')[0]:
                    label = int(row[1])
                    image_labels.append(label)
                    read_file_nums += 1
                    if read_file_nums == image_file_nums:
                        break
                    image_file_name = image_file_names[read_file_nums].split('_')
        image_file_paths = [os.path.join(images_dir_path, image_file_name) for image_file_name in image_file_names]
        data_paths_and_labels = list(zip(image_file_paths, image_labels))
        random.seed(3)
        random.shuffle(data_paths_and_labels)
        return data_paths_and_labels

    def get_data_and_labels(self, is_training, batch_size):
        data_paths_and_labels = self.train_data_paths_and_labels if is_training else self.test_data_paths_and_labels
        old_data_pointer = self.train_data_pointer if is_training else self.test_data_pointer
        data_nums = len(data_paths_and_labels)
        if old_data_pointer + batch_size >= data_nums:
            new_data_pointer = 0
            batch_data_paths_and_labels = data_paths_and_labels[old_data_pointer:]
        else:
            new_data_pointer = old_data_pointer + batch_size
            batch_data_paths_and_labels = data_paths_and_labels[old_data_pointer: new_data_pointer]
        if is_training:
            self.train_data_pointer = new_data_pointer
        else:
            self.test_data_pointer = new_data_pointer
        batch_data = []
        batch_labels = []
        for data_path, label in batch_data_paths_and_labels:
            data = np.array(Image.open(data_path).resize(self.image_size)) / 255
            data = data.transpose((2, 0, 1))
            batch_data.append(data)
            batch_labels.append(label)
        return torch.from_numpy(np.array(batch_data)), torch.from_numpy(np.array(batch_labels))

    def _iter_epoch(self, is_training):

        if is_training:
            self.model.train()
            data_nums = len(self.train_data_paths_and_labels)
            batch_size = self.train_args.train_batch_size
        else:
            self.model.eval()
            data_nums = len(self.test_data_paths_and_labels)
            batch_size = self.train_args.test_batch_size

        trained_data_num = 0
        total_loss = 0
        correct = 0
        batches = (data_nums - 1) // batch_size + 1
        for batch_idx in range(1, batches + 1):
            data, target = self.get_data_and_labels(is_training, batch_size)

            data, target = Variable(data).float(), Variable(target).long()

            if self.train_args.cuda:
                data, target = data.cuda(), target.cuda()

            if is_training:
                self.optim.zero_grad()

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            if is_training:
                loss.backward()
                self.optim.step()

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            total_loss += loss.item()

            if is_training:
                trained_data_num += data.shape[0]
                if batch_idx % self.train_args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.epoch, trained_data_num, data_nums, 100. * batch_idx / batches, loss.item()))

        total_loss /= batches
        if is_training:
            self.train_acc.append(100. * correct / data_nums)
            self.train_loss.append(total_loss)
            self.save_acc.write('Epoch {} \r\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r\n'.format(
                self.epoch, total_loss, correct, data_nums, 100. * correct / data_nums))
        else:
            self.test_acc.append(100. * correct / data_nums)
            self.test_loss.append(total_loss)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                total_loss, correct, data_nums, 100. * correct / data_nums))
            self.save_acc.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r\n\n'.format(
                total_loss, correct, data_nums, 100. * correct / data_nums))
            return correct

    def record_time(self, hint):
        localtime = time.asctime( time.localtime(time.time()))
        self.save_acc.write(hint + localtime + '\r\n\n')

    def plot_acc_loss(self, end_epoch):
        x = np.arange(1,  end_epoch)

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.grid(linestyle=":")
        plt.plot(x, np.array(self.train_loss), label='train')
        plt.plot(x, np.array(self.test_loss), label='test')
        plt.legend()
        plt.savefig("record/" + self.data_name + "_loss.png", dpi=300, format="png")

        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.title("Accuracy")
        plt.grid(linestyle=":")
        plt.plot(x, np.array(self.train_acc), label='train')
        plt.plot(x, np.array(self.test_acc), label='test')
        plt.legend()
        plt.savefig("record/" + self.data_name + "_acc.png", dpi=300, format="png")

    def start_training(self, data_name):

        self._build(data_name=data_name)
        self.record_time('開始時間 : ')

        end_epoch = self.train_args.epochs + 1
        best_correct = 0
        check_count = 0
        for epoch in range(1,  self.train_args.epochs + 1):
            print('Epoch [{} / {}]'.format(epoch, self.train_args.epochs))
            self.epoch = epoch
            self._iter_epoch(is_training=True)
            correct = self._iter_epoch(is_training=False)

            # early stopping
            if correct > best_correct:
                best_correct = correct
                check_count = 0

            else:
                check_count += 1

            if check_count > 5:
                print('\nEarly stop at epoch {}\n'.format(epoch))
                end_epoch = epoch + 1
                break

        self.record_time('結束時間 : ')
        self.save_acc.close()
        self.plot_acc_loss(end_epoch)


if __name__ == '__main__':

    os.chdir('../../')
    data_name = 'DRD'

    lc_train = Central_Train()
    lc_train.start_training(data_name)








