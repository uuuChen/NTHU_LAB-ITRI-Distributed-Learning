from train.switch import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import torch.optim as optim
import time


class Central_Train:

    def __init__(self):
        pass

    def _build(self, data_name):
        self.save_acc = open("record/" + data_name + "_central_record.txt", "w")

        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        self.data_name = data_name

        self.switch = Switch(data_name=data_name)

        self.train_args = self.switch.get_train_args()

        self.train_dataSet, self.test_dataSet = self.switch.get_dataSet(shuffle=True)

        self.model = self.switch.get_model(is_central=True)

        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                                 # deterministic

        if self.train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.model.cuda()  # move all model parameters to the GPU

        self.optim = optim.Adam(self.model.parameters(), lr=self.train_args.lr)

    def _iter_epoch(self, is_training):

        if is_training:
            self.model.train()
            dataSet = self.train_dataSet
            batch_size = self.train_args.train_batch_size
        else:
            self.model.eval()
            dataSet = self.test_dataSet
            batch_size = self.train_args.test_batch_size

        data_nums = dataSet.get_usage_data_nums()

        targets = []
        preds = []
        trained_data_num = 0
        total_loss = 0
        correct = 0
        batches = (data_nums - 1) // batch_size + 1
        for batch_idx in range(1, batches + 1):
            data, target = dataSet.get_data_and_labels(batch_size=batch_size)

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

            targets.extend(target.data.cpu())
            preds.extend(pred.data.cpu())

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
            print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                total_loss, correct, data_nums, 100. * correct / data_nums))
        else:
            self.test_acc.append(100. * correct / data_nums)
            self.test_loss.append(total_loss)
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                total_loss, correct, data_nums, 100. * correct / data_nums))
            self.save_acc.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r\n\n'.format(
                total_loss, correct, data_nums, 100. * correct / data_nums))

            if int(self.epoch) == int(self.train_args.epochs):
                self.plot_confusion_matrix(target=targets, pred=preds,
                    classes=np.array(list(self.test_dataSet.class_id.keys())), data_name=self.data_name)
                self.plot_confusion_matrix(target=targets, pred=preds,
                    classes=np.array(list(self.test_dataSet.class_id.keys())), data_name=self.data_name, normalize=True)
            return correct

    def record_time(self, hint):
        localtime = time.asctime( time.localtime(time.time()))
        self.save_acc.write(hint + localtime + '\r\n\n')

    def plot_confusion_matrix(self, target, pred, classes, data_name,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = data_name + ' Normalized confusion matrix'
            else:
                title = data_name + ' Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(target, pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(target, pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if normalize:
            plt.savefig("record/" + self.data_name + "_confusion_matrix(normalize).png", dpi=300, format="png")
        else:
            plt.savefig("record/" + self.data_name + "_central_confusion_matrix.png", dpi=300, format="png")


    def plot_acc_loss(self, end_epoch):
        x = np.arange(1,  end_epoch)

        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.grid(linestyle=":")
        plt.plot(x, np.array(self.train_loss), label='train')
        plt.plot(x, np.array(self.test_loss), label='test')
        plt.legend()
        plt.savefig("record/" + self.data_name + "_central_loss.png", dpi=300, format="png")

        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.title("Accuracy")
        plt.grid(linestyle=":")
        plt.plot(x, np.array(self.train_acc), label='train')
        plt.plot(x, np.array(self.test_acc), label='test')
        plt.legend()
        plt.savefig("record/" + self.data_name + "_central_acc.png", dpi=300, format="png")

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

            if check_count >= 5 and epoch >= 20:
                print('\nEarly stop at epoch {}\n'.format(epoch))
                end_epoch = epoch + 1
                break

        self.record_time('結束時間 : ')
        self.save_acc.close()
        self.plot_acc_loss(end_epoch)


if __name__ == '__main__':

    os.chdir('../../')
    data_name = 'OCT'

    lc_train = Central_Train()
    lc_train.start_training(data_name)








