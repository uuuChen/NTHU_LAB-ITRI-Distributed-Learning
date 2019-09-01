
from train.central import *
from torch.autograd import Variable

os.chdir('../')

class Local_Central_Train:

    def __init__(self):
        pass

    def _build(self, data_name):
        self.data_name = data_name

        self.central = Central(data_name=data_name)

        self.train_args = self.central.get_train_args()

        self.train_dataSet, self.test_dataSet = self.central.get_dataSet(shuffle=True)

        self.model = self.central.get_model(is_central=True)

        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                                 # deterministic

        if self.train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.model.cuda()  # move all model parameters to the GPU

        self.optim = optim.SGD(self.model.parameters(), lr=self.train_args.lr,  momentum=self.train_args.momentum)

    def _iter_epoch(self, is_training):

        if is_training:
            self.model.train()
            dataSet = self.train_dataSet
            batch_size = self.train_args.train_batch_size
        else:
            self.model.eval()
            dataSet = self.test_dataSet
            batch_size = self.train_args.test_batch_size

        data_nums = dataSet.get_data_nums_from_database()

        trained_data_num = 0
        test_loss = 0
        correct = 0
        batches = (data_nums - 1) // batch_size + 1
        for batch_idx in range(1, batches + 1):

            data, target = dataSet.get_data_and_labels(batch_size=batch_size)

            if self.train_args.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data).float(), Variable(target).long()

            if is_training:
                self.optim.zero_grad()

            output = self.model(data)
            loss = F.nll_loss(output, target)

            if is_training:
                loss.backward()
                self.optim.step()
            else:
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()
                test_loss += loss

            if is_training:
                trained_data_num += data.shape[0]
                if batch_idx % self.train_args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.epoch, trained_data_num, data_nums, 100. * batch_idx / batches, loss.item()))

        if not is_training:
            test_loss /= batches
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, data_nums, 100. * correct / data_nums))


    def start_training(self, data_name):

        self._build(data_name=data_name)

        for epoch in range(1,  self.train_args.epochs + 1):
            self.epoch = epoch
            self._iter_epoch(is_training=True)
            self._iter_epoch(is_training=False)


if __name__ == '__main__':

    lc_train = Local_Central_Train()

    lc_train.start_training('ECG')

    # lc_train.start_training('MNIST')

    # lc_train.start_training('DRD')








