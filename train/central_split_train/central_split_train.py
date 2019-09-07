
from train.switch import *
from torch.autograd import Variable


class Local_Split_Train:

    def __init__(self):
        pass

    def _build(self, data_name):

        self.data_name = data_name

        self.switch = Switch(data_name=data_name)

        self.train_args = self.switch.get_train_args()

        self.train_dataSet, self.test_dataSet = self.switch.get_dataSet(shuffle=True)

        self.server_model = self.switch.get_model(is_server=True)

        self.agent_model = self.switch.get_model(is_agent=True)

        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.train_args.seed)  # seeding the CPU for generating random numbers so that the results are
                                                 # deterministic

        if self.train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.server_model.cuda()  # move all model parameters to the GPU
            self.agent_model.cuda()  # move all model parameters to the GPU

        self.server_optim = optim.SGD(self.server_model.parameters(), lr=self.train_args.lr,
                                      momentum=self.train_args.momentum)

        self.agent_optim = optim.SGD(self.agent_model.parameters(), lr=self.train_args.lr,
                                     momentum=self.train_args.momentum)

    def _iter_epoch(self, is_training):

        if is_training:
            self.server_model.train()
            self.agent_model.train()
            dataSet = self.train_dataSet
            batch_size = self.train_args.train_batch_size
        else:
            self.server_model.eval()
            self.agent_model.eval()
            dataSet = self.test_dataSet
            batch_size = self.train_args.test_batch_size

        data_nums = dataSet.get_data_nums_from_database()

        trained_data_num = 0
        test_loss = 0
        correct = 0
        batches = (data_nums - 1) // batch_size + 1
        for batch_idx in range(1, batches + 1):

            data, target = dataSet.get_data_and_labels(batch_size=batch_size)

            data, target = Variable(data).float(), Variable(target).long()
            if self.train_args.cuda:
                data, target = data.cuda(), target.cuda()

            # agent forward
            if is_training:
                self.agent_optim.zero_grad()
            agent_output = self.agent_model(data)

            server_input = Variable(agent_output, requires_grad=True).float()
            if self.train_args.cuda:
                server_input = server_input.cuda()

            # server forward
            if is_training:
                self.server_optim.zero_grad()
            output = self.server_model(server_input)
            loss = F.cross_entropy(output, target)

            if is_training:
                # server backward
                loss.backward()
                self.server_optim.step()

                # agent backward
                agent_output.backward(gradient=server_input.grad.data)
                self.agent_optim.step()

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

    os.chdir('../../')
    data_name = 'Xray'

    lc_train = Local_Split_Train()
    lc_train.start_training(data_name)










