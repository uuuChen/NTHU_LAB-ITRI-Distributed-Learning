import random
from torch.autograd import Variable

# Model Imports
from model.LeNet import *
from model.AlexNet import *
from model.MLP import *
from model.VGGNet import *

# Socket Imports
from socket_.socket_ import *

from logger import *

class Server(Logger):
    def __init__(self, train_args):

        Logger.__init__(self)

        # server socket setting
        self.server_port_begin = 8080
        self.server_socks = []

        # agent host port list for testing
        self.agent_port_begin = 2048
        self.agents_attrs = []

        # stored data from agent
        self.train_data_nums = [0] * train_args.agent_nums
        self.test_data_nums = [0] * train_args.agent_nums
        self.all_train_data_nums = 0
        self.all_test_data_nums = 0

        # training setting
        self.is_simulate = train_args.is_simulate
        self.is_first_training = True
        self.train_args = train_args

        if self.train_args.model is 'LeNet':
            self.model = Server_LeNet()
        elif self.train_args.model is 'AlexNet':
            self.model = Server_AlexNet()
        elif self.train_args.model is 'MLP':
            self.model = Server_MLP()
        elif self.train_args.model is 'VGGNet':
            self.model = VGGNet()

        # training setup
        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.train_args.seed)  # seeding the CPU for generating random numbers so that the results are
        # deterministic
        if train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.model.cuda()  # move all model parameters to the GPU
        self.optimizer = optim.SGD(self.model.parameters(),
                                     lr=self.train_args.lr,
                                     momentum=self.train_args.momentum)

    def conn_to_agents(self):

        for i in range(self.train_args.agent_nums):
            server_sock = Socket(('localhost', self.server_port_begin + i), True)
            self.server_socks.append(server_sock)

        for i in range(self.train_args.agent_nums):
            self.server_socks[i].accept()
            agents_attr = {
                'name': 'agent_' + str(i + 1),
                'host_port': (self.server_socks[i].addr[0], self.agent_port_begin + i)
            }
            self.agents_attrs.append(agents_attr)

    def get_prev_next_agents_attrs(self, agent_idx):

        prev_agent_idx = agent_idx - 1
        next_agent_idx = agent_idx + 1

        if prev_agent_idx == -1:
            prev_agent_idx = self.train_args.agent_nums - 1
        if next_agent_idx == self.train_args.agent_nums:
            next_agent_idx = 0

        prev_agent_attrs = self.agents_attrs[prev_agent_idx]
        next_agent_attrs = self.agents_attrs[next_agent_idx]

        return prev_agent_attrs, next_agent_attrs

    def get_total_data_nums_from_first_agent(self):

        for i in range(self.train_args.agent_nums):
            if i == 0:
                self.server_socks[i].send(True, 'is_first_agent')
                self.all_train_data_nums = self.server_socks[i].recv('train_data_nums')
                self.all_test_data_nums = self.server_socks[i].recv('test_data_nums')
            else:
                self.server_socks[i].send(False, 'is_first_agent')

    def send_id_lists_to_agents(self):

        all_train_id_list = list(range(1, self.all_train_data_nums + 1))
        all_test_id_list = list(range(1, self.all_test_data_nums + 1))

        random.shuffle(all_train_id_list)
        random.shuffle(all_test_id_list)

        left_train_data_nums = self.all_train_data_nums
        left_test_data_nums = self.all_test_data_nums
        left_agents_nums = self.train_args.agent_nums

        train_start_idx = 0
        test_start_idx = 0

        for i in range(self.train_args.agent_nums):
            agent_train_data_nums = (left_train_data_nums - 1) // left_agents_nums + 1
            agent_test_data_nums = (left_test_data_nums - 1) // left_agents_nums + 1

            agent_train_id_list = all_train_id_list[train_start_idx: train_start_idx + agent_train_data_nums]
            agent_test_id_list = all_test_id_list[test_start_idx: test_start_idx + agent_test_data_nums]

            self.train_data_nums[i] = agent_train_data_nums
            self.test_data_nums[i] = agent_test_data_nums

            self.server_socks[i].send([agent_train_id_list, agent_test_id_list], 'train_test_id_list')

            train_start_idx += agent_train_data_nums
            test_start_idx += agent_test_data_nums

            left_train_data_nums -= agent_train_data_nums
            left_test_data_nums -= agent_test_data_nums

            left_agents_nums -= 1

    def get_data_nums_from_agents(self):

        for i in range(self.train_args.agent_nums):
            self.train_data_nums[i] = self.server_socks[i].recv('train_data_nums')
            self.all_train_data_nums += self.train_data_nums[i]

            self.test_data_nums[i] = self.server_socks[i].recv('test_data_nums')
            self.all_test_data_nums += self.test_data_nums[i]

    def send_train_args_to_agents(self):

        for i in range(self.train_args.agent_nums):
            # send train args to agent
            self.server_socks[i].send(self.train_args, 'train_args')

            # send agent IP and distributed port
            self.server_socks[i].send(self.agents_attrs[i]['host_port'], 'cur_host_port')

    def send_prev_next_agent_attrs(self, agent_idx):

        # get previous and next agent attributes
        prev_agent_attrs, next_agent_attrs = self.get_prev_next_agents_attrs(agent_idx)

        # send prev_agent_attrs, next_agent_attrs to agent
        if self.is_first_training:
            prev_agent_attrs = None
        self.server_socks[agent_idx].send((prev_agent_attrs, next_agent_attrs), 'prev_next_agent_attrs')

        # VERY IMPORTANT !!! server is waiting for previos agent sending model snapshot to current agent
        if not self.is_first_training:
            self.server_socks[agent_idx].sleep()

        self.is_first_training = False

    def train_with_cur_agent(self, agent_idx, epoch, trained_data_num):

        data_nums = self.train_data_nums[agent_idx]
        batches = (data_nums - 1) // self.train_args.batch_size + 1

        for batch_idx in range(1, batches + 1):
            self.optimizer.zero_grad()

            # get agent_output and target from agent
            agent_output = self.server_socks[agent_idx].recv('agent_output')
            target = self.server_socks[agent_idx].recv('target')

            # store gradient in agent_output_clone
            agent_output_clone = Variable(agent_output).float()
            if self.train_args.cuda:
                agent_output_clone = agent_output_clone.cuda()
                target = target.cuda()
            agent_output_clone.requires_grad_()

            # server forward
            server_output = self.model(agent_output_clone)
            loss = F.cross_entropy(server_output, target)

            # server backward
            loss.backward()
            self.optimizer.step()

            # send gradient to agent
            self.server_socks[agent_idx].send(agent_output_clone.grad.data, 'agent_output_clone_grad')

            trained_data_num += len(target)
            if batch_idx % self.train_args.log_interval == 0:
                print('Train Epoch: {} at {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, self.agents_attrs[agent_idx], trained_data_num, self.all_train_data_nums,
                    100. * batch_idx / batches, loss.item()))

        return trained_data_num

    def train_epoch(self, epoch):

        self.model.train()

        trained_data_num = 0
        for i in range(self.train_args.agent_nums):
            self.send_prev_next_agent_attrs(i)

            print('starting training with' + str(self.agents_attrs[i]))
            trained_data_num = self.train_with_cur_agent(i, epoch, trained_data_num)

    def test_with_cur_agent(self, agent_idx, test_loss, correct):

        data_nums = self.test_data_nums[agent_idx]
        batches = (data_nums - 1) // self.train_args.batch_size + 1
        for batch_idx in range(1, batches + 1):

            # get agent_output and target from agent
            agent_output = self.server_socks[agent_idx].recv('agent_output')
            target = self.server_socks[agent_idx].recv('target')

            agent_output_clone = Variable(agent_output).float()

            if self.train_args.cuda:
                agent_output_clone = agent_output_clone.cuda()
                target = target.cuda()

            # server forward
            server_output = self.model(agent_output_clone)
            loss = F.cross_entropy(server_output, target)

            test_loss += loss

            pred = server_output.data.max(1)[1]

            correct += pred.eq(target.data).cpu().sum()

        return test_loss, correct

    def test_epoch(self, epoch):

        self.model.eval()

        test_loss = 0
        correct = 0
        batches = 0

        for i in range(self.train_args.agent_nums):

            self.send_prev_next_agent_attrs(i)

            test_loss, correct = self.test_with_cur_agent(i, test_loss, correct)
            batches += (self.test_data_nums[i] - 1) // self.train_args.batch_size + 1

            if epoch is self.train_args.epochs - 1:
                self.server_socks[i].send(True, 'is_training_done')
            else:
                self.server_socks[i].send(False, 'is_training_done')

        test_loss /= batches

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, self.all_test_data_nums,
            100. * correct / self.all_test_data_nums))

    def start_training(self):

        self.conn_to_agents()

        self.send_train_args_to_agents()

        if not self.is_simulate:  # for real hospitals
            self.get_data_nums_from_agents()

        else:  # for simulated accuracy test
            self.get_total_data_nums_from_first_agent()
            self.send_id_lists_to_agents()

        for epoch in range(self.train_args.epochs):
            # start training and testing
            self.train_epoch(epoch=epoch)
            self.test_epoch(epoch=epoch)


