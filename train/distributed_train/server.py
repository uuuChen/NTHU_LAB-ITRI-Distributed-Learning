import random

import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable

from train.switch import *

# Socket Imports
from socket_.socket_ import Socket

from logger import Logger


class Server(Logger):

    def __init__(self, data_name, use_localhost=True):

        Logger.__init__(self)

        # save train loss and test accuracy
        self.save_acc = open(data_name + "_distributed_acc.txt", "w")

        # get model and train args by "data_name"
        self.switch = Switch(data_name=data_name)
        self.model = self.switch.get_model(is_server=True)
        self.train_args = self.switch.get_train_args()

        # server socket setting
        self.server_port_begin = 8080
        self.server_socks = []

        # agent host port list for testing
        self.agent_port_begin = 2048
        self.agents_attrs = []

        # stored data from agent
        self.train_data_nums = [0] * self.train_args.agent_nums
        self.test_data_nums = [0] * self.train_args.agent_nums
        self.all_train_data_nums = 0
        self.all_test_data_nums = 0

        # training setting
        self.use_localhost = use_localhost
        self.is_simulate = self.train_args.is_simulate
        self.is_first_training = True
        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.train_args.seed) # seeding the CPU for generating random numbers so that the results are
                                                # deterministic
        if self.train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.model.cuda()  # move all model parameters to the GPU
        self.optim = optim.SGD(self.model.parameters(),  lr=self.train_args.lr, momentum=self.train_args.momentum)

    def _conn_to_agents(self):

        for i in range(self.train_args.agent_nums):
            if self.use_localhost:
                host_name = 'localhost'
            else:
                host_name = Socket.get_host_name()
            print('Server is waiting for connections on (\'{}\')'.format(host_name))
            server_sock = Socket((host_name, self.server_port_begin + i), True)
            self.server_socks.append(server_sock)

        for i in range(self.train_args.agent_nums):
            self.server_socks[i].accept()
            agents_attr = {
                'name': 'agent_' + str(i + 1),
                'host_port': (self.server_socks[i].addr[0], self.agent_port_begin + i)
            }
            self.agents_attrs.append(agents_attr)

    def _send_train_args_to_agents(self):

        for i in range(self.train_args.agent_nums):
            self.server_socks[i].send(self.train_args, 'train_args')

    def _send_agents_attrs_to_agents(self):

        for i in range(self.train_args.agent_nums):
            # send agent IP and distributed port
            self.server_socks[i].send(self.agents_attrs[i]['host_port'], 'cur_host_port')

            # send previous and next agent attributes
            prev_agent_attrs, next_agent_attrs = self._get_prev_next_agents_attrs(i)
            self.server_socks[i].send((prev_agent_attrs, next_agent_attrs), 'prev_next_agent_attrs')

    def _get_prev_next_agents_attrs(self, agent_idx):

        prev_agent_idx = agent_idx - 1
        next_agent_idx = agent_idx + 1

        if prev_agent_idx == -1:
            prev_agent_idx = self.train_args.agent_nums - 1
        if next_agent_idx == self.train_args.agent_nums:
            next_agent_idx = 0

        prev_agent_attrs = self.agents_attrs[prev_agent_idx]
        next_agent_attrs = self.agents_attrs[next_agent_idx]

        return prev_agent_attrs, next_agent_attrs

    def _recv_total_data_nums_from_first_agent(self):

        for i in range(self.train_args.agent_nums):
            if i == 0:
                self.server_socks[i].send(True, 'is_first_agent')
                self.all_train_data_nums = self.server_socks[i].recv('train_data_nums')
                self.all_test_data_nums = self.server_socks[i].recv('test_data_nums')
            else:
                self.server_socks[i].send(False, 'is_first_agent')

    def _send_id_lists_to_agents(self):

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

            train_start_idx += agent_train_data_nums
            test_start_idx += agent_test_data_nums

            left_train_data_nums -= agent_train_data_nums
            left_test_data_nums -= agent_test_data_nums

            left_agents_nums -= 1

            self.train_data_nums[i] = agent_train_data_nums
            self.test_data_nums[i] = agent_test_data_nums

            self.server_socks[i].send([agent_train_id_list, agent_test_id_list], 'train_test_id_list')

    def _recv_data_nums_from_agents(self):

        for i in range(self.train_args.agent_nums):
            self.train_data_nums[i] = self.server_socks[i].recv('train_data_nums')
            self.all_train_data_nums += self.train_data_nums[i]

            self.test_data_nums[i] = self.server_socks[i].recv('test_data_nums')
            self.all_test_data_nums += self.test_data_nums[i]

    def _whether_waiting_for_agent(self, cur_agent_idx):

        self.server_socks[cur_agent_idx].send(self.is_first_training, 'is_first_training')

        if not self.is_first_training:
            self.server_socks[cur_agent_idx].sleep()

        self.is_first_training = False

    def _whether_is_training_done(self, cur_agent_idx):

        if self.epoch == self.train_args.epochs - 1:
            self.server_socks[cur_agent_idx].send(True, 'is_training_done')

        else:
            self.server_socks[cur_agent_idx].send(False, 'is_training_done')

    def _train_log(self, cur_agent_idx, loss):
        print('Train Epoch: {} at {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            self.epoch, self.agents_attrs[cur_agent_idx], self.trained_data_num,
            self.all_train_data_nums, 100. * self.trained_data_num / self.all_train_data_nums, loss.item()))

    def _test_log(self):
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.test_loss, self.correct, self.all_test_data_nums,
            100. * self.correct / self.all_test_data_nums))
        self.save_acc.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r\n\n'.format(
            self.test_loss, self.correct, self.all_test_data_nums,
            100. * self.correct / self.all_test_data_nums))

    def _iter_through_agent_database(self, is_training, cur_agent_idx):

        iter_type = 'training' if is_training else 'testing'
        print('start %s with %s' % (iter_type, str(self.agents_attrs[cur_agent_idx])))

        if is_training:
            data_nums = self.train_data_nums[cur_agent_idx]
            batch_size = self.train_args.train_batch_size
        else:
            data_nums = self.test_data_nums[cur_agent_idx]
            batch_size = self.train_args.test_batch_size

        batches = (data_nums - 1) // batch_size + 1

        for batch_idx in range(1, batches + 1):

            # get agent_output and target from agent
            agent_output = self.server_socks[cur_agent_idx].recv('agent_output')
            target = self.server_socks[cur_agent_idx].recv('target')

            server_input = Variable(agent_output, requires_grad=True).float()
            if self.train_args.cuda:
                server_input = server_input.cuda()

            # server forward
            if is_training:
                self.optim.zero_grad()
            server_output = self.model(server_input)
            loss = F.cross_entropy(server_output, target)

            if is_training:
                # server backward
                loss.backward()
                self.optim.step()

                # send gradient to agent
                self.server_socks[cur_agent_idx].send(server_input.grad.data, 'agent_output_grad_data')

                self.trained_data_num += len(target)
                if batch_idx % self.train_args.log_interval == 0:
                    print('Train Epoch: {} at {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.epoch, self.agents_attrs[cur_agent_idx], self.trained_data_num,
                        self.all_train_data_nums, 100. * self.trained_data_num / self.all_train_data_nums, loss.item()))
                    if self.trained_data_num == self.all_train_data_nums:
                        self.save_acc.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r\n'.format(
                            self.epoch, self.trained_data_num,
                            self.all_train_data_nums, 100. * self.trained_data_num / self.all_train_data_nums, loss.item()))
            else:
                self.test_loss += loss.item()
                pred = server_output.data.max(1)[1]
                self.correct += pred.eq(target.data).cpu().sum()

        if not is_training:
            self.batches += batches

    def _iter_one_epoch(self, is_training):

        if is_training:
            self.model.train()
            self.trained_data_num = 0

        else:
            self.model.eval()
            self.test_loss = 0
            self.correct = 0
            self.batches = 0

        for agent_idx in range(self.train_args.agent_nums):

            self._whether_waiting_for_agent(agent_idx)

            self._iter_through_agent_database(is_training, agent_idx)

            if not is_training:
                self._whether_is_training_done(agent_idx)

        if not is_training:
            self.test_loss /= self.batches
            self._test_log()

    def start_training(self):

        self._conn_to_agents()

        self._send_train_args_to_agents()

        self._send_agents_attrs_to_agents()

        if self.is_simulate:  # for simulated accuracy test
            self._recv_total_data_nums_from_first_agent()
            self._send_id_lists_to_agents()

        else:  # for real hospitals usage
            self._recv_data_nums_from_agents()

        # start training and testing
        for epoch in range(self.train_args.epochs):
            self.epoch = epoch
            self._iter_one_epoch(is_training=True)
            self._iter_one_epoch(is_training=False)




