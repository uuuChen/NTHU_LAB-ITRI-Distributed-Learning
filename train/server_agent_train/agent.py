
import torch
from torch.autograd import Variable

from socket_.socket_ import *
from logger import *
import time

class Agent(Logger):

    def __init__(self, model, train_dataSet, test_dataSet, server_host_port, cur_host_port, cur_name):
        Logger.__init__(self)
        self.model = model
        self.train_dataSet = train_dataSet
        self.test_dataSet = test_dataSet
        self.server_host_port = server_host_port
        self.cur_host_port = cur_host_port
        self.to_agent_sock = Socket(cur_host_port, True)
        self.cur_name = cur_name
        self.train_args = None
        self.agent_server_sock = None
        self.optim = None

    def _train_epoch(self):

        self.model.train()
        data_nums = self.train_dataSet.get_data_nums_from_database()
        self.agent_server_sock.send(data_nums, 'data_nums')
        batches = (data_nums - 1) // self.train_args.batch_size + 1

        for batch_idx in range(1, batches + 1):

            data, target = self.train_dataSet.get_data_and_labels(batch_size=self.train_args.batch_size)

            if self.train_args.cuda:
                data = data.cuda()

            data, target = Variable(data).float(), Variable(target).long()

            # agent forward
            self.optim.zero_grad()
            agent_output = self.model(data)

            # send agent_output and target to server
            self.agent_server_sock.send(agent_output, 'agent_output')  # send agent_output
            self.agent_server_sock.send(target, 'target')  # send target

            # receive gradient from server
            agent_output_grad = self.agent_server_sock.recv('agent_output_clone_grad')  # get agent_output_clone

            # agent backward
            agent_output.backward(gradient=agent_output_grad)
            self.optim.step()

            # break

    def _test_epoch(self):

        self.model.eval()

        data_nums = self.test_dataSet.get_data_nums_from_database()
        self.agent_server_sock.send(data_nums, 'data_nums')
        batches = (data_nums - 1) // self.train_args.batch_size + 1

        for batch_idx in range(1, batches + 1):

            data, target = self.test_dataSet.get_data_and_labels(batch_size=self.train_args.batch_size)

            if self.train_args.cuda:
                data = data.cuda()

            data, target = Variable(data).float(), Variable(target).long()

            agent_output = self.model(data)

            # send agent_output and target to server
            self.agent_server_sock.send(agent_output, 'agent_output')  # send agent_output
            self.agent_server_sock.send(target, 'target')  # send target

    def start_training(self):
        while True:
            # connect to server, agent and server socket setting
            self.agent_server_sock = Socket(self.server_host_port, False)
            self.agent_server_sock.connect()

            if self.agent_server_sock.is_right_conn(client_name=self.cur_name):

                # get whether training is done from server
                if self.agent_server_sock.recv('is_training_done'):
                    self.agent_server_sock.close()
                    break

                print('%s starts training....' % self.cur_name)

                # receive previous, next agents from server
                prev_agent_attrs, next_agent_attrs = self.agent_server_sock.recv('prev_next_agent_attrs')

                # connect to last training agent and get model snapshot. prev_agent_attrs is None when the first
                # training
                if prev_agent_attrs is not None:
                    from_agent_sock = Socket(prev_agent_attrs['host_port'], False)
                    from_agent_sock.connect()
                    self.model = from_agent_sock.recv('agent_model')
                    from_agent_sock.close()

                    # VERY IMPORTANT !!! awake server after current agent receiving model snapshot from previous agent
                    self.agent_server_sock.awake()

                # receive train_args from server
                self.train_args = self.agent_server_sock.recv('train_args')
                self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()
                torch.manual_seed(self.train_args.seed)  # seeding the CPU for generating random numbers so that the
                                                         # results are deterministic
                if self.train_args.cuda:
                    torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
                    self.model.cuda()  # move all model parameters to the GPU
                self.optim = torch.optim.SGD(self.model.parameters(),
                                             lr=self.train_args.lr,
                                             momentum=self.train_args.momentum)

                # train an epoch with server
                self._train_epoch()
                self._test_epoch()

                # get whether training is done from server
                if self.agent_server_sock.recv('is_training_done'):
                    self.agent_server_sock.close()
                    break
                else:
                    self.agent_server_sock.awake()

                # send model to next agent
                self.to_agent_sock.accept()
                self.to_agent_sock.send(self.model, 'agent_model')
                self.to_agent_sock.close()

                print('%s done' % self.cur_name)




