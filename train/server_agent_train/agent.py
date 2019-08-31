
import torch
from torch.autograd import Variable

from socket_.socket_ import *
from logger import *
import time

# DataSet Imports
from dataSet.MNIST_dataSet import *
from dataSet.DRD_dataSet import *
from dataSet.Xray_dataSet import *
from dataSet.ECG_dataSet import *
from data.data_args import *  # import data arguments

class Agent(Logger):

    def __init__(self, model, train_dataSet, test_dataSet, server_host_port, cur_name):
        Logger.__init__(self)
        self.model = model
        self.is_simulate = None
        self.train_dataSet = None
        self.test_dataSet = None
        self.train_data_nums = None
        self.test_data_nums = None
        self.train_id_list = None
        self.test_id_list = None
        self.server_host_port = server_host_port
        self.cur_host_port = None
        self.prev_agent_attrs = None
        self.next_agent_attrs = None
        self.to_agent_sock = None
        self.cur_name = cur_name
        self.train_args = None
        self.agent_server_sock = None
        self.optim = None

    def get_dataSet(self, shuffle):
        train_dataSet = None
        test_dataSet = None
        data_name = self.train_args.dataSet
        if data_name == 'MNIST':
            train_dataSet = MNIST_DataSet(MNIST_TRAIN_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)
            test_dataSet = MNIST_DataSet(MNIST_TEST_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)
        elif data_name == 'DRD':
            train_dataSet = DRD_DataSet(DRD_TRAIN_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)
            test_dataSet = DRD_DataSet(DRD_TEST_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)
        elif data_name == 'ECG':
            train_dataSet = ECG_DataSet(ECG_TRAIN_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)
            test_dataSet = ECG_DataSet(ECG_TEST_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)
        elif data_name == 'Xray':
            train_dataSet = Xray_DataSet(Xray_TRAIN_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)
            test_dataSet = Xray_DataSet(Xray_TEST_ARGS, shuffle=shuffle, is_simulate=self.is_simulate)

        self.train_dataSet = train_dataSet
        self.test_dataSet = test_dataSet

    def conn_to_server(self):
        # connect to server, agent and server socket setting
        self.agent_server_sock = Socket(self.server_host_port, False)
        self.agent_server_sock.connect()

    def recv_train_args_from_server(self):
        self.train_args = self.agent_server_sock.recv('train_args')
        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()
        self.is_simulate = self.train_args.is_simulate
        if self.is_simulate:
            self.get_dataSet(shuffle=False)
        else:
            self.get_dataSet(shuffle=True)

    def recv_agents_attrs_from_server(self):
        # receive own IP and distributed port
        self.cur_host_port = self.agent_server_sock.recv('cur_host_port')
        self.to_agent_sock = Socket(self.cur_host_port, True)

        # receive previous, next agents from server
        self.prev_agent_attrs, self.next_agent_attrs = self.agent_server_sock.recv('prev_next_agent_attrs')

    def send_total_data_nums_to_server(self):
        if self.agent_server_sock.recv('is_first_agent'):
            self.send_data_nums_to_server()

    def recv_id_list_from_server(self):
        self.train_id_list, self.test_id_list = self.agent_server_sock.recv('train_test_id_list')

    def training_setting(self):

        if self.is_simulate:
            # set dataSet's "db_id_list" and store train and test data numbers
            self.train_dataSet.set_db_id_list(self.train_id_list)
            self.test_dataSet.set_db_id_list(self.test_id_list)
            train_data_nums = len(self.train_id_list)
            test_data_nums = len(self.test_id_list)
        else:
            train_data_nums = self.train_dataSet.get_data_nums_from_database()
            test_data_nums = self.test_dataSet.get_data_nums_from_database()

        self.train_data_nums = train_data_nums
        self.test_data_nums = test_data_nums

        # seeding the CPU for generating random numbers so that the
        torch.manual_seed(self.train_args.seed)

        # results are deterministic
        if self.train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.model.cuda()  # move all model parameters to the GPU
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.train_args.lr,
                                     momentum=self.train_args.momentum)

    def send_data_nums_to_server(self):
        train_data_nums = self.train_dataSet.get_data_nums_from_database()
        test_data_nums = self.test_dataSet.get_data_nums_from_database()
        self.agent_server_sock.send(train_data_nums, 'train_data_nums')
        self.agent_server_sock.send(test_data_nums, 'test_data_nums')

    def send_model_to_next_agent(self):
        # send model to next agent
        self.to_agent_sock.accept()
        self.to_agent_sock.send(self.model, 'agent_model')
        self.to_agent_sock.close()

    def iter_through_db_once(self, is_training):

        print('%s starts training....' % self.cur_name)

        if is_training:
            self.model.train()
            data_nums = self.train_data_nums
            dataSet = self.train_dataSet
        else:
            self.model.eval()
            data_nums = self.test_data_nums
            dataSet = self.test_dataSet

        batches = (data_nums - 1) // self.train_args.batch_size + 1

        for batch_idx in range(1, batches + 1):

            data, target = dataSet.get_data_and_labels(batch_size=self.train_args.batch_size)

            if self.train_args.cuda:
                data = data.cuda()

            data, target = Variable(data).float(), Variable(target).long()

            # agent forward
            agent_output = self.model(data)
            print(agent_output)

            # send agent_output and target to server
            self.agent_server_sock.send(agent_output, 'agent_output')  # send agent_output
            self.agent_server_sock.send(target, 'target')  # send target

            # receive gradient from server if training
            if is_training:
                self.optim.zero_grad()
                # get agent_output_clone
                agent_output_grad = self.agent_server_sock.recv('agent_output_clone_grad')

                # agent backward
                agent_output.backward(gradient=agent_output_grad)
                self.optim.step()

    def get_prev_model_from_prev_agent(self):
        if not self.agent_server_sock.recv('is_first_training'):
            from_agent_sock = Socket(self.prev_agent_attrs['host_port'], False)
            from_agent_sock.connect()
            self.model = from_agent_sock.recv('agent_model')
            from_agent_sock.close()

            # VERY IMPORTANT !!! awake server after current agent receiving model snapshot from previous agent
            self.agent_server_sock.awake()

    def train_agent(self):
        self.get_prev_model_from_prev_agent()
        self.iter_through_db_once(is_training=True)
        print('%s done training' % self.cur_name)
        self.send_model_to_next_agent()

    def test_agent(self):
        self.get_prev_model_from_prev_agent()
        self.iter_through_db_once(is_training=False)
        print('%s done testing' % self.cur_name)
        # if it is the last epoch
        if self.agent_server_sock.recv('is_training_done'):
            # if it is the last agent to test
            if int(self.cur_name.split("_")[1]) is self.train_args.agent_nums:
                # no need to send model
                self.agent_server_sock.close()
                return True
            else:
                self.send_model_to_next_agent()
                self.agent_server_sock.close()
                return True
        else:
            self.send_model_to_next_agent()
            return False

    def start_training(self):

        self.conn_to_server()

        self.recv_train_args_from_server()

        self.recv_agents_attrs_from_server()

        if self.is_simulate:
            self.send_total_data_nums_to_server()
            self.recv_id_list_from_server()

        else:
            self.send_data_nums_to_server()

        self.training_setting()

        while True:
            self.train_agent()
            done = self.test_agent()
            if done:
                break


           



