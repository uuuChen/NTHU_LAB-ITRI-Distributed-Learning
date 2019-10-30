import torch

from torch.autograd import Variable
import time

from socket_.socket_ import Socket
from logger import Logger

from train.switch import *


class Agent(Logger):

    def __init__(self, server_host_port, cur_name):
        Logger.__init__(self)
        self.server_host_port = server_host_port
        self.cur_name = cur_name

    def _conn_to_server(self):
        self.agent_server_sock = Socket(self.server_host_port, False)
        self.agent_server_sock.connect()

    def _recv_train_args_from_server(self):
        self.train_args = self.agent_server_sock.recv('train_args')

    def _training_setting(self):

        self.train_args.cuda = not self.train_args.no_cuda and torch.cuda.is_available()
        self.is_simulate = self.train_args.is_simulate

        self.switch = Switch(data_name=self.train_args.dataSet)
        self.train_dataSet, self.test_dataSet = self.switch.get_dataSet(shuffle=True, is_simulate=self.is_simulate)

        # save
        if not os.path.exists(self.train_args.save_path + 'agent/'):
            self.model = self.switch.get_model(is_agent=True)
            os.makedirs(self.train_args.save_path + 'agent/')
            print(self.train_args.save_path)
        else:
            save_path = self.train_args.save_path + "agent/" + str(self.train_args.start_epoch) + "epochs_model.pkl"
            print(save_path)
            self.model = torch.load(save_path)


        if self.is_simulate:  # have to wait for "id_list" receiving from server
            self.train_data_nums = None
            self.test_data_nums = None
        else:
            self.train_data_nums = self.train_dataSet.get_usage_data_nums()
            self.test_data_nums = self.test_dataSet.get_usage_data_nums()

        # seeding the CPU for generating random numbers so that the
        torch.manual_seed(self.train_args.seed)

        # results are deterministic
        if self.train_args.cuda:
            torch.cuda.manual_seed(self.train_args.seed)  # set a random seed for the current GPU
            self.model.cuda()  # move all model parameters to the GPU

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_args.lr)

    def _recv_agents_attrs_from_server(self):
        # receive own IP and distributed port
        self.cur_host_port = self.agent_server_sock.recv('cur_host_port')
        self.to_agent_sock = Socket(self.cur_host_port, True)

        # receive previous, next agents from server
        self.prev_agent_attrs, self.next_agent_attrs = self.agent_server_sock.recv('prev_next_agent_attrs')

    def _send_total_data_nums_to_server(self):
        if self.agent_server_sock.recv('is_first_agent'):
            self._send_data_nums_to_server()

    def _recv_id_list_from_server(self):
        self.train_id_list, self.test_id_list = self.agent_server_sock.recv('train_test_id_list')

        # set dataSet's "db_id_list" and store train and test data numbers
        self.train_dataSet.set_usage_data_ids(self.train_id_list)
        self.test_dataSet.set_usage_data_ids(self.test_id_list)

        self.train_data_nums = len(self.train_id_list)
        self.test_data_nums = len(self.test_id_list)

    def _send_data_nums_to_server(self):
        train_data_nums = self.train_dataSet.get_usage_data_nums()
        test_data_nums = self.test_dataSet.get_usage_data_nums()
        print('train_data_nums : {} test_data_nums : {}'.format(train_data_nums, test_data_nums))
        self.agent_server_sock.send(train_data_nums, 'train_data_nums')
        self.agent_server_sock.send(test_data_nums, 'test_data_nums')

    def _send_model_to_next_agent(self):
        # send model to next agent
        self.to_agent_sock.accept()
        self.to_agent_sock.send(self.model.state_dict(), 'model_state_dict')
        self.to_agent_sock.send(self.optim.state_dict(), 'optim_state_dict')
        self.to_agent_sock.close()

    def _iter(self, is_training):

        self._whether_to_recv_model_from_prev_agent()

        self._iter_through_database_once(is_training=is_training)

        if is_training:
            self._send_model_to_next_agent()

        else:
            return self._whether_is_training_done()

    def _whether_to_recv_model_from_prev_agent(self):

        print('\nwait for previous agent {} sending model snapshot...'.format(self.prev_agent_attrs['host_port']))

        if not self.agent_server_sock.recv('is_first_training'):

            from_agent_sock = Socket(self.prev_agent_attrs['host_port'], False)
            from_agent_sock.connect()

            model_state_dict = from_agent_sock.recv('model_state_dict')
            optim_state_dict = from_agent_sock.recv('optim_state_dict')

            self.model.load_state_dict(model_state_dict)
            self.optim.load_state_dict(optim_state_dict)

            from_agent_sock.close()

            # VERY IMPORTANT !!! awake server after current agent receiving model snapshot from previous agent
            self.agent_server_sock.awake()

        print('done ! \n')

    def _iter_through_database_once(self, is_training):

        iter_type = 'Training' if is_training else 'Testing'
        print('{} starts {}....'.format(self.cur_name, iter_type ))

        if is_training:
            self.model.train()
            data_nums = self.train_data_nums
            dataSet = self.train_dataSet
            batch_size = self.train_args.train_batch_size
        else:
            self.model.eval()
            data_nums = self.test_data_nums
            dataSet = self.test_dataSet
            batch_size = self.train_args.test_batch_size

        trained_data_num = 0
        batches = (data_nums - 1) // batch_size + 1

        for batch_idx in range(1, batches + 1):

            data, target = dataSet.get_data_and_labels(batch_size=batch_size)

            data, target = Variable(data).float(), Variable(target).long()
            if self.train_args.cuda:
                data, target = data.cuda(), target.cuda()

            # agent forward
            if is_training:
                self.optim.zero_grad()
            agent_output = self.model(data)

            # send agent_output and target to server
            self.agent_server_sock.send(agent_output, 'agent_output')  # send agent_output
            self.agent_server_sock.send(target, 'target')  # send target

            # receive gradient from server if training
            if is_training:
                agent_output_grad = self.agent_server_sock.recv('agent_output_grad_data')  # get agent_output_clone

                # agent backward
                agent_output.backward(gradient=agent_output_grad)
                self.optim.step()

            trained_data_num += data.shape[0]
            server_host_port = self.agent_server_sock.getpeername()
            print('{} at {}: [{}/{} ({:.0f}%)]'.format(iter_type, server_host_port, trained_data_num, data_nums,
                                                       100.0 * batch_idx / batches))

        train_str = 'training' if is_training else 'testing'
        print('{} done {} \n'.format(self.cur_name, train_str))

    def _whether_is_training_done(self):

        self.cur_epoch = self.agent_server_sock.recv('cur_epoch')
        if self.train_args.epochs == self.cur_epoch:
            is_last_agent = (int(self.cur_name.split("_")[1]) == self.train_args.agent_nums)
            if not is_last_agent:
                self._send_model_to_next_agent()
            self.agent_server_sock.close()
            return True
        else:
            self._send_model_to_next_agent()
            if self.cur_epoch % 5 == 0:
                torch.save(self.model, self.train_args.save_path + 'agent/' + str(self.cur_epoch) + 'epochs_model.pkl')
            return False

    def start_training(self):

        self._conn_to_server()

        self._recv_train_args_from_server()

        self._training_setting()  # train setting after receiving train args

        self._recv_agents_attrs_from_server()

        if self.is_simulate:
            self._send_total_data_nums_to_server()
            self._recv_id_list_from_server()

        else:
            self._send_data_nums_to_server()

        while True:
            self._iter(is_training=True)
            done = self._iter(is_training=False)
            if done:
                break




