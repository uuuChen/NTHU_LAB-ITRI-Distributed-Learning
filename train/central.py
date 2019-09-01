
# DataSet Imports
from dataSet.MNIST_dataSet import MNIST_DataSet
from dataSet.DRD_dataSet import DRD_DataSet
from dataSet.Xray_dataSet import Xray_DataSet
from dataSet.ECG_dataSet import ECG_DataSet
from data.data_args import *  # import data arguments

# Model Imports
from model.LeNet import *
from model.AlexNet import *
from model.MLP import *
from model.VGGNet import *

# Train Args Imports
from train.train_args import MNIST_TRAINING_ARGS, ECG_TRAINING_ARGS, DRD_TRAINING_ARGS

class Central:

    def __init__(self, data_name):

        self._build(data_name)

    def _build(self,data_name):

        if data_name == 'MNIST':

            dataSet = MNIST_DataSet
            train_data_args = MNIST_TRAIN_ARGS
            test_data_args = MNIST_TEST_ARGS

            train_args = MNIST_TRAINING_ARGS

            central_model = LeNet()
            server_model = Server_LeNet()
            agent_model = Agent_LeNet()

        elif data_name == 'DRD':

            dataSet = DRD_DataSet
            train_data_args = DRD_TRAIN_ARGS
            test_data_args = DRD_TEST_ARGS

            train_args = DRD_TRAINING_ARGS

            central_model = AlexNet()
            server_model = Server_AlexNet(flatten_nodes=1024,
                                          num_classes=DRD_COMMON_ARGS['label_class_nums'])
            agent_model = Agent_AlexNet()

        elif data_name == 'ECG':

            dataSet = ECG_DataSet
            train_data_args = ECG_TRAIN_ARGS
            test_data_args = ECG_TEST_ARGS

            train_args = ECG_TRAINING_ARGS

            central_model = MLP(input_node_nums=ECG_COMMON_ARGS['data_length'],
                                label_class_nums=ECG_COMMON_ARGS['label_class_nums'])
            server_model = Server_MLP(conn_node_nums=ECG_COMMON_ARGS['MLP_conn_node_nums'],
                                      label_class_nums=ECG_COMMON_ARGS['label_class_nums'])
            agent_model = Agent_MLP(input_node_nums=ECG_COMMON_ARGS['data_length'],
                                    conn_node_nums=ECG_COMMON_ARGS['MLP_conn_node_nums'])

        elif data_name == 'Xray':

            dataSet = Xray_DataSet
            train_data_args = Xray_TRAIN_ARGS
            test_data_args = Xray_TEST_ARGS

            """ todo """

            train_args = None

            central_model = None
            server_model = None
            agent_model = None

        else:
            raise Exception('DataSet ( %s ) Not Exist !' % data_name)

        self.central_model = central_model
        self.server_model = server_model
        self.agent_model = agent_model

        self.train_args = train_args
        self.dataSet = dataSet
        self.data_args = [train_data_args, test_data_args]

    def get_dataSet(self, **kwargs):

        training_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                training_args[key] = value

        train_data_args, test_data_args = self.data_args

        train_data_args.update(training_args)
        test_data_args.update(training_args)

        train_dataSet = self.dataSet(data_args=train_data_args)
        test_dataSet = self.dataSet(data_args=test_data_args)

        return train_dataSet, test_dataSet

    def get_model(self, is_central=False, is_server=False, is_agent=False):
        if is_central:
            return self.central_model

        elif is_server:
            return self.server_model

        elif is_agent:
            return self.agent_model

    def get_train_args(self):
        return self.train_args



