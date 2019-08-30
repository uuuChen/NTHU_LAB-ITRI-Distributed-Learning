# import os
#
# # DataSet Imports
# from dataSet.MNIST_dataSet import MNIST_DataSet
# from dataSet.DRD_dataSet import DRD_DataSet
# from dataSet.Xray_dataSet import Xray_DataSet
# from dataSet.ECG_dataSet import ECG_DataSet
# from data.data_args import *  # import data arguments
#
# class DataSet:
#
#     def __init__(self, ):
#         pass
#
#     @staticmethod
#     def get_dataSet(data_name, is_simulate=False, shuffle=False):
#
#         train_args = {}
#         train_args['is_simulate'] = is_simulate
#         train_args['shuffle'] = shuffle
#         print(globals())
#         train_data_args = globals()['%s_TRAIN_ARGS' % data_name]
#         test_data_args = globals()['%s_TEST_ARGS' % data_name]
#
#         train_data_args.update(train_args)
#         test_data_args.update(train_args)
#
#         train_dataSet = globals()['%s_DataSet' % data_name](data_args=train_data_args)
#         test_dataSet = globals()['%s_DataSet' % data_name](data_args=test_data_args)
#
#         print(train_dataSet.get_data_nums_from_database())
#         return train_dataSet, test_dataSet
#
#
# DataSet.get_dataSet('MNIST')