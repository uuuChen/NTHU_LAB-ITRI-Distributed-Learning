
#import four kind of datasets
from dataSet.MNIST_dataSet import *
from dataSet.DRD_dataSet import *
from dataSet.Xray_dataSet import *
from dataSet.ECG_dataSet import *
from data.data_args import *  # import data arguments

import csv

with open('data/Xray/sample_labels.csv', newline='') as csvfile:
    # DRD database insert
    train_dataSet = DRD_DataSet(data_args=DRD_TRAIN_ARGS,
                                shuffle=True)
    test_dataSet = DRD_DataSet(data_args=DRD_TEST_ARGS,
                               shuffle=True)

    # local train
    print("Loading DRD training data from local...")
    data, lables = train_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables),len(data)))

    # local test
    print("Loading DRD testing data from local...")
    data, lables = test_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables),len(data)))

    # database train
    print("Loading DRD training data from database...")
    data_num = train_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))

    # database test
    print("Loading DRD testing data from database...")
    data_num = test_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))


    # Xray database insert
    train_dataSet = Xray_DataSet(data_args=Xray_TRAIN_ARGS,
                                shuffle=True)
    test_dataSet = Xray_DataSet(data_args=Xray_TEST_ARGS,
                               shuffle=True)

    # local train
    print("Loading Xray training data from local...")
    data, lables = train_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables), len(data)))

    # local test
    print("Loading Xray testing data from local...")
    data, lables = test_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables), len(data)))

    # database train
    print("Loading Xray training data from database...")
    data_num = train_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))

    # database test
    print("Loading Xray testing data from database...")
    data_num = test_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))

    #ECG database insert
    train_dataSet = ECG_DataSet(data_args=ECG_TRAIN_ARGS,
                                shuffle=True)
    test_dataSet = ECG_DataSet(data_args=ECG_TEST_ARGS,
                               shuffle=True)

    # local train
    print("Loading ECG training data from local...")
    data, lables = train_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables), len(data)))

    # local test
    print("Loading ECG testing data from local...")
    data, lables = test_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables), len(data)))

    # database train
    print("Loading ECG training data from database...")
    data_num = train_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))

    # database test
    print("Loading ECG testing data from database...")
    data_num = test_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))


    #MNIST database insert
    train_dataSet = MNIST_DataSet(data_args=MNIST_TRAIN_ARGS,
                                shuffle=True)
    test_dataSet = MNIST_DataSet(data_args=MNIST_TEST_ARGS,
                               shuffle=True)

    # local train
    print("Loading MNIST training data from local...")
    data, lables = train_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables), len(data)))

    # local test
    print("Loading MNIST testing data from local...")
    data, lables = test_dataSet._get_data_and_labels_from_local()
    print("images number: {}\nlabels number: {}".format(len(lables), len(data)))

    # database train
    print("Loading MNIST training data from database...")
    data_num = train_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))

    # database test
    print("Loading MNIST testing data from database...")
    data_num = test_dataSet.get_data_nums_from_database()
    print("data number: {}".format(data_num))