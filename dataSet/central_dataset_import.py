import os
from train.switch import Switch

def load_dataSet(data_name, from_database=False):
    source = '"database"' if from_database else '"local"'
    print("Loading {} training data from {}...".format(data_name, source))
    train_dataSet, test_dataSet = Switch(data_name).get_dataSet()
    train_data_and_labels_nums = (train_dataSet.get_data_nums_from_database() if from_database else
                                  len(train_dataSet._get_data_and_labels_from_local()[0]))
    test_data_and_labels_nums = (test_dataSet.get_data_nums_from_database() if from_database else
                                 len(test_dataSet._get_data_and_labels_from_local()[0]))
    print("train data and labels numbers: {}".format(train_data_and_labels_nums))
    print("test data and labels numbers: {}\n".format(test_data_and_labels_nums))


if __name__ == '__main__':
    os.chdir("../")
    data_names = ['MNIST']
    # data_names = ['MNIST', 'ECG', 'OCT', 'MD']
    for data_name in data_names:
        load_dataSet(data_name, from_database=False)
        load_dataSet(data_name, from_database=True)
