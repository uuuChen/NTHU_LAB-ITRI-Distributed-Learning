from train.switch import Switch

def delete_dataSet_in_database(data_name, train=False):
    source = '"train_data_labels"' if train else '"test_data_labels"'
    print("deleting {} from {}...".format(source, data_name))
    train_dataSet, test_dataSet = Switch(data_name).get_dataSet()
    train_dataSet.drop_coll_from_database()
    print('drop "train dataSet" done !')
    test_dataSet.drop_coll_from_database()
    print('drop "test dataSet" done !')


if __name__ == '__main__':
    data_names = ['MNIST', 'ECG', 'OCT', 'MD']
    for data_name in data_names:
        delete_dataSet_in_database(data_name, train=False)
        delete_dataSet_in_database(data_name, train=True)