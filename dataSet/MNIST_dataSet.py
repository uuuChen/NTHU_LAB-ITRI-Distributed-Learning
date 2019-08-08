from torchvision import datasets, transforms

from dataSet.data_proc.data_processor import *

DEBUG = False


class MNIST_DataSet(Data_Processor):

    def __init__(self, data_args, shuffle=False):

        data_args['shuffle'] = shuffle

        self.__logger = self.get_logger(unique_name=__name__,
                                        debug=DEBUG)

        Data_Processor.__init__(self, data_args=data_args)

    def _get_data_and_labels_from_local(self):

        self.__logger.debug('Get MNIST Images And Labels From Tensorflow ...')

        data_train = datasets.MNIST(root="./data/",
                                    transform=transforms.ToTensor(),
                                    train=True,
                                    download=True)
        data_test = datasets.MNIST(root="./data/",
                                   transform=transforms.ToTensor(),
                                   train=False)

        if self.train:
            images = data_train.data.tolist()
            labels = data_train.targets.tolist()
        else:
            images = data_test.data.tolist()
            labels = data_test.targets.tolist()

        self.__logger.debug('Done!')

        return images, labels

    def _get_data_and_labels_from_database(self, batch_size):
        return super()._get_data_and_labels_from_database(batch_size=batch_size)

    def get_data_and_labels(self, batch_size, image_size=None, data_preprocess=True, toTensor=True, one_hot=False):

        data, labels = self.coll_read_all(self.coll_name)

        if data_preprocess:
            datas = []
            for data_ in data:
                data_ = data_.reshape(1, 28, 28)
                datas.append(data_)
            data = np.array(datas)

        if one_hot:
            labels = self._trans_labels_to_one_hot(labels=labels,
                                                   class_nums=self.label_class_nums)

        if toTensor:
            data, labels = self._trans_data_and_labels_to_tensor(data=data,
                                                                 labels=labels)

        return data, labels












