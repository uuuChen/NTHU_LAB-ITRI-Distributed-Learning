from tensorflow.examples.tutorials.mnist import input_data

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

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

        if self.train:
            images = mnist.train.images.tolist()
            labels = mnist.train.labels.tolist()

        else:
            images = mnist.test.images.tolist()
            labels = mnist.test.labels.tolist()

        self.__logger.debug('Done!')

        return images, labels

    def _get_data_and_labels_from_database(self, batch_size):
        return super()._get_data_and_labels_from_database(batch_size=batch_size)

    def get_data_and_labels(self, batch_size, toTensor=True, one_hot=True):

        data, labels = self._get_data_and_labels_from_database(batch_size=batch_size)

        if one_hot:
            labels = self._trans_labels_to_one_hot(labels=labels,
                                                   class_nums=self.label_class_nums)

        if toTensor:
            data, labels = self._trans_data_and_labels_to_tensor(data=data,
                                                                 labels=labels)

        return data, labels












