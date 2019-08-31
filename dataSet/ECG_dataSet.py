import csv

from dataSet.data_proc.data_processor import *

DEBUG = False


class ECG_DataSet(Data_Processor):

    def __init__(self, data_args):

        self.__logger = self.get_logger(unique_name=__name__,
                                        debug=DEBUG)

        self.data_labels_csv_file_path = data_args['data_labels_csv_file_path']

        self.label_class_nums = data_args['label_class_nums']

        Data_Processor.__init__(self, data_args=data_args)

    def _read_data_and_labels_from_csv_file(self):

        data = []
        labels = []

        with open(self.data_labels_csv_file_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                data.append(list(map(float, row[:-1])))
                labels.append(int(float(row[-1])))

        return data, labels

    def _get_data_and_labels_from_local(self):

        self.__logger.debug('[Get ECG Data And Labels]')

        data, labels = self._read_data_and_labels_from_csv_file()

        self.__logger.debug('[Get ECG Data And Labels Successfully]')

        return data, labels

    def _get_data_and_labels_from_database(self, batch_size):
        return super()._get_data_and_labels_from_database(batch_size=batch_size)

    def get_data_and_labels(self, batch_size, image_size=None, data_preprocess=False, toTensor=True, one_hot=False):

        data, labels = self._get_data_and_labels_from_database(batch_size=batch_size)

        if one_hot:
            labels = self._trans_labels_to_one_hot(labels=labels,
                                                   class_nums=self.label_class_nums)
        if toTensor:
            data, labels = self._trans_data_and_labels_to_tensor(data=data,
                                                                 labels=labels)

        return data, labels


