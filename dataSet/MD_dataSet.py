import re

from dataSet.data_proc.data_processor import *

DEBUG = False


class MD_DataSet(Data_Processor):

    def __init__(self, data_args):

        self.__logger = self.get_logger(unique_name=__name__, debug=DEBUG)

        self.images_dir_path = data_args['images_dir_path']

        self.images_dir_idx_ptr_path = data_args['images_dir_idx_ptr_path']

        self.label_class_nums = data_args['label_class_nums']

        self.label_str2int = data_args['class_id']

        Data_Processor.__init__(self, data_args=data_args)

    def _get_data_and_labels_from_local(self):

        self.__logger.debug('[Get MD Data And Labels]')

        image_file_paths = []
        image_labels = []
        for root, dirs, file_names in os.walk(self.images_dir_path):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                label_str = re.split(r'[/, \\]', root)[-1]
                label_int = self.label_str2int[label_str]
                image_file_paths.append(file_path)
                image_labels.append(label_int)

        self.__logger.debug('[Get MD Data And Labels Successfully]')

        return image_file_paths, image_labels

    def _get_data_and_labels_from_database(self, batch_size):
        return super()._get_data_and_labels_from_database(batch_size=batch_size)

    def get_data_and_labels(self, batch_size, image_size=None, data_preprocess=True, toTensor=True, one_hot=False):

        data, labels = self._get_data_and_labels_from_database(batch_size=batch_size)

        if data_preprocess:
            data = self._preproc_image_data(data)

        if one_hot:
            labels = self._trans_labels_to_one_hot(labels=labels,
                                                   class_nums=self.label_class_nums)

        if toTensor:
            data, labels = self._trans_data_and_labels_to_tensor(data=data,
                                                                 labels=labels)

        return data, labels





