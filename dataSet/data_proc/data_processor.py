import os
import numpy as np
import torch
import random
from PIL import Image
from abc import ABCMeta, abstractmethod

from dataSet.data_proc.database_proc.mongoDB_processor import MongoDB_Processor
from dataSet.data_proc.file_proc.file_processor import File_Processor

DEBUG = False


class Data_Processor(MongoDB_Processor, File_Processor, metaclass=ABCMeta):

    def __init__(self, data_args):

        self.train = data_args['train']

        self.data_name = data_args['data_name']

        self.use_gridFS = data_args['use_gridFS']

        self.data_nums_dir_path = data_args['data_nums_dir_path']

        self.label_class_nums = data_args['label_class_nums']

        self.data_type = data_args['data_type']

        self.coll_name = data_args['db_data_labels_coll_name']

        unique_name = data_args['data_type'] + '_' + self.data_name + '_' + __name__
        self.__logger = self.get_logger(unique_name=unique_name,
                                        debug=DEBUG)

        MongoDB_Processor.__init__(self, data_args)

        File_Processor.__init__(self)

        self._make_sure_data_and_labels_in_database()

        self.db_id_ptr = 0

        self.db_id_list = list(range(1, self.get_data_nums_from_database() + 1))
        if data_args['shuffle']:
            random.shuffle(self.db_id_list)

    def _upload_data_and_labels_to_database(self):

        if self.use_gridFS:
            data_file_paths, labels = self._get_data_and_labels_from_local()

            local_data_nums = len(data_file_paths)

            error_data_nums = self.gridFS_coll_insert(coll_name=self.coll_name,
                                                      data_file_paths=data_file_paths,
                                                      labels=labels)
            local_data_nums -= error_data_nums

        else:
            data, labels = self._get_data_and_labels_from_local()

            local_data_nums = len(data)

            self.coll_insert(coll_name=self.coll_name,
                             data=data,
                             labels=labels)

        self.write_nums_to_file(file_path=os.path.join(self.data_nums_dir_path, '%s_%s.txt' %
                                                       (self.data_name, self.data_type)),
                                nums=local_data_nums)

    def _make_sure_data_and_labels_in_database(self):

        self.__logger.debug('[Make Sure Data And Labels In Database]')

        db_data_nums = self.get_data_nums_from_database()

        local_data_nums = (
            self.read_nums_from_file(file_path=os.path.join(self.data_nums_dir_path, '%s_%s.txt' %
                                                            (self.data_name, self.data_type))))

        if db_data_nums != local_data_nums or db_data_nums == 0:
            if not db_data_nums == 0:
                self.coll_delete_all(coll_name=self.coll_name)

            self._upload_data_and_labels_to_database()

        self.__logger.debug('[Make Sure Data And Labels In Database Successfully]')

    def _get_data_and_labels_from_database(self, batch_size):

        self._make_sure_data_and_labels_in_database()

        data_nums = self.get_data_nums_from_database()

        old_id_ptr = self.db_id_ptr

        if old_id_ptr + batch_size >= data_nums:
            new_id_ptr = batch_size - data_nums + old_id_ptr
            id_list = self.db_id_list[old_id_ptr:] + self.db_id_list[:new_id_ptr]

        else:
            new_id_ptr = old_id_ptr + batch_size
            id_list = self.db_id_list[old_id_ptr: new_id_ptr]

        self.db_id_ptr = new_id_ptr

        if self.use_gridFS:
            data, labels = self.gridFS_coll_read_batch(coll_name=self.coll_name,
                                                       id_list=id_list)

        else:
            data, labels = self.coll_read_batch(coll_name=self.coll_name,
                                                id_list=id_list)

        return data, labels

    def _preproc_image_data(self, data, image_size):

        preproc_data = []

        for data_ in data:
            data_ = np.array(Image.fromarray(data_).resize(image_size)) / 255
            data_ = data_.transpose((2, 0, 1))
            preproc_data.append(data_)

        return np.array(preproc_data)

    def _preproc_non_image_data(self, data):
        pass

    def _trans_labels_to_one_hot(self, labels, class_nums):

        one_hot_labels = np.zeros((labels.shape[0], class_nums))

        for i in range(one_hot_labels.shape[0]):
            one_hot_labels[i, labels[i]] = 1

        return one_hot_labels

    def _trans_data_and_labels_to_tensor(self, data, labels):
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        return data, labels

    def get_data_nums_from_database(self):

        if self.use_gridFS:
            db_data_nums = self.gridFS_coll_find_all(coll_name=self.coll_name).count()

        else:
            db_data_nums = self.coll_find_all(coll_name=self.coll_name).count()

        return db_data_nums

    @abstractmethod
    def _get_data_and_labels_from_local(self):
        raise NotImplementedError('Must override "_get_data_and_labels_from_local()" ')

    @abstractmethod
    def get_data_and_labels(self, batch_size, image_size, data_preprocess, toTensor, one_hot):
        raise NotImplementedError('Must override "get_data_and_labels()" ')



















