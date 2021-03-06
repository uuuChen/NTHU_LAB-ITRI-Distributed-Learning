import numpy as np
import torch
import random
import collections
from PIL import Image
from abc import ABCMeta, abstractmethod
import os

from dataSet.data_proc.database_proc.mongoDB_processor import MongoDB_Processor
from dataSet.data_proc.file_proc.file_processor import File_Processor

DEBUG = False


class Data_Processor(MongoDB_Processor, File_Processor, metaclass=ABCMeta):

    def __init__(self, data_args):
        self.train = data_args['train']
        self.data_name = data_args['data_name']
        self.use_gridFS = data_args['use_gridFS']

        if 'is_simulate' in data_args.keys():
            self.is_simulate = data_args['is_simulate']
        else:
            self.is_simulate = False

        if 'shuffle' in data_args.keys():
            self.shuffle = data_args['shuffle']
        else:
            self.shuffle = False

        self.data_nums_dir_path = data_args['data_nums_dir_path']
        if not os.path.isdir(self.data_nums_dir_path):
            os.mkdir(self.data_nums_dir_path)

        self.label_class_nums = data_args['label_class_nums']
        self.data_type = data_args['data_type']
        self.coll_name = data_args['db_data_labels_coll_name']
        self.down_sampling = data_args['down_sampling']

        unique_name = data_args['data_type'] + '_' + self.data_name + '_' + __name__
        self.__logger = self.get_logger(unique_name, DEBUG)

        MongoDB_Processor.__init__(self, data_args)
        File_Processor.__init__(self)

        self._make_sure_data_and_labels_in_database()

        self.data_id_ptr = 0
        if self.down_sampling:
            usage_data_ids = self._down_sampling(benchmark_idx=data_args['down_sampling_benchmark'],
                                                 remove_classes=data_args['remove_classes'])
        else:
            usage_data_ids = list(range(1, self.get_data_nums_from_database() + 1))
        self.usage_data_ids = usage_data_ids
        if self.shuffle:
            random.shuffle(self.usage_data_ids)

    def _upload_data_and_labels_to_database(self, data_or_data_path, labels, up_sampling=False):

        local_data_nums = len(data_or_data_path)

        if up_sampling:
            local_data_nums += self.get_data_nums_from_database()

        if self.use_gridFS:
            error_data_nums = self.gridFS_coll_insert(self.coll_name, data_or_data_path, labels)
            local_data_nums -= error_data_nums
        else:
            self.coll_insert(self.coll_name, data_or_data_path, labels)

        self.write_nums_to_file(os.path.join(self.data_nums_dir_path, '%s_%s.txt' % (self.data_name, self.data_type)),
                                local_data_nums)

    def _make_sure_data_and_labels_in_database(self):

        self.__logger.debug('[Make Sure Data And Labels In Database]')

        db_data_nums = self.get_data_nums_from_database()
        local_data_nums = (
            self.read_nums_from_file(os.path.join(self.data_nums_dir_path, '%s_%s.txt' %
                                                  (self.data_name, self.data_type))))

        if db_data_nums != local_data_nums or db_data_nums == 0:
            if not db_data_nums == 0:
                self.drop_coll_from_database()
            data_or_data_path, labels = self._get_data_and_labels_from_local()
            self._upload_data_and_labels_to_database(data_or_data_path, labels)

        self.__logger.debug('[Make Sure Data And Labels In Database Successfully]')

    def _get_data_and_labels_from_database(self, batch_size):
        usage_data_nums = self.get_usage_data_nums()
        old_id_ptr = self.data_id_ptr
        if old_id_ptr + batch_size >= usage_data_nums:
            new_id_ptr = 0
            id_list = self.usage_data_ids[old_id_ptr:]
        else:
            new_id_ptr = old_id_ptr + batch_size
            id_list = self.usage_data_ids[old_id_ptr: new_id_ptr]
        self.data_id_ptr = new_id_ptr
        if self.use_gridFS:
            data, labels = self.gridFS_coll_read_batch(self.coll_name, id_list)
        else:
            data, labels = self.coll_read_batch(self.coll_name, id_list)

        return data, labels

    def _preproc_image_data(self, data, resize=False, image_size=None):
        preproc_data = []
        for data_ in data:
            data_ = np.array(Image.fromarray(data_))
            if resize:
                data_ = data_.resize(image_size)
            data_ = data_.transpose((2, 0, 1)) / 255
            preproc_data.append(data_)
        return np.array(preproc_data)

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
            db_data_nums = self.gridFS_coll_find_all(self.coll_name).count()
        else:
            db_data_nums = self.coll_find_all(self.coll_name).count()
        return db_data_nums

    def get_all_labels_from_database(self):

        """ Get all labels in database that have not been shuffled. """

        if self.use_gridFS:
            all_labels = self.gridFS_coll_read_all_labels(self.coll_name)
        else:
            all_labels = self.coll_read_all_labels(self.coll_name)
        return all_labels

    def drop_coll_from_database(self):
        if self.use_gridFS:
            self.gridFS_coll_delete_all(self.coll_name)

        else:
            self.coll_delete_all(self.coll_name)

    def set_usage_data_ids(self, id_list):

        """ "usage_data_ids" is used in function "_get_data_and_labels_from_database". It can decide to read the data
            and labels that match these ids from the database.

        """

        self.usage_data_ids = id_list

    def get_usage_data_nums(self):
        return len(self.usage_data_ids)

    def _down_sampling(self, benchmark_idx=-1, remove_classes=2):
        labels = self.get_all_labels_from_database()
        labels_nums = {}
        labels_idxs = {}
        sample_labels_nums = {}
        sample_labels_idxs = []
        idx = 0
        if not (1 <= abs(benchmark_idx) <= self.label_class_nums):
            print('bench_mark index exceeds categories numbers! change the index from ({}) to ({})'.
                  format(benchmark_idx, self.label_class_nums))
            benchmark_idx = self.label_class_nums
        if benchmark_idx < 0:
            benchmark_idx += (self.label_class_nums + 1)
        for label in labels:
            if label not in labels_nums.keys():
                labels_nums[label] = 0
                labels_idxs[label] = []
            labels_nums[label] += 1
            labels_idxs[label].append(idx)
            idx += 1
        print('origin labels nums: {}'.format(labels_nums))  # >> {0: 6149, 1: 588, 2: 1283, 4: 166, 3: 221}
        sorted_labels_nums_list = sorted(labels_nums.items(), key=lambda x: x[1], reverse=True)  # 由資料數量高排到低
        benchmark_sample_nums = sorted_labels_nums_list[benchmark_idx - 1][1]
        sorted_labels_nums = collections.OrderedDict(sorted_labels_nums_list)
        total_classes_nums = len(sorted_labels_nums.keys())
        for sorted_cat_idx, label in list(enumerate(sorted_labels_nums.keys(), start=1)):
            if sorted_cat_idx <= benchmark_idx:
                sample_labels_nums[label] = benchmark_sample_nums
            else:
                sample_labels_nums[label] = sorted_labels_nums[label]
            if remove_classes > 0:
                # 將最低 remove_classes 數量的 labels 的 sample_lables_nums 設為 0
                if sorted_cat_idx > total_classes_nums - remove_classes:
                    sample_labels_nums[label] = 0
        print('sample labels nums: {}'.format(dict(sorted(sample_labels_nums.items(), key=lambda x: x[0]))))  # >> {0:
        # 166, 1: 166, 2: 166, 3: 166, 4: 166}
        for label in labels_idxs.keys():
            random.shuffle(labels_idxs[label])
            sample_labels_idxs += labels_idxs[label][:sample_labels_nums[label]]
        sample_labels_idxs.sort()
        return sample_labels_idxs

    @abstractmethod
    def _get_data_and_labels_from_local(self):
        raise NotImplementedError('Must override "_get_data_and_labels_from_local()" ')

    @abstractmethod
    def get_data_and_labels(self, batch_size, image_size, data_preprocess, toTensor, one_hot):
        raise NotImplementedError('Must override "get_data_and_labels()" ')



















