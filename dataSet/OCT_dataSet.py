import csv

from dataSet.data_proc.data_processor import *

DEBUG = False

class_id = {
    'NORMAL': 0,
    'CNV': 1,
    'DME': 2,
    'DRUSEN': 3
}


class OCT_DataSet(Data_Processor):

    def __init__(self, data_args, shuffle=False):
        data_args['shuffle'] = shuffle
        self.__logger = self.get_logger(__name__, DEBUG)
        self.images_dir_path = data_args['images_dir_path']
        self.images_dir_idx_ptr_path = data_args['images_dir_idx_ptr_path']
        self.label_class_nums = data_args['label_class_nums']
        Data_Processor.__init__(self, data_args=data_args)

    def _get_data_and_labels_from_local(self):
        self.__logger.debug('[Get OCT Data And Labels]')

        from_path = self.images_dir_path
        image_file_names = os.listdir(from_path)
        random.shuffle(image_file_names)

        image_labels = []
        # 讀出所有 label 與圖片對應，再與資料夾中所有圖片名稱對應
        for image_file_name in image_file_names:
            label = image_file_name.split('-')[0]
            image_labels.append(int(class_id[label]))

        # 將資料以配對好的形式洗亂
        image_file_paths = [os.path.join(from_path, image_file_name) for image_file_name in image_file_names]
        self.__logger.debug('[Get DRD Data And Labels Successfully]')

        return image_file_paths, image_labels

    def _get_data_and_labels_from_database(self, batch_size):
        return super()._get_data_and_labels_from_database(batch_size=batch_size)

    def get_data_and_labels(self, batch_size, image_size=(256, 256), data_preprocess=True, toTensor=True, one_hot=False):

        data, labels = self._get_data_and_labels_from_database(batch_size=batch_size)

        if data_preprocess:
            preproc_data = []
            for data_ in data:
                data_ = np.array(Image.fromarray(data_).resize(image_size)) / 255
                data_ = data_.reshape(1, image_size[0], image_size[1])
                preproc_data.append(data_)
            data = np.array(preproc_data)

        if one_hot:
            labels = self._trans_labels_to_one_hot(labels=labels,
                                                   class_nums=self.label_class_nums)

        if toTensor:
            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)

        return data, labels




