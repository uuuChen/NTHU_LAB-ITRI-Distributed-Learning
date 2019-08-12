import csv

from dataSet.data_proc.data_processor import *

DEBUG = False

Xray_class_id = {
    'Hernia': 0,
    'Pneumonia': 1,
    'Fibrosis': 2,
    'Edema': 3,
    'Emphysema': 4,
    'Cardiomegaly': 5,
    'Pleural_Thickening': 6,
    'Consolidation': 7,
    'Pneumothorax': 8,
    'Mass': 9,
    'Nodule': 10,
    'Atelectasis': 11,
    'Effusion': 12,
    'Infiltration': 13,
    'No Finding': 14,
}

class Xray_DataSet(Data_Processor):

    def __init__(self, data_args, shuffle=False):

        data_args['shuffle'] = shuffle

        self.__logger = self.get_logger(unique_name=__name__,
                                        debug=DEBUG)

        self.images_dir_path = data_args['images_dir_path']

        self.images_dir_idx_ptr_path = data_args['images_dir_idx_ptr_path']

        self.labels_csv_file_path = data_args['labels_csv_file_path']

        self.label_class_nums = data_args['label_class_nums']

        Data_Processor.__init__(self, data_args=data_args)

    def _read_labels_csv_file(self, csv_file_path, image_file_paths):

        """Read the labels corresponding to image_file_paths from csv file which path is "csv_file_path".

        The format of the csv file should be:

            Title:
                file_name,label

            Row:
                10_right,0
                13_left,0
                etc.

        """

        self.__logger.debug('[Get Labels]')
        self.__logger.debug('Read CSV Labels ( %s ) ...' % csv_file_path)

        image_file_names = self.get_file_names_from_file_paths(file_paths=image_file_paths)

        labels = []

        with open(csv_file_path, newline='') as csvfile:
            read_image_files = 0  # numbers of image files read
            rows = csv.reader(csvfile)

            for row in rows:
                file_name = row[0]
                # make file name from '00030183_004.png' to '00030183_004'
                file_name = file_name.split('.')
                file_name = file_name[0]

                # if csv file name matches image file name, the label of the former will be stored in labels (list)
                if file_name == image_file_names[read_image_files]:  # image_file_name has to remove str '.jpg'
                    label = row[1].split('|')
                    label_id = []
                    for i in range(len(label)):
                        label_id.append(Xray_class_id[label[i]])
                    labels.append(label_id)  # store the label

                    read_image_files += 1
                    if read_image_files == len(image_file_names):  # if numbers of image files read equals numbers of
                                                                   # batch images, then break
                        break

        self.__logger.debug('Done !')

        return labels

    def _get_data_and_labels_from_local(self):

        self.__logger.debug('[Get DRD Data And Labels]')

        image_names_sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])
        image_file_paths = self._read_images_directory(images_dir_path=self.images_dir_path,
                                                       images_dir_idx_ptr_path=self.images_dir_idx_ptr_path,
                                                       sort_key=image_names_sort_key,
                                                       get_image_paths=True)

        image_labels = self._read_labels_csv_file(csv_file_path=self.labels_csv_file_path,
                                                  image_file_paths=image_file_paths)

        self.__logger.debug('[Get DRD Data And Labels Successfully]')

        return image_file_paths, image_labels

    def _get_data_and_labels_from_database(self, batch_size):
        return super()._get_data_and_labels_from_database(batch_size=batch_size)

    def get_data_and_labels(self, batch_size, image_size, data_preprocess=True, toTensor=True, one_hot=False):

        data, labels = self._get_data_and_labels_from_database(batch_size=batch_size)

        if data_preprocess:
            preproc_data = []

            for data_ in data:
                data_ = np.array(Image.fromarray(data_).resize(image_size)) / 255
                data_ = data_.reshape(1,100,100)
                data_ = data_.transpose((2, 0, 1))
                preproc_data.append(data_)

            data = np.array(preproc_data)

        if one_hot:
            labels = self._trans_labels_to_one_hot(labels=labels,
                                                    class_nums=self.label_class_nums)

        if toTensor:
            data = torch.from_numpy(data)
            for i in range(len(labels)):
                labels[i] = torch.from_numpy(labels[i])

        return data, labels




