import os
import pickle
import cv2
import re

from logger import *

DEBUG = False


class File_Processor(Logger):

    def __init__(self):

        self.__logger = self.get_logger(unique_name= __name__,
                                        debug=DEBUG)

        Logger.__init__(self)

    def _read_images_directory(self, images_dir_path, images_dir_idx_ptr_path, sort_key=None, get_images=False,
                               get_image_paths=False, batch_size=None):

        self.__logger.debug('[Get Data]')
        self.__logger.debug('Read Images Directory ( %s ) ...' % images_dir_path)

        image_file_names = os.listdir(images_dir_path)
        image_file_names.sort(key=sort_key)
        image_file_nums = len(image_file_names)

        # if batch size is None, get full batch
        if batch_size is None:
            old_dir_idx_ptr = 0
            batch_size = image_file_nums

        else:
            old_dir_idx_ptr = self.read_nums_from_file(file_path=images_dir_idx_ptr_path)

        # count new directory index pointer by old directory index pointer and batch size, and get batch image file
        # names by old, new directory index pointer
        if old_dir_idx_ptr + batch_size >= image_file_nums:
            new_dir_idx_ptr = batch_size - image_file_nums + old_dir_idx_ptr
            batch_image_file_names = image_file_names[old_dir_idx_ptr:] + image_file_names[:new_dir_idx_ptr]


        else:
            new_dir_idx_ptr = old_dir_idx_ptr + batch_size
            batch_image_file_names = image_file_names[old_dir_idx_ptr: new_dir_idx_ptr]

        # write new directory index pointer numbers to file path: "mages_dir_idx_ptr_path"
        self.write_nums_to_file(file_path=images_dir_idx_ptr_path,
                                nums=new_dir_idx_ptr)

        self.__logger.debug('Totally Read %s Images ...' % len(batch_image_file_names))

        batch_images = []
        batch_image_file_paths = []

        for i, file_name in list(enumerate(batch_image_file_names)):

            image_file_path = os.path.join(images_dir_path, file_name)
            batch_image_file_paths.append(image_file_path)

            if get_images:
                # read image data
                image = cv2.imread(filename=image_file_path)
                batch_images.append(image)

            self.__logger.debug(image_file_path)

            self.__logger.debug('Read %s Images' % int(i + 1))

        self.__logger.debug('Done !')

        if get_images and get_image_paths:
            return batch_images, batch_image_file_paths

        elif get_images:
            return batch_images

        else:
            return batch_image_file_paths

    def get_file_names_from_file_paths(self, file_paths):

        """Get file names from file paths.

        Assuming file path is '~/dir_name/file_name.jpeg', then return file name would be 'file_name'.

        """
        file_names = [re.split(r"[/\\]", file_path)[-1].split('.')[0] for file_path in file_paths]
        return file_names

    @staticmethod
    def write_nums_to_file(file_path, nums):
        with open(file_path, "w") as file:
            file.write(str(nums))

    @staticmethod
    def read_nums_from_file(file_path):
        try:
            with open(file_path, "r") as file:
                nums = int(file.readline())

        except:
            nums = 0

        return nums

    @staticmethod
    def write_list_to_file(file_path, itemlist):
        with open(file_path, 'wb') as fp:
            pickle.dump(itemlist, fp)

    @staticmethod
    def read_list_from_file(file_path):
        try:
            with open(file_path, 'rb') as fp:
                itemlist = pickle.load(fp)

        except:
            itemlist = None

        return itemlist





