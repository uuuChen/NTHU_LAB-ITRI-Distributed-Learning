import csv
import os
from os import walk
import numpy as np
from PIL import Image
import io
import random
from shutil import copyfile
import shutil
import re

def resize(images_dir_path, image_size=(224, 224)):
    print('resize @ {}'.format(images_dir_path))
    count = 0
    for root, dirs, files in walk(images_dir_path):
        for f in files:
            if f == '.DS_Store':
                continue
            if len(re.split(r'[/, \\]', root)) == 3:
                file_path = os.path.join(root, f)
                img = Image.open(file_path)
                if np.array(img).shape != (224, 224, 3):
                    count += 1
                    print(np.array(img).shape)
                    img = img.resize(image_size)
                    img.save(file_path)

    print('convert {} images'.format(count))

def overview(images_dir_path='train/'):
    # check the labels of images in the selected directory
    print('overview @ {}'.format(images_dir_path))
    labels_nums_dict = {}
    personID_labelsNums_dict = {}
    for root, dirs, files in walk(images_dir_path):
        for f in files:
            if len(re.split(r'[/, \\]', root)) == 3:
                personId = re.split(r'[/, \\]', root)[1]
                path = os.path.join(root, f)
                label = re.split(r'[/, \\]', root)[-1]
                if label not in labels_nums_dict.keys():
                    labels_nums_dict[label] = 0
                if personId not in personID_labelsNums_dict.keys():
                    personID_labelsNums_dict[personId] = {}
                if label not in personID_labelsNums_dict[personId].keys():
                    personID_labelsNums_dict[personId][label] = 0
                labels_nums_dict[label] += 1
                personID_labelsNums_dict[personId][label] += 1
    label_0_counter = 0
    label_1_counter = 0
    for personId in sorted(map(int, personID_labelsNums_dict.keys()), reverse=False):
        personId = str(personId)
        label_0_counter += personID_labelsNums_dict[personId]['0']
        label_1_counter += personID_labelsNums_dict[personId]['1']
        print('{}: {:.2f} {:.2f}'.format(personId, 100. * label_0_counter / labels_nums_dict['0'],
                                         100. * label_1_counter / labels_nums_dict['1']))
    print(labels_nums_dict)
    total_nums = sum(labels_nums_dict.values())
    for label in labels_nums_dict.keys():
        print(labels_nums_dict[label] / total_nums)




def pickout(from_path='train/', to_path='test/'):

    print('pick out datas from {} to {}'.format(from_path, to_path))

    labels_file_names_dict = {}
    for root, dirs, files in walk(from_path):
        for f in files:
            print(root)
            if len(re.split(r'[/, \\]', root)) == 3:
                label = root.split('/\\')[-1]
                if label not in labels_file_names_dict.keys():
                    labels_file_names_dict[label] = []
                labels_file_names_dict[label].append(f)
    print(labels_file_names_dict)
    for label in labels_file_names_dict.keys():
        random.shuffle(labels_file_names_dict[label])
        pickout_image_nums = int(len(labels_file_names_dict[label]) * 0.1)
        for i in range(pickout_image_nums):
            pickout_image_name = labels_file_names_dict[label][i]
            from_pickout_image_path = os.path.join(from_path, label, pickout_image_name)
            to_pickout_image_path = os.path.join(to_path, label, pickout_image_name)
            print(from_pickout_image_path)
            shutil.move(from_pickout_image_path, to_pickout_image_path)


# pickout()
resize('train/')
resize('test/')
# overview('train/')
# overview('test/')
# pickout()


