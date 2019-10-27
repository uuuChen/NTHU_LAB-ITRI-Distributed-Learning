import csv
import os
from os import walk
import numpy as np
from PIL import Image
import io
import random
from shutil import copyfile
import shutil

OCT_class_id = {
    'NORMAL': 0,
    'CNV': 1,
    'DME': 2,
    'DRUSEN': 3
}

summary = [
    ['NORMAL', 0],
    ['CNV', 0],
    ['DME', 0],
    ['DRUSEN', 0]
]


def reset():
    global summary

    summary = [
        ['NORMAL', 0],
        ['CNV', 0],
        ['DME', 0],
        ['DRUSEN', 0]
    ]


def resize(images_dir_path='train/', image_size=(224, 224)):
    print('resize @ {}'.format(images_dir_path))
    count = 0
    for root, dirs, files in walk(images_dir_path):
        for f in files:
            if f == '.DS_Store':
                continue
            count += 1
            file_path = os.path.join(root, f)
            print(file_path)
            img = Image.open(file_path)
            img = img.resize(image_size)
            img.save(file_path)

    print('convert {} images'.format(count))


def balance(images_dir_path='sample/', limit = 100000):
    # down sampling the selected directory
    print('balance @ {}, limit is {}'.format(images_dir_path, limit))
    reset()
    # sort file
    sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])

    image_file_names = os.listdir(images_dir_path)
    image_file_names.sort(key=sort_key)
    image_file_nums = len(image_file_names)
    id = 0
    delete = 0
    with open('labels.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if id == image_file_nums:
                break

            if row[0] == image_file_names[id]:
                file_path = os.path.join(images_dir_path, image_file_names[id])
                id += 1
                label = row[1].split('|')

                if len(label) > 1:
                    print('Warning : {} has multi label'.format(row[0]))
                    continue

                if summary[OCT_class_id[label[0]]][1] >= limit:
                    os.remove(file_path)
                    delete += 1
                    # print('too more label : {}, delete {}'.format(label[0], row[0]))
                    continue

                summary[OCT_class_id[label[0]]][1] += 1

    print('delete {} images, remain {}'.format(delete, image_file_nums - delete))
    sum = 0
    for i in range(len(summary)):
        sum += summary[i][1]
        print('{} : {}'.format(summary[i][0], summary[i][1]))
    print('total : {}'.format(sum))


def overview(images_dir_path='test/'):
    # check the labels of images in the selected directory
    print('overview @ {}'.format(images_dir_path))

    reset()
    for root, dirs, files in walk(images_dir_path):
      for f in files:
        # path = os.path.join(root, f)
        label = f.split('-')[0]
        summary[OCT_class_id[label]][1] += 1
    sum = 0
    for i in range(len(summary)):
        sum += summary[i][1]
        print('{} : {}'.format(summary[i][0], summary[i][1]))
    print('Total : {}'.format(sum))


def pickout(from_path='train/', to_path='test/'):

    print('pick out datas from {} to {}'.format(from_path, to_path))

    labels_file_names_dict = {}
    for root, dirs, files in walk(from_path):
        for f in files:
            label = root.split('/')[1]
            if label not in labels_file_names_dict.keys():
                labels_file_names_dict[label] = []
            labels_file_names_dict[label].append(f)

    for label in labels_file_names_dict.keys():
        random.shuffle(labels_file_names_dict[label])
        pickout_image_nums = int(len(labels_file_names_dict[label]) * 0.1)
        for i in range(pickout_image_nums):
            pickout_image_name = labels_file_names_dict[label][i]
            from_pickout_image_path = os.path.join(from_path, label, pickout_image_name)
            to_pickout_image_path = os.path.join(to_path, label, pickout_image_name)
            print(from_pickout_image_path)
            shutil.move(from_pickout_image_path, to_pickout_image_path)


resize()
pickout()



