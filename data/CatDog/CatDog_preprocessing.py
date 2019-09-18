import csv
import os
import numpy as np
from PIL import Image
import io

summary = [
    ['cat', 0],
    ['dog', 0],
]


def reset():
    global summary
    summary = [
        ['cat', 0],
        ['dog', 0],
    ]


def resize(images_dir_path='test/', image_size=(256, 256)):
    # convert (1024, 1024) to (256, 256)
    print('resize @ {}'.format(images_dir_path))

    image_file_names = os.listdir(images_dir_path)

    for file_name in image_file_names:
        # print('convert {}'.format(file_name))
        file_path = os.path.join(images_dir_path, file_name)
        # read img
        img = Image.open(file_path)
        img = img.resize(image_size)
        img.save(file_path)

    print('convert {} images'.format(len(image_file_names)))


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

                if summary[int(label[0])][1] >= limit:
                    os.remove(file_path)
                    delete += 1
                    # print('too more label : {}, delete {}'.format(label[0], row[0]))
                    continue

                summary[int(label[0])] [1] += 1

    print('delete {} images, remain {}'.format(delete, image_file_nums - delete))
    sum = 0
    for i in range(len(summary)):
        sum += summary[i][1]
        print('{} : {}'.format(summary[i][0], summary[i][1]))
    print('total : {}'.format(sum))


def overview(images_dir_path='sample/'):
    # check the labels of images in the selected directory
    print('overview @ {}'.format(images_dir_path))

    reset()
    # sort file
    sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])

    image_file_names = os.listdir(images_dir_path)
    image_file_names.sort(key=sort_key)
    image_file_nums = len(image_file_names)
    id = 0

    with open('trainLabels.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if id == image_file_nums:
                break
            if row[1] == 'level':
                continue
            summary[int(row[1])][1] += 1

    sum = 0
    for i in range(len(summary)):
        sum += summary[i][1]
        print('class {} : {}'.format(summary[i][0], summary[i][1]))
    print('Total : {}'.format(sum))

resize('train/')

