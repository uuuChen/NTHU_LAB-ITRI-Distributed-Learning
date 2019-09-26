import csv
import os
import numpy as np
from PIL import Image
import io
import random
from shutil import copyfile

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

summary = [
    ['Hernia', 0],
    ['Pneumonia', 0],
    ['Fibrosis', 0],
    ['Edema', 0],
    ['Emphysema', 0],
    ['Cardiomegaly', 0],
    ['Pleural_Thickening', 0],
    ['Consolidation', 0],
    ['Pneumothorax', 0],
    ['Mass', 0],
    ['Nodule', 0],
    ['Atelectasis', 0],
    ['Effusion', 0],
    ['Infiltration', 0],
    ['No Finding', 0],
]


def reset():
    global summary
    summary = [
        ['Hernia', 0],
        ['Pneumonia', 0],
        ['Fibrosis', 0],
        ['Edema', 0],
        ['Emphysema', 0],
        ['Cardiomegaly', 0],
        ['Pleural_Thickening', 0],
        ['Consolidation', 0],
        ['Pneumothorax', 0],
        ['Mass', 0],
        ['Nodule', 0],
        ['Atelectasis', 0],
        ['Effusion', 0],
        ['Infiltration', 0],
        ['No Finding', 0],
    ]


def resize(images_dir_path='sample/', image_size=(256, 256)):
    # convert (1024, 1024) to (256, 256)
    print('resize @ {}'.format(images_dir_path))

    image_file_names = os.listdir(images_dir_path)

    i = 0
    for file_name in image_file_names:
        i += 1
        print('convert {}/{}'.format(i, len(image_file_names)))
        file_path = os.path.join(images_dir_path, file_name)
        # read img
        img = Image.open(file_path)
        img = img.resize(image_size)
        img.save(file_path)

    print('convert {} images'.format(len(image_file_names)))


def to_gray(images_dir_path='sample/'):
    # convert (1024, 1024, 4) to (1024, 1204)
    print('to_gray @ {}'.format(images_dir_path))

    image_file_names = os.listdir(images_dir_path)

    convert = 0
    i = 0
    for file_name in image_file_names:
        i += 1
        print('convert {}/{}'.format(i, len(image_file_names)))
        file_path = os.path.join(images_dir_path, file_name)
        # read img
        img = Image.open(file_path)
        # covert img to np
        data = np.array(img)
        if len(data.shape) > 2:
            data = data[:, :, 0:1]
            data = data.reshape(data.shape[0], -1)
            # convert np to img
            img = Image.fromarray(data)
            # replace the original img by preprocessed img
            img.save(file_path)

            convert += 1
            print('convert {}'.format(file_name))

    print('convert {} images, total {} images'.format(convert, len(image_file_names)))


def to_3_channel(images_dir_path='sample/'):
    # convert (1024, 1024, 4) to (1024, 1204)
    print('to_gray @ {}'.format(images_dir_path))

    image_file_names = os.listdir(images_dir_path)

    convert = 0
    t = 0
    for file_name in image_file_names:
        t += 1
        print('convert {}/{}'.format(t, len(image_file_names)))
        file_path = os.path.join(images_dir_path, file_name)
        # read img
        img = Image.open(file_path)
        # covert img to np
        data = np.array(img)

        if len(data.shape) < 3:
            data_ = np.zeros(((data.shape[0], data.shape[1], 3)))
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data_[i][j][0] = data[i][j]
                    data_[i][j][1] = data[i][j]
                    data_[i][j][2] = data[i][j]
            data = np.array(data_)
            # convert np to img
            img = Image.fromarray(np.uint8(data))
            # replace the original img by preprocessed img
            img.save(file_path)

            convert += 1
            print('convert {}'.format(file_name))

    print('convert {} images, total {} images'.format(convert, len(image_file_names)))


def delete_multi_label(images_dir_path='sample/'):
    print('delete_multi_label @ {}'.format(images_dir_path))
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
            # match the img in directory with label in csv
            if row[0] == image_file_names[id]:
                file_path = os.path.join(images_dir_path, image_file_names[id])
                id += 1
                label = row[1].split('|')
                # remove the img with more than one label
                if len(label) > 1:
                    os.remove(file_path)
                    delete += 1

                if id == image_file_nums:
                    break

    print('delete {} images, remain {} images'.format(delete, image_file_nums - delete))


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

                if summary[Xray_class_id[label[0]]][1] >= limit:
                    os.remove(file_path)
                    delete += 1
                    # print('too more label : {}, delete {}'.format(label[0], row[0]))
                    continue

                summary[Xray_class_id[label[0]]][1] += 1

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
    with open('labels.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if id == image_file_nums:
                break

            image_csv_name = row[0].split('_')
            image_file_name = image_file_names[id].split('_')
            while image_csv_name[0] == image_file_name[0] and image_csv_name[1].split('.')[0] == image_file_name[1].split('.')[0]:
                label = row[1].split('|')
                if len(label) > 1:
                    print('Warning : {} has multi label'.format(row[0]))
                    continue
                summary[Xray_class_id[label[0]]][1] += 1
                id += 1
                if id == image_file_nums:
                    break
                image_file_name = image_file_names[id].split('_')
    sum = 0
    for i in range(len(summary)):
        sum += summary[i][1]
        print('{} : {}'.format(summary[i][0], summary[i][1]))
    print('Total : {}'.format(sum))


def pickout(from_path='images_01~11/', to_path='sample_4/', limit=2000, seed=1):

    print('pick out datas from {} to {}'.format(from_path, to_path))

    # reset()
    # sort file
    sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])

    image_file_names = os.listdir(from_path)
    image_file_names.sort(key=sort_key)
    image_file_nums = len(image_file_names)

    image_labels = []
    id = 0
    with open('labels.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if id == image_file_nums:
                break
            image_csv_name = row[0].split('_')
            image_file_name = image_file_names[id].split('_')
            while image_csv_name[0] == image_file_name[0] and image_csv_name[1].split('.')[0] == image_file_name[1].split('.')[0]:
                label = row[1].split('|')
                if len(label) > 1:
                    print('Warning : {} has multi label'.format(row[0]))
                    continue
                image_labels.append(label[0])
                id += 1
                if id == image_file_nums:
                    break
                image_file_name = image_file_names[id].split('_')

    data = list(zip(image_file_names, image_labels))
    random.seed(seed)
    random.shuffle(data)
    list(zip(*data))

    for data_ in data:
        if summary[Xray_class_id[data_[1]]][1] < limit:
            src = os.path.join(from_path, data_[0])
            dest = os.path.join(to_path, data_[0])
            copyfile(src, dest)
            # os.remove(src)
            summary[Xray_class_id[data_[1]]][1] += 1


resize(images_dir_path='sample', image_size=(224, 224))
to_3_channel('sample')




