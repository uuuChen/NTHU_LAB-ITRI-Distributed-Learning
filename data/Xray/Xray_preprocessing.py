import csv
import os
import numpy as np
from PIL import Image
import io

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


def resize(images_dir_path='sample/', image_size = (256,256)):
    # convert (1024, 1024) to (256, 256)

    # sort file
    sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])

    image_file_names = os.listdir(images_dir_path)
    image_file_names.sort(key=sort_key)

    for file_name in image_file_names:
        print('convert {}'.format(file_name))
        file_path = os.path.join(images_dir_path, file_name)
        # read img
        img = Image.open(file_path)
        img.thumbnail(image_size)
        img.save(file_path)

    print('convert {} images'.format(len(image_file_names)))


def to_gray(images_dir_path='sample/'):
    # convert (1024, 1024, 4) to (1024, 1204)

    # sort file
    sort_key = lambda x: (int(x.split('_')[0]), x.split('_')[1])

    image_file_names = os.listdir(images_dir_path)
    image_file_names.sort(key=sort_key)

    convert = 0
    for file_name in image_file_names:
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


def delete_multi_label(images_dir_path='sample/'):
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
                    print('too more label : {}, delete {}'.format(label[0], row[0]))
                    continue

                summary[Xray_class_id[label[0]]][1] += 1

    print('delete {} images'.format(delete))
    sum = 0
    for i in range(len(summary)):
        sum += summary[i][1]
        print('{} : {}'.format(summary[i][0], summary[i][1]))
    print('total : {}'.format(sum))


def overview(images_dir_path='sample/'):
    # check the labels of images in the selected directory

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

            if row[0] == image_file_names[id]:
                label = row[1].split('|')
                id += 1
                if len(label) > 1:
                    print('Warning : {} has multi label'.format(row[0]))
                    continue
                summary[Xray_class_id[label[0]]][1] += 1

    sum = 0
    for i in range(len(summary)):
        sum += summary[i][1]
        print('{} : {}'.format(summary[i][0], summary[i][1]))
    print('total : {}'.format(sum))


resize()
to_gray()
delete_multi_label()
balance(limit=2)


