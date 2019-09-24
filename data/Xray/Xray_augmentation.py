# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import csv
import random

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

Xray_Augmentation = {''}
# datagen = ImageDataGenerator(
#     rotation_range=0.2,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

from_path = 'images_1~11'
to_path = 'train'
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

        if row[0] == image_file_names[id]:
            label = row[1]
            image_labels.append(label)
            id += 1
data = list(zip(image_file_names, image_labels))
random.seed(1)
random.shuffle(data)
list(zip(*data))

for data_ in data:
    if data_[1] == 'Emphysema':
        img_path = os.path.join(from_path, data_[0])
        img = load_img(img_path)  # this is a PIL image, please replace to your own file path
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images

        datagen = ImageDataGenerator(
            brightness_range=[0.2, 1.0],
            horizontal_flip=True,
            fill_mode='nearest'
        )
        i = 0
        print(data_[0])
        for batch in datagen.flow(x, batch_size=1, save_to_dir=to_path, save_prefix=data_[0].split('.')[0], save_format='png'):
            i += 1
            if i > 0:
                break  # otherwise the generator would loop indefinitely