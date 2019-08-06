import os

# --------------------------------
#  ALL dataSet COMMON arguments
# --------------------------------

_GLOBAL_COMMON_ARGS = {
    'client_url': 'mongodb://localhost:27017/',

    'db_train_data_labels_coll_name': 'train_data_labels',

    'db_test_data_labels_coll_name': 'test_data_labels',

    'data_nums_dir_path': 'data/data_nums',

    'train_coll_name': 'train_data_labels',

    'test_coll_name': 'test_data_labels',
}

# --------------------------------
#  PRIVATE dataSet COMMON arguments
# --------------------------------

_MNIST_COMMON_ARGS = {
    'data_name': 'MNIST',

    'dir_path': 'data/MNIST_data/',

    'label_class_nums': 10,

    'use_gridFS': False,
}

_DRD_COMMON_ARGS = {
    'data_name': 'DRD',

    'dir_path': 'data/DRD_data/',

    'label_class_nums': 5,

    'use_gridFS': True,
}

_ECG_COMMON_ARGS = {
    'data_name': 'ECG',

    'dir_path': 'data/ECG_data/',

    'label_class_nums': 5,

    'use_gridFS': False,
}

_MNIST_COMMON_ARGS.update(_GLOBAL_COMMON_ARGS)

_DRD_COMMON_ARGS.update(_GLOBAL_COMMON_ARGS)

_ECG_COMMON_ARGS.update(_GLOBAL_COMMON_ARGS)

# --------------------------------
#  PRIVATE dataSet PRIVATE arguments
# --------------------------------

MNIST_TRAIN_ARGS = {
    'train': True,

    'data_type': 'train',

    'coll_name': _GLOBAL_COMMON_ARGS['train_coll_name'],

    'db_id_ptr_file_path': os.path.join(_MNIST_COMMON_ARGS['dir_path'], 'db_train_id_ptr.txt'),

    'db_id_list_file_path': os.path.join(_MNIST_COMMON_ARGS['dir_path'], 'db_train_id_list.txt'),
}

MNIST_TEST_ARGS = {
    'train': False,

    'data_type': 'test',

    'coll_name': _GLOBAL_COMMON_ARGS['test_coll_name'],
}

DRD_TRAIN_ARGS = {
    'train': True,

    'data_type': 'train',

    'images_dir_path': os.path.join(_DRD_COMMON_ARGS['dir_path'], 'sample'),

    'images_dir_idx_ptr_path': os.path.join(_DRD_COMMON_ARGS['dir_path'], 'train_images_dir_idx_ptr.txt'),

    'labels_csv_file_path': os.path.join(_DRD_COMMON_ARGS['dir_path'], 'trainLabels.csv'),

    'coll_name': _GLOBAL_COMMON_ARGS['train_coll_name'],
}

DRD_TEST_ARGS = {
    'train': False,

    'data_type': 'test',

    'images_dir_path': os.path.join(_DRD_COMMON_ARGS['dir_path'], 'sample'),

    'images_dir_idx_ptr_path': os.path.join(_DRD_COMMON_ARGS['dir_path'], 'test_images_dir_idx_ptr.txt'),

    'labels_csv_file_path': os.path.join(_DRD_COMMON_ARGS['dir_path'], 'trainLabels.csv'),

    'coll_name': _GLOBAL_COMMON_ARGS['test_coll_name'],
}

ECG_TRAIN_ARGS = {
    'train': True,

    'data_type': 'train',

    'data_labels_csv_file_path': os.path.join(_ECG_COMMON_ARGS['dir_path'], 'heartbeat/mitbih_train.csv'),

    'coll_name': _GLOBAL_COMMON_ARGS['train_coll_name'],
}

ECG_TEST_ARGS = {
    'train': False,

    'data_type': 'test',

    'data_labels_csv_file_path': os.path.join(_ECG_COMMON_ARGS['dir_path'], 'heartbeat/mitbih_test.csv'),

    'coll_name': _GLOBAL_COMMON_ARGS['test_coll_name'],
}

# --------------------------------
#  merge dataSET COMMON and PRIVATE arguments
# --------------------------------

MNIST_TRAIN_ARGS.update(_MNIST_COMMON_ARGS)
MNIST_TEST_ARGS.update(_MNIST_COMMON_ARGS)

DRD_TRAIN_ARGS.update(_DRD_COMMON_ARGS)
DRD_TEST_ARGS.update(_DRD_COMMON_ARGS)

ECG_TRAIN_ARGS.update(_ECG_COMMON_ARGS)
ECG_TEST_ARGS.update(_ECG_COMMON_ARGS)



