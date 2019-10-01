import os

# --------------------------------
#  ALL dataSet COMMON arguments
# --------------------------------

GLOBAL_COMMON_ARGS = {
    'client_url': 'mongodb://localhost:27017/',

    'db_train_data_labels_coll_name': 'train_data_labels',

    'db_test_data_labels_coll_name': 'test_data_labels',

    'data_nums_dir_path': 'data/data_nums',

}

# --------------------------------
#  PRIVATE dataSet COMMON arguments
# --------------------------------

MNIST_COMMON_ARGS = {
    'data_name': 'MNIST',

    'dir_path': 'data/MNIST/',

    'label_class_nums': 10,

    'data_length': 28 * 28,

    'use_gridFS': False,

    'down_sampling': False,
}

DRD_COMMON_ARGS = {
    'data_name': 'DRD',

    'dir_path': 'data/DRD_data/',

    'label_class_nums': 5,

    'use_gridFS': True,
}

ECG_COMMON_ARGS = {
    'data_name': 'ECG',

    'dir_path': 'data/ECG_data/',

    'label_class_nums': 5,

    'data_length': 187,

    'MLP_conn_node_nums': 500,

    'use_gridFS': False,

    'down_sampling': False,

}


Xray_COMMON_ARGS = {
    'data_name': 'Xray',

    'dir_path': 'data/Xray/',

    'label_class_nums': 15,

    'use_gridFS': True,
}

CatDog_COMMON_ARGS = {
    'data_name': 'CatDog',

    'dir_path': 'data/CatDog/',

    'label_class_nums': 2,

    'use_gridFS': True,
}

MNIST_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

DRD_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

ECG_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

Xray_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

CatDog_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)


# --------------------------------
#  PRIVATE dataSet PRIVATE arguments
# --------------------------------

MNIST_TRAIN_ARGS = {
    'train': True,

    'data_type': 'train',

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_train_data_labels_coll_name'],

    'db_id_ptr_file_path': os.path.join(MNIST_COMMON_ARGS['dir_path'], 'db_train_id_ptr.txt'),

    'db_id_list_file_path': os.path.join(MNIST_COMMON_ARGS['dir_path'], 'db_train_id_list.txt'),
}

MNIST_TEST_ARGS = {
    'train': False,

    'data_type': 'test',

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_test_data_labels_coll_name'],
}

DRD_TRAIN_ARGS = {
    'train': True,

    'down_sampling': True,

    'data_type': 'train',

    'images_dir_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'sample'),
    # 'images_dir_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'train'),

    'images_dir_idx_ptr_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'train_images_dir_idx_ptr.txt'),

    'labels_csv_file_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'trainLabels.csv'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_train_data_labels_coll_name'],
}

DRD_TEST_ARGS = {
    'train': False,

    'down_sampling': False,

    'data_type': 'test',

    'images_dir_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'sample'),
    # 'images_dir_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'test/train/train'),

    'images_dir_idx_ptr_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'test_images_dir_idx_ptr.txt'),

    'labels_csv_file_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'trainLabels.csv'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_test_data_labels_coll_name'],
}

DRD_TESTING_ARGS = {
    'train': True,

    'data_type': 'train',

    'images_dir_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'train'),

    'images_dir_idx_ptr_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'train_images_dir_idx_ptr.txt'),

    'labels_csv_file_path': os.path.join(DRD_COMMON_ARGS['dir_path'], 'trainLabels.csv'),

    'db_data_labels_coll_name': 'testing_data_labels',

}

ECG_TRAIN_ARGS = {
    'train': True,

    'data_type': 'train',

    'data_labels_csv_file_path': os.path.join(ECG_COMMON_ARGS['dir_path'], 'heartbeat/mitbih_train.csv'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_train_data_labels_coll_name'],
}

ECG_TEST_ARGS = {
    'train': False,

    'data_type': 'test',

    'data_labels_csv_file_path': os.path.join(ECG_COMMON_ARGS['dir_path'], 'heartbeat/mitbih_test.csv'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_test_data_labels_coll_name'],
}

Xray_TRAIN_ARGS = {
    'train': True,

    'down_sampling': False,

    'data_type': 'train',

    'images_dir_path': os.path.join(Xray_COMMON_ARGS['dir_path'], 'train_balanced'),

    'images_dir_idx_ptr_path': os.path.join(Xray_COMMON_ARGS['dir_path'], 'train_images_dir_idx_ptr.txt'),

    'labels_csv_file_path': os.path.join(Xray_COMMON_ARGS['dir_path'], 'labels.csv'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_train_data_labels_coll_name'],
}

Xray_TEST_ARGS = {
    'train': False,

    'down_sampling': False,

    'data_type': 'test',

    'images_dir_path': os.path.join(Xray_COMMON_ARGS['dir_path'], 'test_balanced'),

    'images_dir_idx_ptr_path': os.path.join(Xray_COMMON_ARGS['dir_path'], 'test_images_dir_idx_ptr.txt'),

    'labels_csv_file_path': os.path.join(Xray_COMMON_ARGS['dir_path'], 'labels.csv'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_test_data_labels_coll_name'],
}

CatDog_TRAIN_ARGS = {
    'train': True,

    'data_type': 'train',

    'images_dir_path': os.path.join(CatDog_COMMON_ARGS['dir_path'], 'train'),

    'images_dir_idx_ptr_path': os.path.join(CatDog_COMMON_ARGS['dir_path'], 'train_images_dir_idx_ptr.txt'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_train_data_labels_coll_name'],
}

CatDog_TEST_ARGS = {
    'train': False,

    'data_type': 'test',

    'images_dir_path': os.path.join(CatDog_COMMON_ARGS['dir_path'], 'test'),

    'images_dir_idx_ptr_path': os.path.join(CatDog_COMMON_ARGS['dir_path'], 'test_images_dir_idx_ptr.txt'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_test_data_labels_coll_name'],
}

# --------------------------------
#  merge dataSET COMMON and PRIVATE arguments
# --------------------------------

MNIST_TRAIN_ARGS.update(MNIST_COMMON_ARGS)
MNIST_TEST_ARGS.update(MNIST_COMMON_ARGS)

DRD_TRAIN_ARGS.update(DRD_COMMON_ARGS)
DRD_TEST_ARGS.update(DRD_COMMON_ARGS)
DRD_TESTING_ARGS.update(DRD_COMMON_ARGS)

ECG_TRAIN_ARGS.update(ECG_COMMON_ARGS)
ECG_TEST_ARGS.update(ECG_COMMON_ARGS)

Xray_TRAIN_ARGS.update(Xray_COMMON_ARGS)
Xray_TEST_ARGS.update(Xray_COMMON_ARGS)

CatDog_TRAIN_ARGS.update(CatDog_COMMON_ARGS)
CatDog_TEST_ARGS.update(CatDog_COMMON_ARGS)



