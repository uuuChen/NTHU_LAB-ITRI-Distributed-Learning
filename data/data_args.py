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

    'class_id':  {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9
    },
}

ECG_COMMON_ARGS = {
    'data_name': 'ECG',

    'dir_path': 'data/ECG/',

    'label_class_nums': 5,

    'data_length': 187,

    'MLP_conn_node_nums': 500,

    'use_gridFS': False,

    'down_sampling': False,

    'class_id':  {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
    },
}

OCT_COMMON_ARGS = {
    'data_name': 'OCT',

    'dir_path': 'data/OCT/',

    'label_class_nums': 5,

    'use_gridFS': True,

    'class_id':  {
            'NORMAL': 0,
            'CNV': 1,
            'DME': 2,
            'DRUSEN': 3
    },
}

MD_COMMON_ARGS = {
    'data_name': 'MD',

    'dir_path': 'data/MD/',

    'label_class_nums': 2,

    'use_gridFS': True,

    'class_id': {
        'Parasitized': 0,
        'Uninfected': 1,
    },
}

MNIST_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

ECG_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

OCT_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

MD_COMMON_ARGS.update(GLOBAL_COMMON_ARGS)

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


OCT_TRAIN_ARGS = {
    'train': True,

    'down_sampling': False,

    'data_type': 'train',

    'images_dir_path': os.path.join(OCT_COMMON_ARGS['dir_path'], 'sample_1'),

    'images_dir_idx_ptr_path': os.path.join(OCT_COMMON_ARGS['dir_path'], 'train_images_dir_idx_ptr.txt'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_train_data_labels_coll_name'],
}

OCT_TEST_ARGS = {
    'train': False,

    'down_sampling': False,

    'data_type': 'test',

    'images_dir_path': os.path.join(OCT_COMMON_ARGS['dir_path'], 'sample_test1'),

    'images_dir_idx_ptr_path': os.path.join(OCT_COMMON_ARGS['dir_path'], 'test_images_dir_idx_ptr.txt'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_test_data_labels_coll_name'],
}

MD_TRAIN_ARGS = {
    'train': True,

    'down_sampling': False,

    'data_type': 'train',

    'images_dir_path': os.path.join(MD_COMMON_ARGS['dir_path'], 'train'),

    'images_dir_idx_ptr_path': os.path.join(MD_COMMON_ARGS['dir_path'], 'train_images_dir_idx_ptr.txt'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_train_data_labels_coll_name'],
}

MD_TEST_ARGS = {
    'train': False,

    'down_sampling': False,

    'data_type': 'test',

    'images_dir_path': os.path.join(MD_COMMON_ARGS['dir_path'], 'test'),

    'images_dir_idx_ptr_path': os.path.join(MD_COMMON_ARGS['dir_path'], 'test_images_dir_idx_ptr.txt'),

    'db_data_labels_coll_name': GLOBAL_COMMON_ARGS['db_test_data_labels_coll_name'],
}
# --------------------------------
#  merge dataSET COMMON and PRIVATE arguments
# --------------------------------

MNIST_TRAIN_ARGS.update(MNIST_COMMON_ARGS)
MNIST_TEST_ARGS.update(MNIST_COMMON_ARGS)

ECG_TRAIN_ARGS.update(ECG_COMMON_ARGS)
ECG_TEST_ARGS.update(ECG_COMMON_ARGS)

OCT_TRAIN_ARGS.update(OCT_COMMON_ARGS)
OCT_TEST_ARGS.update(OCT_COMMON_ARGS)

MD_TRAIN_ARGS.update(MD_COMMON_ARGS)
MD_TEST_ARGS.update(MD_COMMON_ARGS)


