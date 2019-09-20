
# System Imports
import pymongo
import re
import numpy as np
import os
import csv
import io
from gridfs import *
from PIL import Image

# Logger Imports
from logger import *


DEBUG = False  # when Debug is True, logger will print debug messages.


class MongoDB_Processor(Logger):

    def __init__(self, data_args):

        Logger.__init__(self)

        self.db_name = data_args['data_name']  # the name of the database to link to, and it equals to the data name

        self.client = pymongo.MongoClient(data_args['client_url'])  # connect to client

        self.db = self.client[self.db_name]  # connect to database

        # get a unique logger for debug, so the get_logger() needs to pass in a unique name
        unique_name = data_args['data_type'] + '_' + self.db_name + '_' + __name__
        self.__logger = self.get_logger(unique_name=unique_name, debug=DEBUG)

    def coll_insert(self, coll_name, data, labels):

        """Insert data and labels into a collection without using gridFS.

        Data and labels must be type (list).

        Args:
            coll_name (str) :  The name of the collection to insert data and labels.
            data      (list):  The data to be inserted into the database.
            labels    (list):  The labels corresponding to the data.

        Returns:
            None.

        """

        coll = self.db[coll_name]

        data_labels_dicts = []

        self.__logger.debug('Insert %s Rows To Database ( %s ) Collection ( %s )...' % (len(data), self.db_name,
                                                                                        coll_name))

        for i in range(len(data)):

            if (i + 1) % 1000 == 0:
                self.__logger.debug('Insert %s Rows' % (i + 1))

            data_label_dict = {
                'ID': i + 1,
                'data': data[i],
                'label': labels[i],
            }

            data_labels_dicts.append(data_label_dict)

        coll.insert_many(data_labels_dicts)

        self.__logger.debug('Done')

    def coll_find_all(self, coll_name):

        """Return the cursor of all the rows in the collection.

        It can't get the cursor of the rows which are inserted into the collection using gridFS.

        Args:
            coll_name (str):  The name of the collection to get the cursor of the rows.

        Returns:
            cursor (cursor): The cursor of the rows of the collection.

        """

        coll = self.db[coll_name]

        cursor = coll.find(no_cursor_timeout=True)

        return cursor

    def coll_find_query(self, coll_name, query):

        """Return the cursor of the rows matching query in the collection.

        It can't get the cursor of the rows which are inserted into the collection using gridFS.

        Args:
            coll_name (str) : The name of the collection to get the cursor of the rows.
            query     (dict): The condition of the rows to be get.

        Returns:
           cursor (cursor): The cursor of the rows matching query of the collection.

        """

        coll = self.db[coll_name]

        cursor = coll.find(query, no_cursor_timeout=True)

        return cursor

    def coll_delete_all(self, coll_name):

        """Delete all the rows in the collection.

        It can't delete the rows which are inserted into the collection using gridFS.

        Args:
            coll_name (str):  The name of the collection to delete all the rows.

        Returns:
            None.

        """

        coll = self.db[coll_name]

        delete_count = coll.count()

        self.__logger.debug('Delete %s Rows From Database ( %s ) ...' % (delete_count, self.db_name))

        coll.delete_many({})

        self.__logger.debug('Done !')

    def coll_read_all_labels(self, coll_name):
        labels = []
        cursor = self.coll_find_all(coll_name=coll_name)
        for i, cursor_ in list(enumerate(cursor, start=1)):
            self.__logger.debug('Read %s Labels' % (int(i) + 1))
            label = cursor_['label']
            labels.append(label)
        self.__logger.debug('Done !')
        return labels

    def coll_read_all(self, coll_name):

        """Return all the data and labels in the collection.

        It can't get the data and labels which are inserted into the collection using gridFS.

        Args:
            coll_name (str):  The name of the collection to get all the data and labels.

        Returns:
            data   (np.ndarray): The data stored in the collection.
            labels (np.ndarray): The labels corresponding to the data stored in the collection.
        """

        data = []
        labels = []

        data_labels = self.coll_find_all(coll_name=coll_name)

        self.__logger.debug('Get %s Rows From Database ( %s ) ...' % (data_labels.count(), self.db_name))

        for i, data_label in list(enumerate(data_labels, start=1)):

            if i % 1000 == 0:
                self.__logger.debug('Read %s Rows' % i)

            data_ = data_label['data']
            label_ = data_label['label']

            data.append(data_)
            labels.append(label_)

        self.__logger.debug('Done !')

        return np.array(data), np.array(labels)

    def coll_read_batch(self, coll_name, id_list):

        """Return a batch of the data and labels in the collection.

        It can't get the data and labels which are inserted into the collection using gridFS.

        Args:
            coll_name (str) :  The name of the collection to get batch of the data and labels.
            id_list   (list):  It indicates the data id list of the data to be read from the database.

        Returns:
            data   (np.ndarray): A batch of data stored in the collection.
            labels (np.ndarray): A batch of labels corresponding to the data stored in the collection.
        """

        batch_data = []
        batch_labels = []

        find_query = {'ID': {"$in": id_list}}
        batch_data_labels = self.coll_find_query(coll_name=coll_name,
                                                 query=find_query)

        self.__logger.debug('id list %s ' % str(id_list))

        self.__logger.debug('Get %s Rows From Database ( %s ) ...' % (batch_data_labels.count(), self.db_name))

        for i, data_label in list(enumerate(batch_data_labels, start=1)):

            if i % 1000 == 0:
                self.__logger.debug('Read %s Rows' % i)

            data = data_label['data']
            label = data_label['label']

            batch_data.append(data)
            batch_labels.append(label)

        self.__logger.debug('Done !')

        return np.array(batch_data), np.array(batch_labels)

    def drop_database(self):

        """Delete the database where the current dataSet is located.

        Args:
            None.

        Returns:
            None.

        """

        self.__logger.debug('Delete Database ( %s ) ...' % self.db_name)

        self.client.drop_database(self.db_name)

        self.__logger.debug('Done !')

    def gridFS_coll_insert(self, coll_name, data_file_paths, labels):

        """Insert data and labels into a collection using gridFS.

        The passed argument must be the the data file paths instead of the data, because gridFS can only upload data to
        the database in the format of file or str. If some data are incomplete, we won't upload them to database.

        In this function, data is opened and uploaded as a file.

        Args:
            data      (list):  The data to be inserted into the database.
            labels    (list):  The labels corresponding to the data.
            coll_name (str) :  The name of the collection to insert data and labels.

        Returns:
            None.

        """

        self.__logger.debug('Insert %s Rows From Local To Database ( %s ) Collection ( %s ) ...' %
                            (len(data_file_paths), self.db_name, coll_name))

        error_data_nums = 0

        fs = GridFS(self.db, collection=coll_name)

        for i in range(len(data_file_paths)):

            if (i + 1) % 100 == 0:
                self.__logger.debug('Insert %s Rows' % int(i + 1))

            data_file_path = data_file_paths[i]

            # # check whether the data is complete
            # byte_data = open(data_file_path, 'rb').read()
            # data = np.array(Image.open(io.BytesIO(byte_data)))
            # if len(data.shape) == 0:  # the data is incomplete, so we have to ignore it
            #     self.__logger.error('%s is incomplete' % data_file_path)
            #     error_data_nums += 1
            #     continue

            # else:  # the data is complete, so we upload data file and corresponding label to database
            label = labels[i]

            dic = {
                "label": label,
                "file_name": re.split(r"[/\\]", data_file_path)[-1],
                "ID": i + 1
            }

            fs.put(open(data_file_path, 'rb'), **dic)

        self.__logger.debug('Done !')

        return error_data_nums

    def gridFS_coll_download_all(self, coll_name, download_dir_path, category_name):

        """Download all the data and labels from a collection to directory using gridFS.

        All data will be stored in directory path: "download_dir_path/category_name" as JPEG files, and all labels
        corresponding to data will be stored in directory path: "download_dir_path/" as "trainLabels.csv" file.

        Args:
            coll_name         (str):  The name of the collection to get data and labels.
            download_dir_path (str):  The path of the directory which store "trainLabels.csv" file.
            category_name     (str):  The name of the directory which store all the downloaded images data.

        Returns:
           None.

        """

        fs = GridFS(self.db, coll_name)

        grid_outs = fs.find(no_cursor_timeout=True)
        grid_outs_count = grid_outs.count()

        self.__logger.debug('Download %s Rows From Database ( %s ) ...' % (grid_outs_count, self.db_name))

        with open(os.path.join(download_dir_path, 'trainLabels.csv'), 'w', newline='') as csvfile:

            for grid_out in grid_outs:

                data = grid_out.read()
                label = grid_out.label
                file_name = grid_out.file_name.split('.')[0]

                self.__logger.debug('\t' + str(os.path.join(download_dir_path, category_name, file_name)))

                out_file = open(os.path.join(download_dir_path, category_name, file_name), 'wb')
                out_file.write(data)

                writer = csv.writer(csvfile)
                writer.writerow([file_name, label])

                out_file.close()

        self.__logger.debug('Done !')

    def gridFS_coll_read_all_labels(self, coll_name):
        all_grid_outs = self.gridFS_coll_find_all(coll_name)
        labels = []
        for i, grid_out in list(enumerate(all_grid_outs)):
            self.__logger.debug('Read %s Labels' % (int(i) + 1))
            label = grid_out.label
            labels.append(label)
        self.__logger.debug('Done !')
        return labels

    def gridFS_coll_read_batch(self, coll_name, id_list):

        """Return a batch size of the data and labels in the collection using gridFS.

        It can't get the data and labels which are inserted into the collection without using gridFS.

        Args:
            coll_name  (str) :  The name of the collection to get a batch size of the data and labels.
            id_list    (list):  It indicates the data id list of the data to be read from the database.

        Returns:
            data   (np.ndarray): The data stored in the collection.
            labels (np.ndarray): The labels corresponding to the data stored in the collection.

        """

        fs = GridFS(self.db, coll_name)

        find_query = {'ID': {"$in": id_list}}

        batch_grid_outs = fs.find(find_query)

        batch_grid_outs_count = batch_grid_outs.count()

        self.__logger.debug('Read %s Rows From Database ( %s ) ...' % (batch_grid_outs_count, self.db_name))

        batch_images = []
        batch_labels = []

        for i, grid_out in list(enumerate(batch_grid_outs)):

            self.__logger.debug('Read %s Images' % (int(i) + 1))

            byte_data = grid_out.read()

            image = np.array(Image.open(io.BytesIO(byte_data)))
            label = grid_out.label

            batch_images.append(image)
            label = np.array(label)
            batch_labels.append(label)

        self.__logger.debug('Done !')

        return np.array(batch_images), np.array(batch_labels)

    def gridFS_coll_delete_all(self, coll_name):

        """Delete all the rows in the collection using gridFS.

        It can't delete the rows which are inserted into the collection without using gridFS.

        Args:
           coll_name (str):  The name of the collection to delete all the rows.

        Returns:
           None.

        """

        fs = GridFS(self.db, coll_name)

        delete_count = fs.find(no_cursor_timeout=True).count()

        self.__logger.debug('Delete %s Rows From Database ( %s ) ...' % (delete_count, self.db_name))

        for grid_out in fs.find(no_cursor_timeout=True):

            id = grid_out._id

            fs.delete(id)

        self.__logger.debug('Done !')

    def gridFS_coll_find_all(self, coll_name):

        """Return the cursor of the rows in the collection using gridFS.

        It can't get the cursor of the rows which are inserted into the collection without using gridFS.

        Args:
            coll_name (str):  The name of the collection to get the cursor of the rows.

        Returns:
            cursor (cursor): The cursor of the rows of the collection.

        """

        fs = GridFS(self.db, coll_name)

        cursor = fs.find(no_cursor_timeout=True)

        return cursor





















