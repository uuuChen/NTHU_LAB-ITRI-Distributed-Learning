import pymongo
from gridfs import *
"""
<< WARNING >>
you will delete the following database on mongodb:
Xray, DRD, ECG, MNIST
"""

if __name__ == '__main__':
    client = pymongo.MongoClient('mongodb://localhost:27017/')

    client.drop_database('OCT')
    # client.drop_database('DRD')
    # client.drop_database('ECG')
    # client.drop_database('MNIST')