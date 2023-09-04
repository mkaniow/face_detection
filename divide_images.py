'''
This file divides every image and json files in folder to 3 parts (train, test, validation)

Folder structure:
/data
    /images (all resized images)
    /labels (all json data with coords of face [if exists])
    /train
        /images
        /labels
    /test
        /images
        /labels
    /validate
        /images
        /labels

TRAIN_PRC - it's percentage of images that will be assigned to train data set (value between 0 - 1)
TEST_PRC - it's percentage of images that will be assigned to test data set (value between 0 - 1)
TRAIN_PRC + TEST_PRC <= 1

rest of images will be assigned to validation data set
'''

import os
import random

TRAIN_PRC = 0.6

TEST_PRC = 0.2

for item in os.listdir('data/images'):
    number = random.random()
    # assign to train
    if number < TRAIN_PRC:
        old = os.path.join('data', 'images', item)
        new = os.path.join('data', 'train', 'images', item)
        os.replace(old, new)
        try:
            splited = os.path.splitext(item)
            old_json_file = os.path.join('data', 'labels', splited[0] + '.json')
            new_json_file = os.path.join('data', 'train', 'labels', splited[0] + '.json')
            os.replace(old_json_file, new_json_file)
        except:
            pass
    # assign to test
    elif number >= TRAIN_PRC and number < TRAIN_PRC + TEST_PRC:
        old = os.path.join('data', 'images', item)
        new = os.path.join('data', 'test', 'images', item)
        os.replace(old, new)
        try:
            splited = os.path.splitext(item)
            old_json_file = os.path.join('data', 'labels', splited[0] + '.json')
            new_json_file = os.path.join('data', 'test', 'labels', splited[0] + '.json')
            os.replace(old_json_file, new_json_file)
        except:
            pass
    # assign to validation
    else:
        old = os.path.join('data', 'images', item)
        new = os.path.join('data', 'validate', 'images', item)
        os.replace(old, new)
        try:
            splited = os.path.splitext(item)
            old_json_file = os.path.join('data', 'labels', splited[0] + '.json')
            new_json_file = os.path.join('data', 'validate', 'labels', splited[0] + '.json')
            os.replace(old_json_file, new_json_file)
        except:
            pass