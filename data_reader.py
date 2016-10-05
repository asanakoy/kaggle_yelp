import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
import time

DATA_ROOT = '/home/artem/workspace/kaggle/yelp/data'


def get_pathes(data_dir):
    pathes = {'train': dict(), 'test': dict()}
    split_name = {'test': 'test', 'train': 'training'}
    keys = ['business', 'checkin', 'review', 'user']
    for split in pathes.keys():
        for key in keys:

            pathes[split][key] = os.path.join(data_dir, 'yelp_{}_set'.format(split_name[split]),
                                              'yelp_{}_set_{}.json'.format(split_name[split], key))
    return pathes


def read_train(data_dir):
    pathes = get_pathes(data_dir)
    objects = []
    with open(pathes['train']['review']) as data_file:
        lines = data_file.readlines()
        for line in lines:
            objects.append(json.loads(line))

    print len(objects)
    data = {k: list() for k in objects[0].keys()}
    for obj in objects:
        for k in data.keys():
            data[k].append(obj[k])

    df = pd.DataFrame(data=data)
    df['date'] = pd.to_datetime(df['date'])
    assert np.all(pd.notnull(df['date']))
    # filter review without text
    text_len = df.text.apply(len)
    df = df[text_len > 0]

    train_votes = df.votes.apply(lambda x: x['useful'])
    return df, train_votes


def read_test(data_dir):
    pathes = get_pathes(data_dir)
    objects = []
    with open(pathes['test']['review']) as data_file:
        lines = data_file.readlines()
        for line in lines:
            objects.append(json.loads(line))

    print len(objects)
    data = {k: list() for k in objects[0].keys()}
    for obj in objects:
        for k in data.keys():
            data[k].append(obj[k])

    df = pd.DataFrame(index=data['review_id'], data=data)
    df['date'] = pd.to_datetime(df['date'])
    assert np.all(pd.notnull(df['date']))
    return df
