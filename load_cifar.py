#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:08:12 2017

@author: user
"""
import pickle
import numpy as np

def transformer(data, label):
    data = data.reshape((-1, 32, 32))
    data = data.astype(np.float32)
    mean = data.mean(axis=(1,2))
    std = data.std(axis=(1,2))
    for d,m in zip(data[:],mean[:]):
        d-=m
    for d,s in zip(data[:],std[:]):
        d/=s
    for d in data[:]:
        d+=0.5
    data = data.reshape((-1, 3, 32, 32))
    return data, label
def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d[b'data'], d[b'labels']

def load():
    train_file = ['data/data_batch_1','data/data_batch_2','data/data_batch_3',
                  'data/data_batch_4','data/data_batch_5']

    traindata, trainlabel = unpickle(train_file[0])
    for f in train_file[1:]:
        data, label = unpickle(f)
        traindata = np.concatenate((traindata, data),0)
        trainlabel = np.concatenate((trainlabel, label),0)
    testdata, testlabel = unpickle('data/test_batch')
    return transformer(traindata, trainlabel), transformer(testdata, testlabel)