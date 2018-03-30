
import os
import sys
import random
import warnings
import pickle

import numpy as np
import pandas as pd

from scipy import stats

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage import exposure as skexposure



DATA_ROOT = '/data/contest_data'
TRAIN_PATH = DATA_ROOT + '/stage1_train/'
MODEL_PATH = '/data/models'


def flattened_trainset(width, height, channels):

    # Get train IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), height , width, channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), height, width, 1), dtype=np.bool)
    train_shapes = []

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:channels]
        train_shapes.append(img.shape)
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((height, width, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (height, width), mode='constant', 
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    return X_train, Y_train

