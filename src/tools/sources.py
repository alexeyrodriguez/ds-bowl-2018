
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


def interval_array(xs):
    return xs.argmax(), len(xs) - 1 - np.flip(xs, axis=0).argmax()

def inclusive_bounds(mask):
    x_mask = np.max(mask, axis=0)
    y_mask = np.max(mask, axis=1)
    return interval_array(x_mask), interval_array(y_mask)

def bounds_size(bounds):
    (start_x, end_x), (start_y, end_y) = bounds
    return end_x - start_x + 1, end_y - start_y + 1

def bounds_center(bounds):
    (start_x, _), (start_y, _) = bounds
    width, height = bounds_size(bounds)
    return int(start_x + width / 2 - 1), int(start_y + height / 2 - 1)

def mask_centers_sizes(masks):
    shape = masks[0].shape
    H, W = shape
    centers = np.zeros(shape=shape).astype('int')
    widths = np.zeros(shape=shape)
    heights = np.zeros(shape=shape)
    diffx = np.zeros(shape=shape)
    diffy = np.zeros(shape=shape)
    for mask in masks:
        bounds = inclusive_bounds(mask)
        x, y = bounds_center(bounds)
        w, h = bounds_size(bounds)
        centers[y, x] = 1
        widths[y, x] = w
        heights[y, x] = h
        # Hack to easily be able to extract a mask for training time
        diffx = diffx + (mask > 0).astype('float') * (np.outer(np.ones(H), np.arange(W)) - x + 0.01)
        diffy = diffy + (mask > 0).astype('float') * (np.outer(np.arange(H), np.ones(W)) - y + 0.01)
    return centers, widths, heights, diffx, diffy

def flattened_trainset_ex(width, height, channels):

    # Get train IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), height , width, channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), height, width, 1), dtype=np.bool)

    centers = np.zeros((len(train_ids), height, width, 1), dtype=np.int)
    widths = np.zeros((len(train_ids), height, width, 1))
    heights = np.zeros((len(train_ids), height, width, 1))
    diffx = np.zeros((len(train_ids), height, width, 1))
    diffy = np.zeros((len(train_ids), height, width, 1))

    train_shapes = []

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:channels]
        train_shapes.append(img.shape)
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((height, width, 1), dtype=np.bool)
        masks = []

        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (height, width), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
            masks.append(np.squeeze(mask_))

        Y_train[n] = mask

        mcenters, mwidths, mheights, mdiffx, mdiffy = mask_centers_sizes(masks)
        centers[n] = np.expand_dims(mcenters, axis=-1)
        widths[n] = np.expand_dims(mwidths, axis=-1)
        heights[n] = np.expand_dims(mheights, axis=-1)
        diffx[n] = np.expand_dims(mdiffx, axis=-1)
        diffy[n] = np.expand_dims(mdiffy, axis=-1)

    return X_train, Y_train, centers, widths, heights, diffx, diffy


def raw_trainset(ix):

    # Get train IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    id_ = list(train_ids)[ix]

    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')# [:,:,:channels]

    mask = []
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask.append(imread(path + '/masks/' + mask_file))

    return img, mask

