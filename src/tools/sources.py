
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



DATA_ROOT = '/home/jupyter/dsb-2018'
TRAIN_PATH = DATA_ROOT + '/stage1_train/'
TEST_PATH = DATA_ROOT + '/stage1_test/'
MODEL_PATH = '/home/jupyter/dsb-2018/models'

def flattened_trainset(width, height, channels, samples=None):

    # Get train IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    if samples is not None:
        train_ids = train_ids[:samples]

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
            mask_ = resize(mask_, (height, width), mode='constant', preserve_range=True) > 0
            mask_ = np.expand_dims(mask_, axis=-1)
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
    return int(start_x + width / 2.0 - 1), int(start_y + height / 2.0 - 1)

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
        # 0.01 for mask hack

        diffx = diffx + (mask > 0).astype('float') * (np.outer(np.ones(H), np.arange(W)) - x + 0.01)
        diffy = diffy + (mask > 0).astype('float') * (np.outer(np.arange(H), np.ones(W)) - y + 0.01)
    return centers, widths, heights, diffx, diffy


def mask_centers_sizes_2(masks):
    '''
    Function to extract centers and bounding boxes from masks.
    The bounding boxes specify the delta to sides from the center.
    E.g. Bounding box is specified by:
     * Left top corner (center_x - diffx1, center_y - diffy1)
     * Right bottom corner: (center_x + diffx2, center_y + diffy2)
     
    It also extracts overlaps but these are not used
    '''
    shape = masks[0].shape
    H, W = shape
    overlap = np.zeros(shape=shape).astype('int')
    centers = np.zeros(shape=shape).astype('int')
    diffx1 = np.zeros(shape=shape)
    diffy1 = np.zeros(shape=shape)
    diffx2 = np.zeros(shape=shape)
    diffy2 = np.zeros(shape=shape)

    for mask in masks:
        (x1, x2), (y1, y2) = inclusive_bounds(mask)
        bounds = inclusive_bounds(mask)
        x, y = bounds_center(bounds)
        centers[y, x] = 1

        # Hack to easily be able to extract a mask for training time
        # 0.01 for mask hack
        diffx1 = diffx1 + (mask > 0).astype('float') * (np.outer(np.ones(H), np.arange(W)) - x1 + 0.01)
        diffy1 = diffy1 + (mask > 0).astype('float') * (np.outer(np.arange(H), np.ones(W)) - y1 + 0.01)
        diffx2 = diffx2 + (mask > 0).astype('float') * (np.outer(np.ones(H), np.arange(W)) - x2 + 0.01)
        diffy2 = diffy2 + (mask > 0).astype('float') * (np.outer(np.arange(H), np.ones(W)) - y2 + 0.01)
        overlap = overlap + (mask > 0).astype('float')
        
    overlap = overlap > 1.0
    diffx1 = diffx1 * (overlap < 0.5).astype('float')
    diffy1 = diffy1 * (overlap < 0.5).astype('float')
    diffx2 = diffx2 * (overlap < 0.5).astype('float')
    diffy2 = diffy2 * (overlap < 0.5).astype('float')


    return centers, overlap, diffx1, diffy1, diffx2, diffy2


def flattened_trainset_ex(width, height, channels, samples=None):

    # Get train IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    if samples is not None:
        train_ids = train_ids[:samples]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), height , width, channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), height, width, 1), dtype=np.bool)

    centers = np.zeros((len(train_ids), height, width, 1), dtype=np.int)
    widths = np.zeros((len(train_ids), height, width, 1))
    heights = np.zeros((len(train_ids), height, width, 1))
    diffx = np.zeros((len(train_ids), height, width, 1))
    diffy = np.zeros((len(train_ids), height, width, 1))

    train_meta = []
    resized_masks = []

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:channels]
        train_meta.append((id_, img.shape[1], img.shape[0]))
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((height, width, 1), dtype=np.bool)
        masks = []

        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = resize(mask_, (height, width), mode='constant', preserve_range=True) > 0
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)
            masks.append(np.squeeze(mask_))

        Y_train[n] = mask

        mcenters, mwidths, mheights, mdiffx, mdiffy = mask_centers_sizes(masks)
        centers[n] = np.expand_dims(mcenters, axis=-1)
        widths[n] = np.expand_dims(mwidths, axis=-1)
        heights[n] = np.expand_dims(mheights, axis=-1)
        diffx[n] = np.expand_dims(mdiffx, axis=-1)
        diffy[n] = np.expand_dims(mdiffy, axis=-1)
        resized_masks.append(masks)

    return X_train, Y_train, centers, widths, heights, diffx, diffy, resized_masks, train_meta

def flattened_trainset_ex_2(width, height, channels, samples=None):
    '''
    Extract data from data set: input, merged resized masks, resized masks
    Also generate derived data: centers for each mask, deltas to bounding boxes, overlaps
    The function and dependent functions are monstruous but emphasis was on quick iteration
    '''

    # Get train IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    if samples is not None:
        train_ids = train_ids[:samples]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), height , width, channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), height, width, 1), dtype=np.bool)

    centers = np.zeros((len(train_ids), height, width, 1), dtype=np.int)
    overlap = np.zeros((len(train_ids), height, width, 1), dtype=np.int)
    diffx1 = np.zeros((len(train_ids), height, width, 1))
    diffy1 = np.zeros((len(train_ids), height, width, 1))
    diffx2 = np.zeros((len(train_ids), height, width, 1))
    diffy2 = np.zeros((len(train_ids), height, width, 1))


    train_meta = []
    resized_masks = []

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:channels]
        train_meta.append((id_, img.shape[1], img.shape[0]))
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((height, width, 1), dtype=np.bool)
        masks = []

        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = resize(mask_, (height, width), mode='constant', preserve_range=True) > 0
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)
            masks.append(np.squeeze(mask_))

        Y_train[n] = mask

        mcenters, moverlaps, mdiffx1, mdiffy1, mdiffx2, mdiffy2 = mask_centers_sizes_2(masks)
        centers[n] = np.expand_dims(mcenters, axis=-1)
        overlap[n] = np.expand_dims(moverlaps, axis=-1)
        diffx1[n] = np.expand_dims(mdiffx1, axis=-1)
        diffy1[n] = np.expand_dims(mdiffy1, axis=-1)
        diffx2[n] = np.expand_dims(mdiffx2, axis=-1)
        diffy2[n] = np.expand_dims(mdiffy2, axis=-1)

        resized_masks.append(masks)

    return X_train, Y_train, centers, diffx1, diffy1, diffx2, diffy2, resized_masks, train_meta



def flattened_testset_ex(width, height, channels):

    # Get train IDs
    test_ids = next(os.walk(TEST_PATH))[1]

    # Get and resize train images and masks
    X_test = np.zeros((len(test_ids), height , width, channels), dtype=np.uint8)

    test_meta = []

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:channels]
        test_meta.append((id_, img.shape[1], img.shape[0]))
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        X_test[n] = img

    return X_test, test_meta

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

