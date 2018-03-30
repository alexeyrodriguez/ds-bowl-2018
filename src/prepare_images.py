# Script to generate initial set of images with which to work 

from tools import sources

X_train, Y_train = sources.flattened_trainset(128, 128, 3)
