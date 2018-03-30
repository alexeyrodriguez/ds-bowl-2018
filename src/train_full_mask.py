
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tools import sources
from tools import unetmodel


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

X_train, Y_train = sources.flattened_trainset(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
model = unetmodel.u_net_model(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(sources.MODEL_PATH + '/model-dsbowl2018-unet-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=2,
                    callbacks=[earlystopper, checkpointer])
