
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tools import sources
from tools import unetmodel


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

tx, ty, tc, tw, th, tdx, tdy = sources.flattened_trainset_ex(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
model = unetmodel.u_net_model_ext(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(sources.MODEL_PATH + '/model-dsbowl2018-unet-multi-task-1.h5', verbose=1, save_best_only=True)
results = model.fit(tx,
                    [ty, tc, tw, th, tdx, tdy],
                    validation_split=0.1,
                    batch_size=16,
                    epochs=20,
                    callbacks=[earlystopper, checkpointer])

