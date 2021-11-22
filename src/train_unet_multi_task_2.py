import argparse

import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.models

from tools import sources
from tools import unetmodel
from tools import unet_ext_masks


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--samples", type=int)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--mask-with-weights", action='store_true')

args = parser.parse_args()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

tx, ty, tc, tdx1, tdy1, tdx2, tdy2, trm, tmw, _ = sources.flattened_trainset_ex_2(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, args.samples)
model = unetmodel.u_net_model_ext(
    IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, centers_weight=10.0, mask_with_weights=args.mask_with_weights)

if args.mask_with_weights:
    weighted_mask = np.concatenate([ty, tmw], axis=-1)
else:
    weighted_mask = ty

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(args.output, verbose=1)#, save_best_only=True)
results = model.fit(tx,
                    [weighted_mask, tc, tdx1, tdy1, tdx2, tdy2],
                    validation_split=0.0,
                    batch_size=16,
                    epochs=args.epochs,
                    callbacks=[earlystopper, checkpointer])


final_model = keras.models.load_model(args.output, compile=False)
pty, ptc, ptdx1, ptdy1, ptdx2, ptdy2 = final_model.predict(tx)
iou = unet_ext_masks.batch_precision_iou(trm, pty, ptc, ptdx1, ptdx2, ptdy1, ptdy2)
print(f'Final IOU precision: {iou}')