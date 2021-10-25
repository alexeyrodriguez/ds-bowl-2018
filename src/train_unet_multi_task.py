import argparse

from keras.callbacks import EarlyStopping, ModelCheckpoint

from tools import sources
from tools import unetmodel

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--samples", type=int)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

tx, ty, tc, tw, th, tdx, tdy, _, _ = sources.flattened_trainset_ex(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, args.samples)
model = unetmodel.u_net_model_ext(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(args.output, verbose=1)#, save_best_only=True)
results = model.fit(tx,
                    [ty, tc, tw, th, tdx, tdy],
                    validation_split=0.0,
                    batch_size=16,
                    epochs=args.epochs,
                    callbacks=[earlystopper, checkpointer])

