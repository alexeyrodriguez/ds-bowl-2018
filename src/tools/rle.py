import numpy as np

def encode(mask):
    v = mask.T.flatten()

    preceding = np.zeros(v.size)
    preceding[1:] = v[:-1]
    starts = np.argwhere((preceding==0) & (v==1)).flatten()

    following = np.zeros(v.size)
    following[:-1] = v[1:]
    stops = np.argwhere((following==0) & (v==1)).flatten()

    lengths = stops-starts+1
    starts = starts+1

    return zip(starts, lengths)
