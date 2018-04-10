import numpy as np
from skimage import morphology
from skimage import transform

from . import rle

def masks(predicted_mask, thr=0.5):
    mmasks = morphology.label(np.squeeze(predicted_mask) > 0.5)
    imasks = []
    for c in range(len(np.unique(mmasks)) - 1):
        imasks.append((mmasks==(c+1)).astype(float))
    return imasks

def submission(file_name, predicted_masks, meta, n):
    with open(file_name, 'w') as sub:
        sub.write('ImageId,EncodedPixels\n')
        for ix in range(n):
            t_masks = masks(predicted_masks[ix])
            t_masks = [transform.resize(t_mask, (meta[ix][2], meta[ix][1])) for t_mask in t_masks]
            for t_mask in t_masks:
                enc_mask = list(rle.encode(t_mask > 0.5))
                enc_mask = [str(y) for x in enc_mask for y in x]
                enc_mask = ' '.join(enc_mask)
                sub.write('{},{}\n'.format(meta[ix][0], enc_mask))
