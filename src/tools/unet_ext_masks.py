import numpy as np
from . import rect


def extract_rectangles(mask, widths, heights, diffx, diffy):
    diffx = np.squeeze(diffx)
    diffy = np.squeeze(diffy)
    widths = np.squeeze(widths)
    heights = np.squeeze(heights)

    rows, cols = np.where((np.abs(diffx) < 1) & (np.abs(diffy) < 1))

    candidate_rectangles = [(x, y, widths[y, x], heights[y, x])
                            for (y, x) in zip(rows, cols)
                            if widths[y, x] > 2 and heights[y, x] > 2]

    coord_rectangles = [(max(0, x - w/2), max(0, y - h/2), x + w/2, y + h/2)
                        for (x, y, w, h) in candidate_rectangles]

    filtered_rectangles = [(x1, y1, x2, y2)
                           for (x1, y1, x2, y2) in coord_rectangles
                           if mask[int(y1):int(y2), int(x1):int(x2)].mean() >= 0.5]

    merged_rectangles = rect.merge(filtered_rectangles)

    return merged_rectangles

def get_rects_from_centers(centers, diffx1, diffx2, diffy1, diffy2, thr=0.0):
    def r(x):
        return int(np.round(x))
    rects = []
    for x in range(128):
        for y in range(128):
            if centers[y, x] > thr:
                rects.append((r(x-diffx1[y,x]), r(y-diffy1[y,x]), r(x-diffx2[y,x]), r(y-diffy2[y,x])))
    return rects

def extract_masks(mask, rects, thr=0.5):
    mask = mask.copy().squeeze() >= thr
    shp = mask.shape[:2]
    masks = []
    for x1, y1, x2, y2 in rects:
        aux_mask = np.zeros(shp)
        aux_mask[int(y1):int(y2)+1, int(x1):int(x2)] = 1
        masks.append(mask * aux_mask)
    return masks

def iou(m1, m2):
    intersection = (m1*m2)
    union = float(m1.flatten().sum() + m2.flatten().sum() - intersection.flatten().sum())
    res = float(intersection.flatten().sum()) / union
    return res

def agg_iou(masks_gt, masks, thr=0.0):
    masks_gt = [(m>thr).astype('int') for m in masks_gt]
    masks = [(m>thr).astype('int') for m in masks]
    cands = np.zeros((len(masks_gt), len(masks)))
    
    # Compare all ground truths to masks
    for i, m1 in enumerate(masks_gt):
        for j, m2 in enumerate(masks):
            cands[i, j] = iou(m1, m2)
            
    # Descending ordering induced by index
    ys, xs = np.unravel_index(np.argsort(-cands, axis=None), cands.shape)
    
    # Select matches greedily
    vis_ys = set()
    vis_xs = set()
    ious = []
    for y, x in zip(ys, xs):
        if y not in vis_ys and x not in vis_xs:
            ious.append((y, x, cands[y, x]))
            vis_ys.add(y)
            vis_xs.add(x)
            
    # Some masks in the ground truth were not matched, add entries for score computation
    for i, _ in enumerate(masks_gt):
        if not i in vis_ys:
            ious.append((i, -1, 0.0))
    
    # Some predictions were not matched, add entries for score computation
    for j, _ in enumerate(masks):
        if not j in vis_xs:
            ious.append((-1, j, 0.0))
    
    return ious

def precision_at_thr(ious, thr):
    number_gt = len(set([iou[0] for iou in ious]))
    number_pred = len(set([iou[1] for iou in ious]))

    ious = [iou for iou in ious if iou[2] > thr]
    detected_gt = len(set([iou[0] for iou in ious]))
    used_predicted = len(set([iou[1] for iou in ious]))

    true_positives = min(detected_gt, used_predicted)
    false_negatives = number_gt - true_positives
    false_positives = number_pred - true_positives
    return float(true_positives) / float(true_positives + false_negatives + false_positives)

def precision_iou(masks_gt, masks):
    ious = agg_iou(masks_gt, masks)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = [precision_at_thr(ious, threshold) for threshold in thresholds]
    # print(precisions)
    return np.mean(precisions)

