import numpy as np
from . import rect


def extract_rectangles(mask, widths, heights, diffx, diffy):
    diffx = np.squeeze(diffx)
    diffy = np.squeeze(diffx)
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


def extract_masks(mask, widths, heights, diffx, diffy):
    rects = extract_rectangles(mask, widths, heights, diffx, diffy)
    masks = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        aux_mask = np.zeros(mask.shape)
        aux_mask[int(y1):int(y2)+1, int(x1):int(x2)] = 1
        masks.append(mask * aux_mask)
    return masks

def iou(m1, m2):
    intersection = (m1*m2).flatten().sum()
    union = float(m1.flatten().sum() + m2.flatten().sum() - intersection)
    return float(intersection) / union

def agg_iou(masks_gt, masks):
    masks_gt = [(m>0).astype('int') for m in masks_gt]
    masks = [(m>0).astype('int') for m in masks]
    return [(i, j, iou(m1, m2))
            for i, m1 in enumerate(masks_gt)
            for j, m2 in enumerate(masks)]

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
    return np.mean([precision_at_thr(ious, threshold) for threshold in thresholds])

