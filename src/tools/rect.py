import functools

def intersect(r1, r2):
    (x1, y1, x2, y2), (x1_, y1_, x2_, y2_) = r1, r2
    return (max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_))

def union(r1, r2):
    (x1, y1, x2, y2), (x1_, y1_, x2_, y2_) = r1, r2
    return (min(x1, x1_), min(y1, y1_), max(x2, x2_), max(y2, y2_))

def area(r):
    (x1, y1, x2, y2) = r
    return (y2 - y1) * (x2 - x1)

def non_empty(r):
    (x1, y1, x2, y2) = r
    return x2 > x1 and y2 > y1

def overlap(r1, r2, thr=0.5):
    r12 = intersect(r1, r2)
    return non_empty(r12) and area(r12) >= thr * area(r1) and area(r12) >= thr * area(r2)

def merge(rects):
    processed = []
    while True:
        if len(rects) < 2:
            return processed + rects
        r1 = rects[0]
        overlapping = [r for r in rects[1:] if overlap(r1, r)]
        non_overlapping = [r for r in rects[1:] if not overlap(r1, r)]

        r1_ = functools.reduce(union, [r1] + overlapping)
        if overlapping:
            rects = [r1_] + non_overlapping
        else:
            processed.append(r1_)
            rects = non_overlapping

