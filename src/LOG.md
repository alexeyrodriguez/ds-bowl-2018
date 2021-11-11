Thursday November 11, 2021
==========================

Precision for IOU drops for the first image because it turns out
I was using ground truth centers for predicting rectangles rather than
predictions. Once this was fixed the IOU precision becomes 0.34.

Also added to the IOU computation unmatched ground truths and masks (IOU of 0)

Applied rounding when converting to ints, IOU precision goes up to 0.428.



Monday October 25, 2021
=======================

Today:
* Encapsulate some functionality better
* Debug which masks decrease the IOU-derived scores

* IOUs for smallest images is the smallest. Could we try to increase the pixel precision for the
  smallest images? IOUs at this point (0.366, 0.38, 0.464, 0.558)


Monday October 18, 2021
=======================

Reviewed the metric calculation that I stole from some Kaggle forum.

It was bogus not to compute the matches between predictions and ground truths.

I did it and now I think the threshold prediction makes sense. However the
metric is quite low as the IOUs of the different masks is relatively low.

We would need to increase the metric by increasing the IOU.

Currently I suppose the problems are here:
* Mask still does not match ground truth well.
* Bounding boxes cannot segment nuclei that are really close by.

A change of design without bounding boxes might be necessary as per the winning
solutions.

However the goal of improving the bounding box regression was achieved.

How to know whether abandoning the bounding box approach is the next step (which I won't take),
if we can't get a high score on overfitted examples, probably the approach has inherent limitations
(bounding box).

I will compute a maskwise IOU to see how much we are missing from the mask itself.

Current score for first overfitted image is: 0.3678

(Mainly because we don't have more matches with higher IOUs).

With a threshold of 0.35, we get an IOU of 0.85 for the entire mask.
With the previous threshold of 0.5, it would be 0.834.

And now the precision metric goes up to 0.41 when we use the new threshold of 0.35.
Probably we could train longer to get the precision further up.



Tuesday October 12, 2021
========================

We implemented a training where there is higher weight given to the deltas and less
weight to overlapping prediction.

For the two images that were overfitted now we have a nice bounding of the cells that
were detected.

The next step is to extract perhaps the masks and compute some loss? Would that be
worth doing? Perhaps yes.

Then we have a good first baseline and we can call it done.

The good news is that we fixed the size of rectangles and so on.

Monday October 11, 2021
=======================

We have fixed a couple of things with data loading.

At the moment we want to see how high quality the centers and overlaps can be.

So we are trying to overfit the network to see if it has enough capacity for it.

At 600 epochs with a loss_weight of 10000 for overlaps it matches the reference pretty good
when training with two images. Overfitting works.

Now with centers.

It also works with centers.

In order to make it work decently with both centers and overlaps I had to raise epochs to 800

Now with 800 epochs and with 50 as loss weight for the other tasks I think I have a good setup
for an MVP. The deltas for the bounding box are not looking great, but if use the centers prediction,
it might be good enough.

python train_unet_multi_task_2.py --output ../../dsb-2018/models/model-dsbowl2018-unet-ext_2.h5 --epochs 800 --samples 2

===

What is the next thing to do?

Weighted learning, what was that for?

It was for the following:

ValueError: `class_weight` is only supported for Models with a single output.
Appears when running train_unet_multi_task_2.py

Some ideas that we can use:
https://datascience.stackexchange.com/questions/41698/how-to-apply-class-weight-to-a-multi-output-model





Some command lines:

cd ds-bowl-2018/src/

python train_unet_multi_task.py --output ../../dsb-2018/models/model-dsbowl2018-unet-ext.h5 --epochs 100 --samples 32



python train_unet_multi_task_2.py --output ../../dsb-2018/models/model-dsbowl2018-unet-ext_2.h5 --epochs 100 --samples 32



