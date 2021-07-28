import os
import numpy as np



def calc_IoU(a, b, a_box, b_box):
    # step1:
    x1 = np.maximum(a_box[0], b_box[0])
    y1 = np.maximum(a_box[1], b_box[1])
    x2 = np.minimum(a_box[2], b_box[2])
    y2 = np.minimum(a_box[3], b_box[3])
    w = x2-x1
    h = y2-y1
    if w<=0 or h<=0:
        return 0.
    # step 2
    x1 = int(np.minimum(a_box[0], b_box[0]))
    y1 = int(np.minimum(a_box[1], b_box[1]))
    x2 = int(np.maximum(a_box[2], b_box[2]))
    y2 = int(np.maximum(a_box[3], b_box[3]))

    mask_a = np.zeros((y2-y1+1, x2-x1+1))
    mask_b = np.zeros((y2-y1+1, x2-x1+1))

    mask_a[int(a_box[1]-y1): int(a_box[1]-y1)+a.shape[0], int(a_box[0]-x1): int(a_box[0]-x1)+a.shape[1]] = a
    mask_b[int(b_box[1]-y1): int(b_box[1]-y1)+b.shape[0], int(b_box[0]-x1): int(b_box[0]-x1)+b.shape[1]] = b

    inter = (mask_a*mask_b).sum()
    union = mask_a.sum()+mask_b.sum()-inter.sum()
    IoU = inter/(union+1e-6)
    return IoU


def non_maximum_suppression_numpy_masks(masks, bboxes, nms_thresh=0.5):
    """
    bboxes: num_insts x 5 [y1,y2,x1,x2,conf]
    """
    if len(bboxes)==0:
        return None
    # x1 = bboxes[:,0]
    # y1 = bboxes[:,1]
    # x2 = bboxes[:,2]
    # y2 = bboxes[:,3]
    conf = bboxes[:,4]
    sorted_index = np.argsort(conf)      # Ascending order
    keep_index = []

    while len(sorted_index)>0:
        # get the last biggest values
        curr_index = sorted_index[-1]
        keep_index.append(curr_index)
        if len(sorted_index)==1:
            break
        # pop the value
        sorted_index = sorted_index[:-1]
        IoU = []
        for index in sorted_index:
            IoU.append(calc_IoU(masks[index],
                                masks[curr_index],
                                bboxes[index,:4],
                                bboxes[curr_index,:4]))
        IoU = np.asarray(IoU, np.float32)
        sorted_index = sorted_index[IoU<=nms_thresh]
    return keep_index



def non_maximum_suppression_numpy(bboxes, nms_thresh=0.5):
    """
    bboxes: num_insts x 5 [y1,y2,x1,x2,conf]
    """
    if len(bboxes)==0:
        return None
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    conf = bboxes[:,4]
    area_all = (x2-x1)*(y2-y1)
    sorted_index = np.argsort(conf)      # Ascending order
    keep_index = []

    while len(sorted_index)>0:
        # get the last biggest values
        curr_index = sorted_index[-1]
        keep_index.append(curr_index)
        if len(sorted_index)==1:
            break
        # pop the value
        sorted_index = sorted_index[:-1]
        # get the remaining boxes
        yy1 = np.take(y1, indices=sorted_index)
        xx1 = np.take(x1, indices=sorted_index)
        yy2 = np.take(y2, indices=sorted_index)
        xx2 = np.take(x2, indices=sorted_index)
        # get the intersection box
        yy1 = np.maximum(yy1, y1[curr_index])
        xx1 = np.maximum(xx1, x1[curr_index])
        yy2 = np.minimum(yy2, y2[curr_index])
        xx2 = np.minimum(xx2, x2[curr_index])
        # calculate IoU
        w = xx2-xx1
        h = yy2-yy1

        w = np.maximum(0., w)
        h = np.maximum(0., h)

        inter = w*h

        rem_areas = np.take(area_all, indices=sorted_index)
        union = (rem_areas-inter)+area_all[curr_index]
        IoU = inter/union
        sorted_index = sorted_index[IoU<=nms_thresh]

    # pr_masks = np.take(pr_masks, keep_index, axis=0)
    # pr_bboxes = np.take(pr_bboxes, keep_index, axis=0)
    return keep_index
