import numpy as np

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)


def find_max_overlap(d_BB_mask, d_BB_box, BBGT_mask, BBGT_bboxes):
    # BBGT_bboxes: [y1,x1,y2,x2]
    # glue_bboxes: [g_y1, g_x1, g_y2, g_x2]
    num_gt, image_h, image_w = BBGT_mask.shape
    ovmax = -np.inf
    jmax = -1
    # glue pred_mask
    mask = np.zeros(shape=(image_h, image_w), dtype=np.float32)
    x1,y1,x2,y2 = d_BB_box
    mask[int(y1):int(y1)+d_BB_mask.shape[0],
         int(x1):int(x1)+d_BB_mask.shape[1]] = d_BB_mask
    # keep index
    ixmin = np.maximum(BBGT_bboxes[:, 0], x1)
    iymin = np.maximum(BBGT_bboxes[:, 1], y1)
    ixmax = np.minimum(BBGT_bboxes[:, 2], x2)
    iymax = np.minimum(BBGT_bboxes[:, 3], y2)
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih
    keep_index = inters>0.

    for i in range(num_gt):
        if keep_index[i]:
            overlaps = mask_iou(mask, BBGT_mask[i,:,:])
            if overlaps > ovmax:
                ovmax = overlaps
                jmax = i
    return ovmax, jmax

def sorted_BB_mask(BB_mask, sorted_ind):
    out_masks = []
    for ind in sorted_ind:
        out_masks.append(BB_mask[ind])
    return out_masks



def seg_evaluation(BB_mask, BB_bboxes, dsets, all_scores,
                   img_id, npos, temp_overlaps, ov_thresh=0.5):
    if np.any(BB_bboxes):
        BB_conf = np.asarray(BB_bboxes[:,4], dtype=np.float32)
        # Step2: sort detections according to the confidences
        sorted_ind = np.argsort(-BB_conf)
        BB_mask = sorted_BB_mask(BB_mask, sorted_ind)
        BB_conf = BB_conf[sorted_ind]
        BB_bboxes = BB_bboxes[sorted_ind, :4]
        all_scores.extend(BB_conf)
        nd = BB_bboxes.shape[0]
    else:
        nd = 0
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    BBGT_mask, BBGT_bboxes = dsets.load_gt_masks_bboxes(dsets.load_annoFolder(img_id))
    nd_gt = BBGT_mask.shape[0]
    det_flag = [False] * nd_gt
    npos = npos + nd_gt

    for d in range(nd):
        d_BB_mask = BB_mask[d]
        d_BB_bbox = BB_bboxes[d]
        ovmax, jmax = find_max_overlap(d_BB_mask, d_BB_bbox, BBGT_mask, BBGT_bboxes)
        if ovmax > ov_thresh:
            if not det_flag[jmax]:
                tp[d] = 1.
                det_flag[jmax] = 1
                temp_overlaps.append(ovmax)
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    return fp, tp, all_scores, npos, temp_overlaps



def dec_evaluation(BB_bboxes, dsets, all_scores, img_id, npos, ov_thresh):
    if np.any(BB_bboxes):
        pr_conf = BB_bboxes[:, 4]
        pr_bboxes = BB_bboxes[:, :4]
        sorted_ind = np.argsort(-pr_conf)
        pr_bboxes = pr_bboxes[sorted_ind, :]
        pr_conf = pr_conf[sorted_ind]
        all_scores.extend(pr_conf)
        nd = pr_bboxes.shape[0]
    else:
        nd = 0
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    BBGT_box = dsets.load_gt_bboxes(dsets.load_annoFolder(img_id))
    nd_gt = BBGT_box.shape[0]
    det_flag = [False] * nd_gt
    npos = npos + nd_gt

    for d in range(nd):
        bb = pr_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = BBGT_box.astype(float)
        jmax = -1
        if BBGT.shape[0] > 0:
            iymin = np.maximum(BBGT[:, 0], bb[0])
            ixmin = np.maximum(BBGT[:, 1], bb[1])
            iymax = np.minimum(BBGT[:, 2], bb[2])
            ixmax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            union = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                     (BBGT[:, 2] - BBGT[:, 0]) *
                     (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax >= ov_thresh:
            if not det_flag[jmax]:
                tp[d] = 1.
                det_flag[jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    return fp, tp, all_scores, npos
