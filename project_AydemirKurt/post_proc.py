import numpy as np
import cv2
from scipy.sparse import coo_matrix
from datasets import affine_funcs


def accumulate_votes(votes, shape):
    # xs, ys must be smaller than shape size
    # shape: h x w
    # Hough Voting
    xs = votes[:, 0]
    ys = votes[:, 1]
    ps = votes[:, 2]
    tl = [np.floor(ys).astype('int32'), np.floor(xs).astype('int32')]
    tr = [np.floor(ys).astype('int32'), np.ceil(xs).astype('int32')]
    bl = [np.ceil(ys).astype('int32'), np.floor(xs).astype('int32')]
    br = [np.ceil(ys).astype('int32'), np.ceil(xs).astype('int32')]
    dx = xs - tl[1]
    dy = ys - tl[0]
    tl_vals = ps * (1. - dx) * (1. - dy)
    tr_vals = ps * dx * (1. - dy)
    bl_vals = ps * dy * (1. - dx)
    br_vals = ps * dy * dx
    data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
    I = np.concatenate([tl[0], tr[0], bl[0], br[0]])
    J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
    good_inds = np.logical_and(I >= 0, I < shape[0])
    good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))
    constructed_mask = np.asarray(coo_matrix((data[good_inds], (I[good_inds], J[good_inds])), shape=shape).todense())
    return constructed_mask

def affine_mask_process(dsets,
                        img_id,
                        pr_masks,
                        pr_bboxes0,
                        seg_thresh):

    ori_image = dsets.load_image(dsets.img_ids.index(img_id))
    image_h, image_w, c = ori_image.shape
    out_masks = []
    out_bboxes = []
    for b in range(len(pr_masks)):
        for n in range(len(pr_masks[b])):
            mask_patch = pr_masks[b][n].cpu().numpy()
            bbox = pr_bboxes0[b][n]
            cenx, ceny, w, h, score = bbox
            output = affine_funcs.glue_back_masks(mask_patch,
                                                  bbox[:4],
                                                  image_h=512,
                                                  image_w=512,
                                                  seg_thresh=seg_thresh)
            constructed_mask, g_x1, g_y1, g_w, g_h = output
            if output is None:
                print('None')
                continue
            mask_new = np.zeros((512, 512))
            mask_new[g_y1:g_y1+g_h, g_x1:g_x1+g_w] = constructed_mask
            mask_new = cv2.resize(mask_new, (image_w, image_h), cv2.INTER_NEAREST)
            rr,cc = np.where(mask_new==1.)
            if rr.any():
                y1 = np.min(rr)
                x1 = np.min(cc)
                y2 = np.max(rr)
                x2 = np.max(cc)
                mask_path = mask_new[y1:y2+1, x1:x2+1]
                out_masks.append(mask_path)
                out_bboxes.append([x1,y1,x2,y2,score])

    return out_masks, np.asarray(out_bboxes)











