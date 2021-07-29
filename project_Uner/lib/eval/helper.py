import numpy as np
from PIL import Image
import cv2


def green_background(image, mask):
    green_image = np.zeros_like(image)
    green_image[:, :, :] = [0, 255, 0]
    out_img = (image * mask + (green_image * (1 - mask))).astype("uint8")
    out_img = Image.fromarray(out_img)
    return out_img


def get_iu(seg, gt):
    intersection = np.count_nonzero(seg & gt)
    union = np.count_nonzero(seg | gt)

    return intersection, union


def get_disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))


def compute_boundary_acc(gt, seg):
    gt = gt.astype(np.uint8)
    seg = seg.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/300
    num_steps = 5

    seg_acc = [None] * num_steps

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

        kernel = get_disk_kernel(curr_radius)
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        gt_in_bound = gt[boundary_region]
        seg_in_bound = seg[boundary_region]

        num_edge_pixels = (boundary_region).sum()
        num_seg_gd_pix = ((gt_in_bound) * (seg_in_bound) + (1-gt_in_bound) * (1-seg_in_bound)).sum()

        seg_acc[i] = num_seg_gd_pix / num_edge_pixels

    return sum(seg_acc)/num_steps