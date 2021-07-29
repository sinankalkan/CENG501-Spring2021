import argparse
import numpy as np
from PIL import Image
from pathlib import Path

from tqdm import tqdm

from lib.eval.helper import get_iu, compute_boundary_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOS Evaluation')
    parser.add_argument('-t', '--gt-dir', help='Path to ground truth mask folder', type=str,
                        default='/home/onur/Desktop/pixery/datasets/HRSOD_release/HRSOD_test_mask')
                        # default='/home/onur/Desktop/pixery/datasets/DUTS-TE/DUTS-TE-Mask')
    parser.add_argument('-p', '--pred-dir', help='Path to predicted mask folder', type=str,
                        default='/home/onur/Desktop/pixery/workspace/mos/saved/outputs/HRSOD/fine_mask')
                        # default='/home/onur/Desktop/pixery/workspace/mos/saved/outputs/DUTS-TE/coarse_mask')
    args = parser.parse_args()

    gt_paths = Path(args.gt_dir).glob("*.png")
    pred_dir = Path(args.pred_dir)

    total_intersection = 0
    total_union = 0
    total_boundary_acc = 0
    total_num_images = 0

    tbar = tqdm(list(gt_paths))
    for gt_path in tbar:
        gt = np.array(Image.open(gt_path).convert('L'))
        pred = np.array(Image.open(pred_dir / gt_path.name).convert('L'))
        assert gt.shape == pred.shape
        gt = gt > 128
        pred = pred > 128
        intersection, union = get_iu(gt, pred)
        boundary_acc = compute_boundary_acc(gt, pred)

        total_intersection += intersection
        total_union += union
        total_boundary_acc += boundary_acc
        total_num_images += 1

    iou = total_intersection / total_union
    mba = total_boundary_acc / total_num_images
    print(f'IoU: {iou}, mBA: {mba}')