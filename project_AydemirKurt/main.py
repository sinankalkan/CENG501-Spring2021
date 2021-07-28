import argparse
import numpy as np
import os
from module import Module


def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=500, help='maximum of objects')
    parser.add_argument('--nms_thresh', type=float, default=0.2, help='non maximum threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='detection confidence threshold')
    parser.add_argument('--seg_thresh', type=float, default=0.53, help='segmentation binary threshold')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=0, help='number of gpus, 0 means only 1 gpu')
    parser.add_argument('--dataset', type=str, default='neural', help='dataset name, your need to create your own in module.py')
    parser.add_argument('--data_dir', type=str, default='../../Datasets/', help='data directory')
    parser.add_argument('--eval_type', type=str, default='seg', help='evaluation type')
    parser.add_argument('--phase', type=str, default='train', help='phase: train, val, test')

    args = parser.parse_args()
    return args


def run_seg_ap(object_is, args):
    print('evaluating segmentation using PASCAL2010 metric')
    thresh = np.linspace(0.5, 0.95, 10)
    ap_list = []
    iou_list = []
    for v in thresh:
        ap, iou = object_is.seg_eval(args, ov_thresh=v, use_07_metric=False)
        ap_list.append(ap*100)
        iou_list.append(iou*100)
    np.savetxt(os.path.join('weights_'+args.dataset, 'seg_ap_list.txt'), ap_list, fmt='%.4f')
    np.savetxt(os.path.join('weights_'+args.dataset, 'seg_iou_list.txt'), iou_list, fmt='%.4f')
    print(np.mean(ap_list))



def run_dec_ap(object_is, args):
    print('evaluating detection using PASCAL2010 metric')
    thresh = np.linspace(0.5, 0.95, 10)
    ap_list = []
    for v in thresh:
        ap = object_is.dec_eval(args, ov_thresh=v, use_07_metric=False)
        ap_list.append(ap*100)
    np.savetxt(os.path.join('weights_'+args.dataset, 'dec_ap_list.txt'), ap_list, fmt='%.4f')
    print(np.mean(ap_list))


if __name__ == '__main__':
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    resume_files = {
        'kaggle': 'model_99.pth',
        'neural': 'model_80.pth',
        'plant': 'model_85.pth',
        'default': 'model_last.pth'
    }

    args.resume = resume_files[args.dataset]

    obj_is = Module(args)
    if args.phase == 'train':
        obj_is.train(args)
    elif args.phase == 'test':
        obj_is.test(args)
    elif args.phase == 'eval':
        if args.eval_type == 'seg':
            run_seg_ap(obj_is, args)
        else:
            run_dec_ap(obj_is, args)
