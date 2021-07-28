import torch
import torch.nn as nn
import os
import numpy as np
import loss
import decoder
import post_proc
from datasets.dataset_neural import Neural
from datasets.dataset_plant import Plant
from datasets.dataset_kaggle import Kaggle
import eval_parts
import nms
from models.network import Network
import time
import cv2
from datasets import affine_funcs
import matplotlib.pyplot as plt


def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def collater(data):
    batch_data_dict = {}
    for name in data[0]:
        batch_data_dict[name] = []
    # iterate batch
    for sample in data:
        for name in sample:
            batch_data_dict[name].append(sample[name])
    for name in batch_data_dict:
        if name not in ['gt_bboxes', 'gt_rois', 'img_id']:
            batch_data_dict[name] = torch.stack(batch_data_dict[name], dim=0)
    return batch_data_dict


class Module(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'wh': 2,
                 'reg': 2}
        self.model = Network(heads=heads,
                             pretrained=True,
                             down_ratio=args.down_ratio,
                             final_kernel=1,
                             head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'kaggle':Kaggle, 'plant':Plant, 'neural': Neural}


    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def save_model(self, path, epoch, model):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch, 'state_dict': state_dict}
        torch.save(data, path)

    def load_model(self, model, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

        return model

    def set_device(self, ngpus, device):
        if ngpus > 1:
            self.model = nn.DataParallel(self.model).to(device)
        else:
            self.model = self.model.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def train(self, args):
        weights_file = 'weights_'+args.dataset
        if not os.path.exists(weights_file):
            os.mkdir(weights_file)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())   
        print(pytorch_total_params)

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.set_device(args.ngpus, self.device)

        criterion = loss.CtdetLoss()

        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=args.down_ratio)
                 for x in ['train', 'val']}

        dsets_loader = {'train': torch.utils.data.DataLoader(dsets['train'],
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=args.num_workers,
                                                             pin_memory=True,
                                                             drop_last=True,
                                                             collate_fn=collater),

                        'val':torch.utils.data.DataLoader(dsets['val'],
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=1,
                                                          pin_memory=True,
                                                          collate_fn=collater)}


        print('Starting training...')
        train_loss = []
        val_loss = []
        # ap_05 = []
        # ap_07 = []
        # iou_05 = []
        # iou_07 = []
        for epoch in range(1, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            train_loss.append(epoch_loss)

            epoch_loss = self.run_epoch(phase='val',
                                        data_loader=dsets_loader['val'],
                                        criterion=criterion)
            val_loss.append(epoch_loss)

            np.savetxt(os.path.join(weights_file, 'train_loss.txt'), train_loss, fmt='%.6f')
            np.savetxt(os.path.join(weights_file, 'val_loss.txt'), val_loss, fmt='%.6f')

            self.save_model(os.path.join(weights_file, 'model_last.pth'), epoch, self.model)
            if epoch % 5 == 0 or epoch ==1:
                self.save_model(os.path.join(weights_file, 'model_{}.pth'.format(epoch)), epoch, self.model)
                # ap_05_out, iou_05_out = self.seg_eval(args=args, ov_thresh=0.5)
                # ap_07_out, iou_07_out = self.seg_eval(args=args, ov_thresh=0.7)
                # ap_05.append(ap_05_out)
                # ap_07.append(ap_07_out)
                # iou_05.append(iou_05_out)
                # iou_07.append(iou_07_out)
                # np.savetxt('ap_05.txt', ap_05, fmt='%.6f')
                # np.savetxt('ap_07.txt', ap_07, fmt='%.6f')
                # np.savetxt('iou_05.txt', iou_05, fmt='%.6f')
                # np.savetxt('iou_07.txt', iou_07, fmt='%.6f')


    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        for data_dict in data_loader:
            for name in data_dict:
                if name not in ['gt_bboxes', 'gt_rois']:
                    data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_dict = self.model(data_dict)
                    loss = criterion(data_dict, pr_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_dict = self.model(data_dict)
                    loss = criterion(data_dict, pr_dict)

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss

    def post_processing(self, args, pr_dict, dsets, img_id):
        image = dsets.load_image(dsets.img_ids.index(img_id))
        pr_bboxes = []
        pr_rois = []
        for bbox, roi in zip(pr_dict['pr_bboxes'], pr_dict['pr_rois']):
            roi = roi.data.cpu().numpy()
            bbox[2] *= 1.1
            bbox[3] *= 1.1
            cenx, ceny, w, h, score, cls = bbox
            output = affine_funcs.glue_back_masks(roi,
                                                  bbox[:4],
                                                  image_h=args.input_h,
                                                  image_w=args.input_w,
                                                  seg_thresh=args.seg_thresh)
            constructed_mask, g_x1, g_y1, g_w, g_h = output
            if output is None:
                print('None')
                continue
            mask_new = np.zeros((args.input_h, args.input_w))
            mask_new[g_y1:g_y1+g_h, g_x1:g_x1+g_w] = constructed_mask
            mask_new = cv2.resize(mask_new, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
            rr,cc = np.where(mask_new==1.)
            if rr.any():
                y1 = np.min(rr)
                x1 = np.min(cc)
                y2 = np.max(rr)
                x2 = np.max(cc)
                pr_rois.append(mask_new[y1:y2+1, x1:x2+1])
                pr_bboxes.append([x1,y1,x2,y2,score])

        pr_bboxes = np.asarray(pr_bboxes, np.float32)
        keep_index = nms.non_maximum_suppression_numpy_masks(pr_rois, pr_bboxes, nms_thresh=args.nms_thresh)
        if len(keep_index)!=len(pr_rois):
            pr_rois = np.take(pr_rois, keep_index, axis=0)
            pr_bboxes = np.take(pr_bboxes, keep_index, axis=0)
        out_dict = {'pr_bboxes': pr_bboxes,
                    'pr_rois': pr_rois}
        return out_dict

    def run_test(self, args, data_dict, dsets):
        for name in data_dict:
            if name in ['image']:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
        with torch.no_grad():
            pr_dict = self.model.forward_dec(data_dict['image'])
        torch.cuda.synchronize(self.device)
        pr_bboxes = self.decoder.ctdet_decode(pr_dict)  # [cenx, ceny, w, h, score, cls]
        pr_bboxes[:, :4] *= args.down_ratio
        if np.any(pr_bboxes):
            data_dict['pr_bboxes'] = pr_bboxes
            with torch.no_grad():
                pr_dict = self.model.forward_seg_test(data_dict, pr_dict)
            out_dict = self.post_processing(args, pr_dict, dsets, data_dict['img_id'][0])
        else:
            out_dict = {'pr_bboxes': [],
                        'pr_rois': []
                        }
        return out_dict

    def test(self, args):
        save_path = os.path.join('weights_'+args.dataset, 'save_imgs')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model = self.load_model(self.model, os.path.join('weights_'+args.dataset, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]

        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
        total_time = []
        for cnt, data_dict in enumerate(data_loader):
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))
            begin_time = time.time()
            out_dict = self.run_test(args, data_dict, dsets)
            total_time.append(time.time() - begin_time)

            img_id = data_dict['img_id'][0]
            print(img_id)
            ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            image_h, image_w, c = ori_image.shape
            pr_copy = ori_image.copy()
            annoFolder = dsets.load_annoFolder(img_id)
            BBGT_mask = dsets.load_gt_masks(annoFolder)
            for i in range(BBGT_mask.shape[0]):
                mask = BBGT_mask[i, :, :]
                ori_image = self.map_mask_to_image(mask, ori_image)
            pr_image = pr_copy.copy()

            for roi, bbox in zip(out_dict['pr_rois'], out_dict['pr_bboxes']):
                x1, y1, x2, y2 = np.asarray(bbox[:4], np.int32)
                mask = np.zeros(shape=(image_h, image_w), dtype=np.float32)
                mask[y1:y2 + 1, x1:x2 + 1] = roi
                pr_image = apply_mask(pr_image, mask, alpha=0.8)
            if not args.dataset == 'kaggle':
                img_id = img_id[:-4]
            cv2.imwrite(os.path.join(save_path, img_id+'.png'), np.uint8(pr_image))


        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))


    def seg_eval(self, args, ov_thresh, use_07_metric=False):
        self.model = self.load_model(self.model, os.path.join('weights_'+args.dataset, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]

        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
        all_tp = []
        all_fp = []
        all_scores = []
        temp_overlaps = []
        npos = 0
        for cnt, data_dict in enumerate(data_loader):
            # print('processing {}/{} image ...'.format(cnt, len(data_loader)))
            out_dict = self.run_test(args, data_dict, dsets)
            fp, tp, all_scores, npos, temp_overlaps = eval_parts.seg_evaluation(BB_mask=out_dict['pr_rois'],
                                                                                BB_bboxes=out_dict['pr_bboxes'],
                                                                                dsets=dsets,
                                                                                all_scores=all_scores,
                                                                                img_id=data_dict['img_id'][0],
                                                                                npos=npos,
                                                                                temp_overlaps=temp_overlaps,
                                                                                ov_thresh=ov_thresh)
            all_fp.extend(fp)
            all_tp.extend(tp)
        # step5: compute precision recall
        all_fp = np.asarray(all_fp)
        all_tp = np.asarray(all_tp)
        all_scores = np.asarray(all_scores)
        sorted_ind = np.argsort(-all_scores)
        all_fp = all_fp[sorted_ind]
        all_tp = all_tp[sorted_ind]
        all_fp = np.cumsum(all_fp)
        all_tp = np.cumsum(all_tp)
        rec = all_tp / float(npos)
        prec = all_tp / np.maximum(all_tp + all_fp, np.finfo(np.float64).eps)

        ap = eval_parts.voc_ap(rec, prec, use_07_metric=use_07_metric)
        print("ap@{} is {}, iou is {}".format(ov_thresh, ap, np.mean(temp_overlaps)))
        return ap, np.mean(temp_overlaps)
    


    def dec_eval(self, args, ov_thresh, use_07_metric=False):
        self.model = self.load_model(self.model, os.path.join('weights_'+args.dataset, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]

        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
        all_tp = []
        all_fp = []
        all_scores = []
        npos = 0
        for cnt, data_dict in enumerate(data_loader):
            # print('processing {}/{} image ...'.format(cnt, len(data_loader)))
            out_dict = self.run_test(args, data_dict, dsets)
            fp, tp, all_scores, npos = eval_parts.dec_evaluation(BB_bboxes=out_dict['pr_bboxes'],
                                                                    dsets=dsets,
                                                                    all_scores=all_scores,
                                                                    img_id=data_dict['img_id'][0],
                                                                    npos=npos,
                                                                    ov_thresh=ov_thresh)  
            all_fp.extend(fp)
            all_tp.extend(tp)
        # step5: compute precision recall
        all_fp = np.asarray(all_fp)
        all_tp = np.asarray(all_tp)
        all_scores = np.asarray(all_scores)
        sorted_ind = np.argsort(-all_scores)
        all_fp = all_fp[sorted_ind]
        all_tp = all_tp[sorted_ind]
        all_fp = np.cumsum(all_fp)
        all_tp = np.cumsum(all_tp)
        rec = all_tp / float(npos)
        prec = all_tp / np.maximum(all_tp + all_fp, np.finfo(np.float64).eps)

        ap = eval_parts.voc_ap(rec, prec, use_07_metric=use_07_metric)
        print("ap@{} is {}".format(ov_thresh, ap))
        return ap