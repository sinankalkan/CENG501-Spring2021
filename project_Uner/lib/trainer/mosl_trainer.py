import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from lib.base import BaseTrainer
from lib.utils import AverageMeter
import lib.metrics as metrics


class MosLTrainer(BaseTrainer):
    def __init__(self, config, resume, train_loader, save_dir, log_dir, val_loader=None):
        super().__init__(config, resume, train_loader, save_dir, log_dir, val_loader)

        self.train_vis = self.config.trainer.get('visualize_train_batch', False)
        self.val_vis = self.config.trainer.get('visualize_val_batch', False)
        self.vis_count = self.config.trainer.get('vis_count', len(self.train_loader))
        self.log_per_batch = self.config.trainer.get('log_per_batch', int(np.sqrt(self.train_loader.batch_size)))

        # Losses
        self.criterion_bce = nn.BCELoss().to(self.device)

    def _train_epoch(self, epoch):
        vis_save_dir = self.visualize_dir / 'train' / str(epoch)
        vis_save_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()
        train_vis_count = 0
        tic = time.time()
        self._reset_metrics()

        tbar = tqdm(self.train_loader)
        for batch_idx, data in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            batches_done = (epoch-1) * len(self.train_loader) + batch_idx

            image = data['image'].to(self.device)
            label = data['mask'].to(self.device)
            self.optimizer.zero_grad()

            # LOSS & OPTIMIZE
            loss = 0
            predictions, uc_maps = self.model(image)
            for pred in predictions:
                loss += self.criterion_bce(pred, label)
            loss = loss / len(predictions)

            # total loss
            loss.backward()
            self.optimizer.step()

            # update metrics
            self.loss_meter.update(loss.item())
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # Visualize batch & Tensorboard log
            if batch_idx % self.log_per_batch == 0:
                self._log_train_tensorboard(batches_done)
                if train_vis_count < self.vis_count and self.train_vis:
                    train_vis_count += image.shape[0]
                    self._visualize_batch(image, label, predictions, batch_idx, vis_save_dir, predef='mask')
                    self._visualize_batch(image, label, uc_maps, batch_idx, vis_save_dir, predef='ucmap')

            tbar.set_description(self._training_summary(epoch))

    def _valid_epoch(self, epoch):
        vis_save_dir = self.visualize_dir / 'test' / str(epoch)
        vis_save_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.val_loader)
        with torch.no_grad():
            val_vis_count = 0
            for batch_idx, data in enumerate(tbar):
                image = data['image'].to(self.device)
                label = data['mask'].to(self.device)

                loss = 0
                predictions, _ = self.model(image)
                for pred in predictions:
                    loss += self.criterion_bce(pred, label)

                self.loss_meter.update(loss.item())
                self._update_metrics(pred, label)

                # Visualize batch
                if val_vis_count < self.vis_count and self.val_vis:
                    val_vis_count += image.shape[0]
                    self._visualize_batch(image, label, predictions, batch_idx, vis_save_dir)

                # PRINT INFO
                if batch_idx == len(tbar)-1:
                    tbar.set_description(self._validation_summary(epoch))

            self._log_validation_tensorboard(epoch)
        return self.sqrt_w_iou.compute()

    def _update_metrics(self, pred, gt):
        pred = pred.detach().squeeze().cpu().numpy()
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        pred = pred.astype(np.int64)
        gt = gt.detach().squeeze().long().cpu().numpy()
        self.cm.update(pred, gt)

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter = AverageMeter()

        self.cm = metrics.ConfusionMatrix(num_classes=2)
        self.accuracy = metrics.Accuracy(self.cm)
        self.recall = metrics.Recall(self.cm)
        self.class_0_recall = metrics.Recall(self.cm, average=False)[0]
        self.class_1_recall = metrics.Recall(self.cm, average=False)[1]
        self.mean_iou = metrics.mIoU(self.cm)
        self.freq_w_iou = metrics.FreqWIoU(self.cm)
        self.sqrt_w_iou = metrics.SqrtWIoU(self.cm)

    def _log_train_tensorboard(self, step):
        self.write_item(name='loss', value=self.loss_meter.avg, step=step)

        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.write_item(name=f'Learning_rate_generator_{i}', value=opt_group['lr'], step=self.wrt_step)

    def _log_validation_tensorboard(self, step):
        self.write_item(name='loss', value=self.loss_meter.avg, step=step)
        self.write_item(name='acc', value=self.accuracy.compute(), step=step)
        self.write_item(name='miou', value=self.mean_iou.compute(), step=step)
        self.write_item(name='freqWiou', value=self.freq_w_iou.compute(), step=step)
        self.write_item(name='sqrtWiou', value=self.sqrt_w_iou.compute(), step=step)
        self.write_item(name='class0_recall', value=self.class_0_recall.compute(), step=step)
        self.write_item(name='class1_recall', value=self.class_1_recall.compute(), step=step)

    def _training_summary(self, epoch):
        return f'TRAIN [{epoch}] ' \
               f'Loss: {self.loss_meter.val:.3f}({self.loss_meter.avg:.3f}) | '\
               f'LR {self.optimizer.param_groups[0]["lr"]:.5f} | ' \
               f'B {self.batch_time.avg:.2f} D {self.data_time.avg:.2f}'

    def _validation_summary(self, epoch):
        return f'EVAL [{epoch}] | '\
               f'Acc: {self.accuracy.compute():.3f} | ' \
               f'Class0Rec: {self.class_0_recall.compute():.3f} | ' \
               f'Class1Rec: {self.class_1_recall.compute():.3f} | ' \
               f'mIOU: {self.mean_iou.compute():.3f} | ' \
               f'SqrtWIOU: {self.sqrt_w_iou.compute():.3f} | ' \
               f'FreqWIOU: {self.freq_w_iou.compute():.3f}'

    def _visualize_batch(self, img, label, outputs, step, vis_save_dir, vis_shape=(128, 128), predef=''):
        img = self.train_loader.dataset.denormalize(img.clone(), device=self.device)
        img = F.interpolate(img, size=vis_shape, mode='bilinear', align_corners=True)

        for i, out in enumerate(outputs):
            if out.max() == 255:
                out = out / 255.
            out = F.interpolate(out, size=vis_shape, mode='bilinear', align_corners=True)
            outputs[i] = out.cpu().detach().repeat(1, 3, 1, 1)

        if label.max() == 255:
            label = label / 255.

        label = F.interpolate(label, size=vis_shape, mode='bilinear', align_corners=True)
        vis_list = [img.cpu().detach(), label.cpu().detach().repeat(1, 3, 1, 1)] + outputs
        vis_img = torch.cat(vis_list, dim=-1)
        save_image(vis_img, str(vis_save_dir / f'{predef}_index_{step}.png'), nrow=1)
