from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F
from config import EDGES


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        pred = self._tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
      '''
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss

class CtdetLoss(torch.nn.Module):
    def __init__(self):
        super(CtdetLoss, self).__init__()
        self.L_hm = FocalLoss()
        self.L_wh =  RegL1Loss()
        self.L_off = RegL1Loss()

    def seg_loss(self, pr_dict):
        loss_seg = []
        for pr_roi, gt_roi in zip(pr_dict['pr_rois'], pr_dict['gt_rois']):
            # loss_seg.append(F.binary_cross_entropy(pr_roi, torch.from_numpy(gt_roi).to(pr_roi.device), size_average=True))
            loss_seg.append(F.binary_cross_entropy(pr_roi, gt_roi, size_average=True))
        if len(loss_seg):
            loss_seg = torch.mean(torch.stack(loss_seg, dim=0))
        else:
            loss_seg = None
        return loss_seg

    def rend_points_loss(self, pr_dict):
        loss_rend = []
        for pr_pt, gt_pt in zip(pr_dict['pr_pts_class0'],pr_dict['gt_pts_class0']):
            loss_rend.append(F.binary_cross_entropy(pr_pt, gt_pt, size_average=True))
        for pr_pt, gt_pt in zip(pr_dict['pr_pts_class1'],pr_dict['gt_pts_class1']):
            loss_rend.append(F.binary_cross_entropy(pr_pt, gt_pt, size_average=True))
        if len(loss_rend):
            loss_rend = torch.mean(torch.stack(loss_rend, dim=0))
        else:
            loss_rend = None
        return loss_rend

    def forward(self, data_dict, pr_dict):
        # detection loss
        hm_loss  = self.L_hm(pr_dict['hm'],  data_dict['hm'])
        wh_loss  = self.L_wh(pr_dict['wh'], data_dict['reg_mask'], data_dict['ind'], data_dict['wh'])
        off_loss = self.L_off(pr_dict['reg'], data_dict['reg_mask'], data_dict['ind'], data_dict['reg'])
        loss_dec = hm_loss + 0.1*wh_loss + off_loss

        # segmentation loss
        loss_seg = self.seg_loss(pr_dict)
        if loss_seg is not None:
            loss = loss_dec + 2*loss_seg
        else:
            loss = loss_dec
        # print(loss_dec)
        # print(loss_seg)
        # print(loss_rend)
        return loss


class DetectionLossAll(nn.Module):
    def __init__(self, kp_radius):
        super(DetectionLossAll, self).__init__()
        self.kp_radius = kp_radius


    def kp_map_loss(self, pr_kp, gt_kp):
        loss = F.binary_cross_entropy(pr_kp, gt_kp)
        return loss

    def short_offset_loss(self, pr_short, gt_short, gt_kp):
        loss = torch.abs(pr_short - gt_short)/self.kp_radius
        gt_2kps_map = []
        for i in range(gt_kp.shape[1]):
            gt_2kps_map.append(gt_kp[:,i,:,:])
            gt_2kps_map.append(gt_kp[:,i,:,:])

        gt_2kps_map = torch.stack(gt_2kps_map, dim=1)
        loss = loss * gt_2kps_map
        loss = torch.sum(loss)/(torch.sum(gt_2kps_map) + 1e-10)
        return loss

    def mid_offset_loss(self, pr_mid, gt_mid, gt_kp):
        loss = torch.abs(pr_mid - gt_mid)/self.kp_radius
        gt_4edge_map = []
        # bi-direction
        for i, edge in enumerate((EDGES + [edge[::-1] for edge in EDGES])):
            from_kp = edge[0]
            gt_4edge_map.extend([gt_kp[:,from_kp,:,:], gt_kp[:,from_kp,:,:]])
        gt_4edge_map = torch.stack(gt_4edge_map, dim=1)
        loss = loss * gt_4edge_map
        loss = torch.sum(loss)/(torch.sum(gt_4edge_map) + 1e-10)
        return loss

    def forward(self, prediction, groundtruth):
        pr_kp, pr_short, pr_mid = prediction
        gt_kp = groundtruth[:,:5,:,:]
        gt_short = groundtruth[:,5:5+10,:,:]
        gt_mid = groundtruth[:,5+10:,:,:]
        loss_kp = self.kp_map_loss(pr_kp, gt_kp)
        loss_short = self.short_offset_loss(pr_short, gt_short, gt_kp)
        loss_mid = self.mid_offset_loss(pr_mid, gt_mid, gt_kp)
        loss = loss_kp + loss_short + 0.25 * loss_mid
        return loss