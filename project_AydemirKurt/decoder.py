import torch.nn.functional as F
import numpy as np
import torch

class DecDecoder(object):
    def __init__(self, K, conf_thresh):
        self.K = K
        self.conf_thresh = conf_thresh

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K)
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

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

    def ctdet_decode(self, pr_dict):
        hm = pr_dict['hm']
        reg = pr_dict['reg']
        wh = pr_dict['wh']

        batch, cat, height, width = hm.size()
        heat = self._nms(hm)

        scores, inds, clses, ys, xs = self._topk(heat)
        if reg is not None:
            reg = self._tranpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.K, 2)
            xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.K, 1) + 0.5
            ys = ys.view(batch, self.K, 1) + 0.5

        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 2)
        clses = clses.view(batch, self.K, 1).float()
        scores = scores.view(batch, self.K, 1)
        bboxes = torch.cat([xs,
                            ys,
                            wh[..., 0:1],
                            wh[..., 1:2]], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

        index = (scores>self.conf_thresh).squeeze(0).squeeze(1)
        detections = detections[:,index,:]

        return detections.squeeze(0).data.cpu().numpy()