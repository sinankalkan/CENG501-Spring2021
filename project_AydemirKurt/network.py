import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet
from .layers import CombinationModule
import numpy as np
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(Network, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet50(pretrained=pretrained)
        channels = [3, 64, 256, 512, 1024, 2048]

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)

        self.seg_combine = nn.ModuleList([CombinationModule(64, 64, instance_norm=True),
                                          CombinationModule(256, 64, instance_norm=True),
                                          CombinationModule(512, 256, instance_norm=True),
                                          CombinationModule(1024, 512, instance_norm=True)])

        self.c0_conv = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.seg_head = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 1, 3, 1, 1))

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv,
                                             kernel_size=7, padding=7//2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes,
                                             kernel_size=7, padding=7 // 2, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv,
                                             kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

        self.point_head0 = nn.Sequential(nn.Conv1d(64, 1, 1), nn.Sigmoid())
        self.point_head1 = nn.Sequential(nn.Conv1d(128, 1, 1), nn.Sigmoid())

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward_dec(self, x):
        """
        torch.Size([2, 3, 512, 512])      c0
        torch.Size([2, 64, 256, 256])     c1
        torch.Size([2, 256, 128, 128])    c2
        torch.Size([2, 512, 64, 64])      c3
        torch.Size([2, 1024, 32, 32])     c4
        torch.Size([2, 2048, 16, 16])     c5
        """
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
        dec_dict['hm'] = torch.sigmoid(dec_dict['hm'])
        dec_dict['feat'] = [x[0], x[1], c2_combine, c3_combine, c4_combine]
        return dec_dict


    def point_sample(self, image, grids, **kwargs):
        # image: (N x C x H x W)
        # grids: (N, P, 2) OR (N, H, W, 2)
        if grids.dim()==3:
            grids = grids.unsqueeze(2)  # (N, P, 1, 2)
        output = F.grid_sample(image, 2.0*grids-1.0, align_corners=True)
        return output

    def rend_grids_sampling_train(self, x, N, k=3, beta=0.75):
        mask = x.clone().detach()
        mask = (-1)*abs(mask*2-1)  # uncertain map
        # step1: create random sample points, num_pts = k*N
        sample_grids = torch.rand(size=(k*N, 2), device=x.device)  # num x 2, range [0, 1]
        sampled_pts = self.point_sample(mask.unsqueeze(0).unsqueeze(0),
                                        sample_grids.unsqueeze(0), align_corners=False).squeeze(-1)   # [1, 1, P]
        # step2: extract pts at the uncertainty map, num_pts = beta*N
        _, index = torch.topk(sampled_pts, k=int(beta*N), dim=-1)
        # step3: shift index to batch, no need here
        # step4: replace sample_grids with boundary floating pts
        rend_grids = sample_grids[index.view(-1), :]  # num x 2
        # step5: coverage points, no need here:
        # coverage = torch.rand(size=(N - int(beta * N), 2), device=x.device)
        # out = torch.cat([rend_grids, coverage], 0)
        return rend_grids.unsqueeze(0)


    def seg_branch(self, rois):
        for i in range(len(rois) - 2, -1, -1):
            rois[i] = self.seg_combine[i](rois[i + 1], rois[i])
        x = self.seg_head(rois[0])
        x = torch.sigmoid(x).squeeze(0).squeeze(0)
        return x, rois

    def get_grids_numpy(self, bbox):
        cen_x, cen_y, w, h = bbox
        x1 = cen_x - w / 2
        y1 = cen_y - h / 2
        x2 = cen_x + w / 2
        y2 = cen_y + h / 2
        x = np.linspace(x1, x2, num=int(w))
        y = np.linspace(y1, y2, num=int(h))
        grids = np.stack([np.repeat(x[np.newaxis, :], y.shape[0], axis=0),
                          np.repeat(y[:, np.newaxis], x.shape[0], axis=1)], axis=2)
        grids = grids[np.newaxis, :, :, :]
        return np.asarray(grids, np.float32)

    def bilinear_sampling(self, feat, bbox):
        bbox[2] *= 1.1
        bbox[3] *= 1.1
        cenx, ceny, w, h = bbox
        if w<2 or h<2:
            return None
        b, c, IH, IW = feat.shape
        grids = self.get_grids_numpy(bbox)  # 1 x h x w x 2
        # normalization for pyTorch metric
        grids[:, :, :, 0] = (grids[:, :, :, 0] / (IW - 1)) * 2 - 1
        grids[:, :, :, 1] = (grids[:, :, :, 1] / (IH - 1)) * 2 - 1
        grids = torch.from_numpy(grids)#.expand(size=(b, h, w, 2))
        grids = grids.to(feat.device)
        roi = F.grid_sample(input=feat, grid=grids, align_corners=True)
        return roi

    def crop_rois(self, x, bbox, b):
        """
        x: b x c x IH x IW
        bbox: 1 x 4 [cen_x, cen_y, w, h]
        b: batch index
        """
        roi_lists = []
        for i in range(5):
            s = float(2**(i))
            roi_s = self.bilinear_sampling(x[i][b:b+1,:,:,:], bbox/s)
            if roi_s is None:
                break
            roi_lists.append(roi_s)
        return roi_lists


    def forward_seg(self, data_dict, pr_dict):
        # initialization ---------------------------------
        gt_bboxes = data_dict['gt_bboxes']
        gt_rois = data_dict['gt_rois']
        feat = pr_dict['feat']
        pr_dict['gt_rois'] = []
        pr_dict['pr_rois'] = []
        pr_dict['pr_pts_class0'] = []
        pr_dict['pr_pts_class1'] = []
        pr_dict['gt_pts_class0'] = []
        pr_dict['gt_pts_class1'] = []
        #--------------------------------------------------
        feat[0] = self.c0_conv(feat[0])
        for b in range(feat[0].shape[0]):
            for gt_bbox, gt_roi in zip(gt_bboxes[b], gt_rois[b]):
                cropped_rois = self.crop_rois(feat, gt_bbox, b)
                if len(cropped_rois) == 0:
                    continue
                pr_roi, feat_rois = self.seg_branch(cropped_rois)
                gt_roi = torch.from_numpy(gt_roi).to(pr_roi.device).unsqueeze(0).unsqueeze(0)
                pr_dict['gt_rois'].append(gt_roi.squeeze(0).squeeze(0))
                pr_dict['pr_rois'].append(pr_roi)
        return pr_dict


    def forward_seg_test(self, data_dict, pr_dict):
        pr_bboxes = data_dict['pr_bboxes']
        feat = pr_dict['feat']
        pr_dict['pr_bboxes'] = []
        pr_dict['pr_rois'] = []
        feat[0] = self.c0_conv(feat[0])
        for pr_bbox in pr_bboxes:
            cropped_rois = self.crop_rois(feat, pr_bbox[:4], 0)
            if len(cropped_rois) == 0:
                continue
            pr_roi, feat_rois  = self.seg_branch(cropped_rois)
            pr_dict['pr_bboxes'].append(pr_bbox)
            pr_dict['pr_rois'].append(pr_roi)
        return pr_dict



    def rend_grids_sampling_test(self, x, N, k=1, beta=0.5):
        mask = x.clone().detach()
        IH, IW = mask.shape
        mask = (-1)*abs(mask*2-1)  # uncertainty map
        sample_grids = torch.rand(size=(k*N, 2), device=x.device)  # num x 2, range [0, 1]
        sampled_pts = self.point_sample(mask.unsqueeze(0).unsqueeze(0),
                                        sample_grids.unsqueeze(0), 
                                        align_corners=True).squeeze(-1)   # [1, 1, P]                                        
        _, index = torch.topk(sampled_pts, k=int(beta*N), dim=-1)        
        rend_grids = sample_grids[index.view(-1), :]  # num x 2    

        rend_grids[:,0] = rend_grids[:,0] * (IW - 1)
        rend_grids[:,1] = rend_grids[:,1] * (IH - 1)

        return index, rend_grids


    def forward(self, data_dict):
        pr_dict = self.forward_dec(data_dict['image'])
        pr_dict = self.forward_seg(data_dict, pr_dict)
        return pr_dict
