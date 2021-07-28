import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from model.rpn.rpn import _RPN
from model.roi.roi_pool import ROIPool
from model.roi.roi_align import ROIAlign
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from utils.net_utils import smooth_l1_loss
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.net_utils import grad_reverse

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, class_agnostic, out_depth):
        super().__init__()
        self.n_classes = num_classes
        self.class_agnostic = class_agnostic
        self.regression_weights = (10., 10., 5., 5.)
        
        self.RCNN_rpn = _RPN(out_depth)

        self.RCNN_proposal_target = _ProposalTargetLayer(self.regression_weights)
        
        if cfg.GENERAL.POOLING_MODE == 'pool':
            self.RCNN_roi_layer = ROIPool(1.0/16.0, cfg.GENERAL.POOLING_SIZE)
        elif cfg.GENERAL.POOLING_MODE == 'align':
            self.RCNN_roi_layer = ROIAlign(1.0/16.0, cfg.GENERAL.POOLING_SIZE, 0, True)
        else:
            raise ValueError('There is no implementation for "{}" ROI layer'
                             .format(cfg.GENERAL.POOLING_MODE))

        self.RCNN_roi_aligncat1 = ROIAlign(1.0/16.0, cfg.GENERAL.POOLING_SIZE, 0, True)
        self.RCNN_roi_aligncat2 = ROIAlign(1.0/16.0, cfg.GENERAL.POOLING_SIZE, 0, True)
        self.RCNN_roi_aligncat3 = ROIAlign(1.0/16.0, cfg.GENERAL.POOLING_SIZE, 0, True)
        self.RCNN_roi_aligncat4 = ROIAlign(1.0/16.0, cfg.GENERAL.POOLING_SIZE, 0, True)
        
    def forward(self, im_data, im_info, gt_boxes,target = False):
        if self.training:
            assert gt_boxes is not None
    
        batch_size = im_data.size(0)
        
        base_feature1 = self.RCNN_base1(im_data)
        base_feature2 = self.RCNN_base2(base_feature1)
        base_feature3 = self.RCNN_base3(base_feature2)
        base_feature4 = self.RCNN_base4(base_feature3)
        base_feature = self.RCNN_base5(base_feature4)
        
        D1_out = self.D1_fcnn(grad_reverse(base_feature1))
        D2_out = self.D2_fcnn(grad_reverse(base_feature2))
        D3_out = self.D3_fcnn(grad_reverse(base_feature3))
        D4_out = self.D4_fcnn(grad_reverse(base_feature4)) 

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feature, im_info, gt_boxes)
        
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_target = roi_data
            
            rois_label = rois_label.view(-1).long()
            rois_target = rois_target.view(-1, rois_target.size(2))
        
        pooled_feat = self.RCNN_roi_layer(base_feature, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._feed_pooled_feature_to_top(pooled_feat)
        
        if self.training:
            pass_feat = pooled_feat.view(batch_size*256,4096)
            caia_out = self.caia_clsf(grad_reverse(pass_feat))
        else:
            pass_feat = pooled_feat
            caia_out = self.caia_clsf(grad_reverse(pass_feat))

        
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            gather_idx = rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, gather_idx)
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            pos_idx = torch.nonzero(rois_label > 0).view(-1)
            RCNN_loss_bbox = smooth_l1_loss(bbox_pred[pos_idx],
                                            rois_target[pos_idx],
                                            size_average=False)
            RCNN_loss_bbox = RCNN_loss_bbox / rois_label.numel()
        else:
            cls_score = F.softmax(cls_score, 1)
            cls_score = cls_score.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_transform_inv(rois[:, :, 1:5], bbox_pred, self.regression_weights)
            bbox_pred = clip_boxes(bbox_pred, im_info, batch_size)

        return cls_score, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, D1_out, D2_out, D3_out, D4_out,caia_out

    def _prepare_pooled_feature(self, pooled_feature):
        raise NotImplementedError
        
    def _init_modules(self):
        raise NotImplementedError
        
    def _init_weights(self):
        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01)
        normal_init(self.RCNN_cls_score, 0, 0.01)
        normal_init(self.RCNN_bbox_pred, 0, 0.001)

    def init(self):
        self._init_modules()
        self._init_weights()
