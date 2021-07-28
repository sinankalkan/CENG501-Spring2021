import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.encoder import Encoder
from lib.base.base_model import BaseModel
from lib.models.utils import point_sample, ConvBNReLU


# Modules in green(HierPR) share weights globally
class HierPRBlock(nn.Module):
    def __init__(self, enc_feat_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(enc_feat_channels+1, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, uncertainty_map, prev_mask, enc_f):
        # The Basic PointRend (PR) module [11] is a three-layer perceptron.
        # At each point to perform refinements, the input is the concatenation of the feature vector from pre-specified
        # feature activations and the prediction vector from coarse prediction map at that spatial location.
        # The potential resolution inconsistency is dealt with by interpolation.

        # While in the refining branch, all the HierPR modules
        # are only connected to the first block, using the same features to refine all the predictions.

        # At both resolutions, we set HierPR to refine 10% pixels in the feature maps.
        # warning: it probably refers to deconv feature maps but implemented as if ucmap is refered
        assert uncertainty_map.dim() == 4, "ucmap must be N(Batch)CHW"
        B, C, H, W = prev_mask.shape
        K = (H * W) // 10

        H_step, W_step = 1 / H, 1 / W
        with torch.no_grad():
            _, idx = uncertainty_map.view(B, -1).topk(K, dim=1, largest=False)
            if self.training:
                coverage = torch.randint(0, (H*W), (B, K//4), device=uncertainty_map.device)
                idx = torch.cat((idx, coverage), dim=1)
                K += K//4
            points = torch.zeros(B, K, 2, dtype=torch.float, device=uncertainty_map.device)
            points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
            points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
            points = points.detach()

        coarse = point_sample(prev_mask, points, align_corners=False)
        fine = point_sample(enc_f, points, align_corners=False)

        refined_points = torch.cat([coarse, fine], dim=1)
        refined_points = self.mlp(refined_points)
        point_indices = idx.unsqueeze(1).expand(-1, C, -1)
        prev_mask = prev_mask.reshape(B, C, H * W)
        prev_mask = torch.scatter(prev_mask, 2, point_indices, refined_points)
        prev_mask = prev_mask.view(B, C, H, W)
        return prev_mask


# those in blue share weights when vertically aligned
class HierPRLayer(nn.Module):
    def __init__(self, output_stride, in_channels, hierpr):
        super().__init__()
        self.upscale_factor = output_stride // 2
        self.coarse_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.hierpr = hierpr

    # HierPR block needs three inputs:
    # pre-specified feature maps from the encoder (enc_f)
    # feature maps from the deconvolutional layers (dec_f)
    # predictions from previous HierPR block (prev_mask)
    def forward(self, dec_f, enc_f, prev_mask=None):
        coarse_mask = self.coarse_head(dec_f)
        coarse_mask = F.interpolate(coarse_mask, scale_factor=2, mode='bilinear', align_corners=False)
        uncertainty_map = (coarse_mask - 0.5).abs()  # distance to 0.5
        if prev_mask is not None:
            prev_mask = F.interpolate(prev_mask, scale_factor=2, mode='bilinear', align_corners=False)
            coarse_mask += prev_mask
            coarse_mask /= 2.
        refined_mask = self.hierpr(uncertainty_map, coarse_mask, enc_f)
        out_mask = F.interpolate(refined_mask, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        return refined_mask, uncertainty_map, out_mask


class MaskEncodingLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels+2, in_channels+2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels+2),
            nn.ReLU(),
            nn.Conv2d(in_channels+2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x, temp_mask1, temp_mask2):
        masks = torch.cat((temp_mask1, temp_mask2), dim=1)
        h, w = x.size(2), x.size(3)
        masks = F.interpolate(input=masks, size=(h, w), mode='bilinear', align_corners=False)
        f = torch.cat((x, masks), dim=1)
        return self.conv(f)


# ----------- PSPNet Modules -----------
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in
                      self.stages]
        priors = set_priors + [feats]
        return self.bottleneck(torch.cat(priors, 1))


# Recover branch of the decoder
# Similar to PSPNet decoder
# TODO: try to replace upsample with the deconvolution
class PSPUpsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(x_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x, up):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)

        p = self.conv(torch.cat([x, up], 1))
        sc = self.shortcut(x)

        p = p + sc

        p2 = self.conv2(p)

        return self.relu(p + p2)


class MeticulousDecoder(nn.Module):
    def __init__(self, enc_stride2_channels, enc_stride4_channels, enc_stride8_channels):
        super(MeticulousDecoder, self).__init__()
        self.hierpr = HierPRBlock(enc_stride2_channels)
        self.mask_encoding_layer = MaskEncodingLayer(256)
        self.psp_f8 = PSPModule(enc_stride8_channels, 256)

        # recover branch
        self.recover_f8 = PSPUpsample(256, 256+enc_stride4_channels, 128)
        self.recover_f4 = PSPUpsample(128, 128+enc_stride2_channels, 64)

        # refine branch
        self.refine_f8 = HierPRLayer(8, 256, self.hierpr)
        self.refine_f4 = HierPRLayer(4, 128, self.hierpr)
        self.refine_f2 = HierPRLayer(2, 64, self.hierpr)

        # stabilize hierpr layer
        self.init_hierpr = HierPRLayer(8, enc_stride8_channels, self.hierpr)

    def forward(self, x):
        enc_f2, enc_f4, enc_f8 = x

        s8_0_mask, s8_0_uc, s8_0_out = self.init_hierpr(enc_f8, enc_f=enc_f2, prev_mask=None)

        # cycle 1
        dec_f8_1 = self.psp_f8(enc_f8)
        s8_1_mask, s8_1_uc, s8_1_out = self.refine_f8(dec_f8_1, enc_f=enc_f2, prev_mask=None)

        # cycle 2
        dec_f8_2 = self.mask_encoding_layer(dec_f8_1, s8_0_out, s8_1_out)
        s8_2_mask, s8_2_uc, s8_2_out = self.refine_f8(dec_f8_2, enc_f=enc_f2, prev_mask=None)
        dec_f4_2 = self.recover_f8(dec_f8_2, enc_f4)

        s4_2_mask, s4_2_uc, s4_2_out = self.refine_f4(dec_f4_2, enc_f=enc_f2, prev_mask=s8_2_mask)
        s8_4_2_out = (s8_2_out + s4_2_out) / 2.

        # cycle 3
        dec_f8_3 = self.mask_encoding_layer(dec_f8_1, s8_0_out, s8_4_2_out)
        s8_3_mask, s8_3_uc, s8_3_out = self.refine_f8(dec_f8_3, enc_f=enc_f2, prev_mask=None)
        dec_f4_3 = self.recover_f8(dec_f8_3, enc_f4)

        s4_3_mask, s4_3_uc, s4_3_out = self.refine_f4(dec_f4_3, enc_f=enc_f2, prev_mask=s8_3_mask)
        dec_f2_3 = self.recover_f4(dec_f4_3, enc_f2)

        s2_3_mask, s2_3_uc, s2_3_out = self.refine_f2(dec_f2_3, enc_f=enc_f2, prev_mask=s4_3_mask)
        return [
            [s8_0_out, s8_1_out, s8_2_out, s8_3_out, s4_2_out, s4_3_out, s2_3_out],
            [s8_0_uc, s8_1_uc, s8_2_uc, s8_3_uc, s4_2_uc, s4_3_uc, s2_3_uc],
        ]


class MeticulousNet(BaseModel):
    def __init__(self, in_channels=3, backbone='resnet50', pretrained_encoder=True):
        super(MeticulousNet, self).__init__()
        self.encoder = Encoder(in_channels, backbone=backbone, pretrained=pretrained_encoder)
        self.decoder = MeticulousDecoder(self.encoder.stride2_features,
                                         self.encoder.stride4_features,
                                         self.encoder.stride8_features)

    def forward(self, x):
        enc_feature = self.encoder(x)
        return self.decoder(enc_feature)