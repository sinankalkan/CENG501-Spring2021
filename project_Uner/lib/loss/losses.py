import torch
import torch.nn as nn
import torch.nn.functional as F


def jaccard(intersection, union, eps=1e-15):
    return (intersection) / (union - intersection + eps)


def dice(intersection, union, eps=1e-15, smooth=1.):
    return (2. * intersection + smooth) / (union + smooth + eps)


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss(reduction="none")

    def forward(self, input, target, weight):
        loss = self.criterion(input, target)
        loss = loss * weight
        loss = torch.mean(loss)
        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss(reduction="none")

    def forward(self, input, target, weight):
        loss = self.criterion(input, target)
        loss = loss * weight
        loss = loss.mean()
        return loss


class BCESoftJaccardDice(nn.Module):
    def __init__(self, bce_weight=0.5, mode="dice", eps=1e-7, smooth=1.):
        super(BCESoftJaccardDice, self).__init__()
        self.bce_weight = bce_weight
        self.eps = eps
        self.mode = mode
        self.smooth = smooth
        self.wbce = WeightedBCELoss()

    def forward(self, input, target, weight=None):
        if weight is None:
            loss = self.bce_weight * F.binary_cross_entropy(input, target)
        else:
            loss = self.bce_weight * self.wbce(input, target, weight)

        if self.bce_weight < 1.:
            targets = (target == 1).float()
            intersection = (input * targets).sum()
            union = input.sum() + targets.sum()
            if self.mode == "dice":
                score = dice(intersection, union, self.eps, self.smooth)
            elif self.mode == "jaccard":
                score = jaccard(intersection, union, self.eps)
            loss -= (1 - self.bce_weight) * torch.log(score)
        return loss


def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """

    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    # edges= torch.cat([edge,edge,edge,edge,edge,edge,edge], dim=0)
    # print(preds.shape, edges.shape)
    b, c, h, w = edges.shape

    # Shape: [b,].
    num_pos = torch.sum(edges, dim=[1, 2, 3], keepdim=True).float()
    # print("pos", num_pos.shape)

    num_neg = c * h * w - num_pos
    weight = torch.zeros_like(edges)
    weight.masked_scatter_(edges > 0.1, torch.ones_like(edges) * 2 * num_neg / (num_pos + num_neg))
    weight.masked_scatter_(edges <= 0.1, torch.ones_like(edges) * num_pos / (num_pos + num_neg))

    losses = F.binary_cross_entropy(preds.float(),
                                    edges.float(),
                                    weight=weight,
                                    reduction='none')
    loss = torch.mean(losses)
    return loss