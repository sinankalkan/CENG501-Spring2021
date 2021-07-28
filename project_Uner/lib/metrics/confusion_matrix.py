"""Adapted from PyTorch-Ignite."""
import numpy as np

from lib.metrics.base_metric import BaseMetric, MetricsLambda, NotComputableError


class ConfusionMatrix(BaseMetric):
    def __init__(self, num_classes, average=None):
        assert num_classes > 0
        assert average is None or average in {'samples', 'recall', 'precision'}
        self.num_classes = num_classes
        self.average = average
        super().__init__()

    def reset(self):
        self.num_examples = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        assert pred.shape == target.shape
        self.num_examples += pred.shape[0]
        pred, target = pred.reshape(-1), target.reshape(-1)

        mask = (target >= 0) & (target < self.num_classes)
        indices = self.num_classes * target[mask] + pred[mask]
        bins = np.bincount(indices, minlength=self.num_classes ** 2)
        cm = bins.reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += cm

    def compute(self):
        if self.num_examples == 0:
            raise NotComputableError("Confusion matrix must have at least one example before it can be computed.")
        if self.average is not None:
            self.confusion_matrix = self.confusion_matrix.astype(float)
            if self.average == 'samples':
                return self.confusion_matrix / self.num_examples
            elif self.average == 'recall':
                return self.confusion_matrix / (self.confusion_matrix.sum(axis=1)[..., np.newaxis] + 1e-15)
            elif self.average == 'precision':
                return self.confusion_matrix / (self.confusion_matrix.sum(axis=0) + 1e-15)
        return self.confusion_matrix


def IoU(cm, ignore_index=None):
    iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal() + 1e-15)
    if ignore_index is None:
        return iou
    else:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError(
                    f"ignore_index {ignore_index} is larger than the length of IoU vector {len(iou_vector)}"
                )
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)


def mIoU(cm, ignore_index=None):
    return IoU(cm, ignore_index=ignore_index).mean()


def FreqWIoU(cm, ignore_index=None):
    iou = IoU(cm, ignore_index=ignore_index)
    freq = cm.sum(axis=1) / (cm.sum() + 1e-15)
    return (freq * iou).sum()


def SqrtWIoU(cm, ignore_index=None):
    iou = IoU(cm, ignore_index=ignore_index)
    freq_sqrt = MetricsLambda(np.sqrt, cm.sum(axis=1))
    freq_sqrt = freq_sqrt / (freq_sqrt.sum() + 1e-15)
    return (freq_sqrt * iou).sum()


def Accuracy(cm):
    return cm.diagonal().sum() / (cm.sum() + 1e-15)


def Recall(cm, average=True):
    recall = cm.diagonal() / (cm.sum(axis=1) + 1e-15)
    if average:
        return recall.mean()
    return recall


if __name__ == '__main__':
    target = np.random.randint(0, 3, (1, 30, 30))
    pred = np.random.randint(0, 3, (1, 30, 30))
    c = ConfusionMatrix(3)
    d = ((c + 2) ** 0.5).mean()

    iou = IoU(c)
    c.update(pred, target)
    print(iou.compute())
