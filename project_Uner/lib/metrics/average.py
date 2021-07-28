import numpy as np

from .base_metric import BaseMetric, NotComputableError


class Average(BaseMetric):
    """Adapted from PyTorch-Ignite."""

    def __init__(self):
        super().__init__()

    def reset(self):
        self.accumulator = np.array(0.0)
        self.num_examples = 0

    def update(self, value, pre_avgd_num_ex=None):
        if pre_avgd_num_ex is not None:
            self.accumulator = self.accumulator + value * pre_avgd_num_ex
            self.num_examples += pre_avgd_num_ex
        elif isinstance(value, np.ndarray) and value.ndim > 0:
            self.accumulator = self.accumulator + value.sum(axis=0)
            self.num_examples += value.shape[0]
        else:
            self.accumulator += value
            self.num_examples += 1

    def compute(self):
        if self.num_examples == 0:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )

        return self.accumulator / self.num_examples
