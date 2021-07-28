from abc import abstractmethod
import numpy as np

import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    def __init__(self, width, height, mean=None, std=None, mode="train"):
        assert mode in ["train", "val"], "Not available mode! Use one of ['train', 'val']"
        self.files = []
        self.width = width
        self.height = height
        self.mode = mode
        self.mean, self.std = mean, std
        if None in [mean, std]:
            self.mean = BaseDataset.MEAN
            self.std = BaseDataset.STD
        self.mean = torch.tensor(self.mean).float()
        self.std = torch.tensor(self.std).float()
        self.to_tensor = TF.Compose([TF.ToTensor(), TF.Normalize(self.mean, self.std)])

    @abstractmethod
    def _load_data_(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = f"Dataset: {self.__class__.__name__}\n"
        fmt_str += f"    # data: {self.__len__()}\n"
        return fmt_str

    def denormalize(self, tensors, inplace=True, device=None):
        _mean = torch.as_tensor(self.mean, dtype=torch.float, device=device)[None, :, None, None]
        _std = torch.as_tensor(self.std, dtype=torch.float, device=device)[None, :, None, None]
        if not inplace:
            tensors = tensors.clone()

        tensors.mul_(_std).add_(_mean)
        return tensors

    def normalize(self, tensors, inplace=True, device=None):
        _mean = torch.as_tensor(self.mean, dtype=torch.float, device=device)[None, :, None, None]
        _std = torch.as_tensor(self.std, dtype=torch.float, device=device)[None, :, None, None]
        if not inplace:
            tensors = tensors.clone()

        tensors.sub_(_mean).div_(_std)
        return tensors

    @staticmethod
    def calc_mask_weight(target):
        target = np.array(target/255, dtype=int)
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = class_sample_count / np.sum(class_sample_count)
        samples_weight = weight[1-target]
        return samples_weight
