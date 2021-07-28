import warnings
import numpy as np
from PIL import Image
from pathlib import Path

import torch

from torchvision import transforms
import torchvision.transforms.functional as F

from lib.base import BaseDataset
from lib.data.helper import mask2numpy
from lib.data.helper import reseed
from lib.data.helper import modify_boundary
from lib.data.joint_transforms import (
    Compose,
    JointResize,
    RandomHorizontallyFlip,
    RandomRotate,
)


warnings.filterwarnings('ignore', message=r'.*Corrupt EXIF data\.  Expecting to read .+ bytes but only got .+\.')


class MosLDataset(BaseDataset):
    def __init__(self, root, img_mask_folders, width, height, mean, std, mode, **kwargs):
        super().__init__(width, height, mean, std, mode)
        self.root = Path(root).resolve()
        self.img_mask_folders = img_mask_folders
        self.files = self.get_files()

        self.input_size = (height, width)
        if self.mode == 'train':
            self.joint_transform = Compose([
                JointResize((width, height)),
                RandomHorizontallyFlip(),
                RandomRotate(30)
            ])
            img_transform = [transforms.ColorJitter(0.1, 0.1, 0.1)]
            self.mask_transform = transforms.ToTensor()
        else:
            img_transform = [transforms.Resize(self.input_size, interpolation=Image.BILINEAR)]
            self.mask_transform = transforms.Compose([
                transforms.Resize(self.input_size, interpolation=Image.NEAREST),
                transforms.ToTensor()
                ]
            )
        self.img_transform = transforms.Compose(
            [
                *img_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def get_files(self):
        filenames = []
        for img_folder, mask_folder in self.img_mask_folders:
            img_folder = self.root / img_folder
            mask_folder = self.root / mask_folder
            for img_path in sorted(img_folder.glob("*.*")):
                mask_path = (mask_folder / img_path.name).with_suffix('.png')
                filenames.append((str(img_path), str(mask_path)))
        return filenames

    def _load_data_(self, index):
        image_path, mask_path = self.files[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')
        return image, mask

    def __getitem__(self, index):
        img, mask = self._load_data_(index)
        assert img.size == mask.size, f'Failed: {self.files[index]}'
        if self.mode == 'train':
            img, mask = self.joint_transform(img, mask)
            img = self.img_transform(img)
            mask = self.mask_transform(mask)
        else:
            img = self.img_transform(img)
            mask = self.mask_transform(mask)

        mask[mask < 0.5] = 0.0
        mask[mask >= 0.5] = 1.0
        # mask = torch.from_numpy(mask).unsqueeze(0).float()
        return {'image': img, 'mask': mask}


class MosHDataset(BaseDataset):
    def __init__(self, root, img_mask_folders, width, height, mean, std, mode, **kwargs):
        super().__init__(width, height, mean, std, mode)
        self.root = Path(root).resolve()
        self.img_mask_folders = img_mask_folders
        self.files = self.get_files()

        self.input_size = [height, width]

        if self.mode == 'train':
            self.bilinear_dual_transform = transforms.Compose([
                transforms.RandomCrop((224, 224), pad_if_needed=True),
                transforms.RandomHorizontalFlip()
            ])

            self.bilinear_dual_transform_im = transforms.Compose([
                transforms.RandomCrop((224, 224), pad_if_needed=True),
                transforms.RandomHorizontalFlip()
            ])

            self.im_transform = transforms.Compose([
                transforms.ColorJitter(0.2, 0.05, 0.05, 0),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                ),
            ])
        else:
            self.bilinear_dual_transform = transforms.Compose([
                transforms.Resize(896, interpolation=Image.NEAREST),
                transforms.CenterCrop(896),
            ])

            self.bilinear_dual_transform_im = transforms.Compose([
                transforms.Resize(896, interpolation=Image.BILINEAR),
                transforms.CenterCrop(896),
            ])

            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                ),
            ])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_files(self):
        filenames = []
        for img_folder, mask_folder in self.img_mask_folders:
            img_folder = self.root / img_folder
            mask_folder = self.root / mask_folder
            for mask_path in sorted(mask_folder.glob("*.png")):
                img_path = (img_folder / mask_path.name).with_suffix('.jpg')
                filenames.append((str(img_path), str(mask_path)))
        return filenames

    def _load_data_(self, index):
        image_path, mask_path = self.files[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def __getitem__(self, index):
        im, gt = self._load_data_(index)
        seed = np.random.randint(2147483647)

        reseed(seed)
        im = self.bilinear_dual_transform_im(im)

        reseed(seed)
        gt = self.bilinear_dual_transform(gt)

        iou_max = 1.0
        iou_min = 0.7
        iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
        seg = modify_boundary((np.array(gt)>0.5).astype('uint8')*255, iou_target=iou_target)

        im = self.im_transform(im)
        gt = self.gt_transform(gt)
        seg = self.seg_transform(seg)
        return {'image': im, 'mask': gt, 'pert_mask': seg}

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    dataset = MosHDataset()