from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from lib.models.meticulousnet import MeticulousNet
from lib.utils import load_checkpoint
from lib.base.base_dataset import BaseDataset
from lib.eval.helper import green_background


class CoarseModule(nn.Module):
    def __init__(self, checkpoint_model, input_size, device='cpu'):
        super().__init__()
        checkpoint = load_checkpoint(checkpoint_model)
        config = checkpoint['config']

        self.model = MeticulousNet(**config.arch.args)
        self.device = device
        self.model.to(device)
        self.model.load_pretrained_weights(checkpoint['state_dict'])
        self.model.eval()

        self.input_size = input_size
        self.image_transform = transforms.Compose([
            transforms.Resize(input_size, Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(BaseDataset.MEAN, BaseDataset.STD)
        ])

    @torch.no_grad()
    def forward(self, image):
        image = self.image_transform(image).unsqueeze(0).to(self.device)
        outputs, _ = self.model(image)
        mask = outputs[-1]
        return mask


def coarse_mask_demo():
    ckpt_path = Path("../../saved/checkpoints/mosl_checkpoint.pth").resolve()

    img_folder = Path('../../datasets/HRSOD_Subset/test_images')
    coarse_image_path = Path('../../saved/outputs/HRSOD_Subset/coarse/')
    coarse_mask_path = Path('../../saved/outputs/HRSOD_Subset/coarse_mask/')
    coarse_image_path.mkdir(parents=True, exist_ok=True)
    coarse_mask_path.mkdir(parents=True, exist_ok=True)

    coarse_module = CoarseModule(ckpt_path, input_size=(336, 336), device='cuda')
    for img_path in img_folder.glob("*.jpg"):
        img = Image.open(img_path).convert('RGB')
        coarse_mask = coarse_module(img)
        coarse_mask = F.interpolate(coarse_mask, size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False)
        coarse_mask = coarse_mask[0, 0]
        coarse_mask_out = Image.fromarray((coarse_mask.detach().cpu().numpy() * 255).astype('uint8'))
        coarse_mask_out.save((coarse_mask_path / img_path.name).with_suffix('.png'))
        coarse_img = green_background(img, coarse_mask[..., None].cpu().numpy())
        coarse_img.save(coarse_image_path / img_path.name)


if __name__ == '__main__':
    coarse_mask_demo()