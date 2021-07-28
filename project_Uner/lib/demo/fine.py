import natsort
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from lib.models.meticulousnet import MeticulousNet
from lib.utils import load_checkpoint
from lib.base.base_dataset import BaseDataset

from natsort import natsorted, ns


class FineModule(nn.Module):
    def __init__(self, low_ckpt, high_ckpt, low_size, high_size, device='cpu'):
        super().__init__()
        low_ckpt = load_checkpoint(low_ckpt)
        high_ckpt = load_checkpoint(high_ckpt)

        self.low_model = MeticulousNet(**low_ckpt['config'].arch.args)
        self.high_model = MeticulousNet(**high_ckpt['config'].arch.args)

        self.device = device

        self.low_model.to(device)
        self.low_model.load_pretrained_weights(low_ckpt['state_dict'])
        self.low_model.eval()

        self.high_model.to(device)
        self.high_model.load_pretrained_weights(high_ckpt['state_dict'])
        self.high_model.eval()

        self.low_resize = transforms.Resize(low_size)
        self.high_resize = transforms.Resize(high_size)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(BaseDataset.MEAN, BaseDataset.STD)
        ])
        self.pad = nn.ZeroPad2d(16)

    @torch.no_grad()
    def low_forward(self, image):
        image = self.low_resize(image)
        image = self.image_transform(image).unsqueeze(0).to(self.device)
        outputs, _ = self.low_model(image)
        mask = outputs[-1]
        return mask

    @torch.no_grad()
    def forward(self, image, step_size=224):
        orig_h, orig_w = image.size[1], image.size[0]
        coarse_mask = self.low_forward(image)
        # image = self.high_resize(image)
        img = self.image_transform(image).unsqueeze(0).to(self.device)
        _, _, h, w = img.shape
        coarse_mask = F.interpolate(coarse_mask, size=(h, w), mode='bilinear', align_corners=False)
        coarse_mask = TF.normalize(coarse_mask, mean=[0.5], std=[0.5])
        coarse_mask[coarse_mask >= 0.0] = 1.0
        coarse_mask[coarse_mask < 0.0] = -1.0

        fine = torch.zeros_like(coarse_mask)
        for x_idx in range((w) // step_size):
            for y_idx in range((h) // step_size):

                start_x = x_idx * step_size
                start_y = y_idx * step_size
                end_x = start_x + step_size
                end_y = start_y + step_size

                # Bound x/y range
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(w-1, end_x)
                end_y = min(h-1, end_y)

                # Take crop
                img_part = img[:, :, start_y:end_y, start_x:end_x]
                mask_part = coarse_mask[:, :, start_y:end_y, start_x:end_x]
                inp = torch.cat((img_part, mask_part), dim=1)
                old_shape = (inp.shape[2], inp.shape[3])
                if inp.shape[2] % 8 != 0 or inp.shape[3] % 8 != 0:
                    new_w = (inp.shape[2] // 16) * 16
                    new_h = (inp.shape[3] // 16) * 16
                    inp = F.interpolate(inp, size=(new_w, new_h), mode='bilinear', align_corners=True)
                preds, _ = self.high_model(inp)
                out_part = preds[-1]
                if (out_part.shape[2], out_part.shape[3]) != old_shape:
                    out_part = F.interpolate(out_part, size=old_shape, mode='bilinear', align_corners=True)
                fine[:, :, start_y:end_y, start_x:end_x] += out_part
        fine = F.interpolate(fine, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return fine


if __name__ == '__main__':
    low_ckpt_path = Path("../../saved/MeticulousNet_L/07-02_18-06/checkpoints/checkpoint-epoch360.pth").resolve()
    high_ckpt_path = Path("../../saved/MeticulousNet_H/07-10_19-10/checkpoints/checkpoint-epoch40.pth").resolve()
    fine_outputs = Path('../../saved/outputs/fine/')
    fine_outputs.mkdir(parents=True, exist_ok=True)

    img_folder = Path('../../datasets/test/image')
    fine_module = FineModule(low_ckpt_path, high_ckpt_path, low_size=(336, 336), high_size=(896, 896), device='cuda')
    for img_path in natsorted(img_folder.glob("*.jpg"), alg=ns.PATH):
        img = Image.open(img_path).convert('RGB')
        fine_mask = fine_module(img, step_size=336)
        fine_mask = fine_mask[0, 0]
        fine_mask[fine_mask > 0.5] = 1.0
        fine_img = (img * fine_mask[..., None].cpu().numpy()).astype("uint8")
        fine_img = Image.fromarray(fine_img)
        fine_img.save(fine_outputs / img_path.name)
