from dataloader.trsfrms import must_transform
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as trs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#import cv2
class CUB(Dataset):
    def __init__(self, root, mode='train', transform=None):
        super(CUB, self).__init__()
        self.transform = transform
        dataset = ImageFolder(root=root)
        self.img_path = []
        self.target = []
        if mode == 'train':
            for (path, label) in dataset.imgs:
                if label < 50:
                    self.img_path.append(path)
                    self.target.append(label)
        elif mode == 'val':
            for (path, label) in dataset.imgs:
                if 49 < label < 100:
                    self.img_path.append(path)
                    self.target.append(label)
        elif mode == 'trainval':
            for (path, label) in dataset.imgs:
                if  label < 100:
                    self.img_path.append(path)
                    self.target.append(label)
        elif mode == 'test':
            for (path, label) in dataset.imgs:
                if label > 99:
                    self.img_path.append(path)
                    self.target.append(label)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_path[idx]).convert('RGB'))
        lbl = np.array(self.target[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


if __name__ == '__main__':
    img_mean = np.array([104, 117, 128]).reshape(1, 1, 3)
    transforms_tr = trs.Compose([must_transform(), trs.RandomResizedCrop(224), trs.RandomHorizontalFlip()])
    transforms_test = trs.Compose([must_transform(), trs.Resize(256), trs.CenterCrop(224)])
    cub_train = CUB(root="./data/CUB_200_2011/images", mode='trainval', transform=transforms_tr)
    cub_val = CUB(root= "./data/CUB_200_2011/images",  mode='val', transform=transforms_test)
    cub_test = CUB(root= "./data/CUB_200_2011/images", mode='test', transform=None)
    img, lbl = cub_train[1]
    img_test, lbl_test = cub_val[2]
    unnormalized = (img_test.numpy().transpose(1, 2, 0) + img_mean)[:, :, ::-1].astype(np.uint8)
    plt.imshow(unnormalized)
    img = Image.fromarray(unnormalized)
    img.show()


