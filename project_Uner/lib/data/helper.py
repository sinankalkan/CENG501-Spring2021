import numpy as np
from imgaug import augmenters as iaa
import cv2
import random
import torch
import math

from lib.data.de_transform import perturb_seg

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def mask2numpy(mask):
    mask = np.array(mask, dtype=np.float)[..., 0]  # shape: (H, W, #SegmapsPerImage)
    mask /= 255.
    return mask


class MosLAugmentation:
    def __init__(self, img_size):
        self.img_resize = iaa.Resize({'width': img_size[1], 'height': img_size[0]})

        self.aug_img = iaa.Sequential([
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
            iaa.Sometimes(0.4, iaa.Sequential([
                iaa.Sometimes(0.5, iaa.AddToHueAndSaturation(value=(-25, 25), per_channel=True)),
                iaa.Sometimes(0.5, iaa.Add((-20, 20), per_channel=0.5)),
                iaa.Sometimes(0.5, iaa.GammaContrast(gamma=(0.25, 1.25)))]))

            # iaa.OneOf([
            #     iaa.Sometimes(0.2, iaa.CLAHE()),
            #     iaa.Sometimes(0.4, iaa.Sequential([
            #         iaa.Sometimes(0.5, iaa.AddToHueAndSaturation(value=(-25, 25), per_channel=True)),
            #         iaa.Sometimes(0.5, iaa.Add((-20, 20), per_channel=0.5)),
            #         iaa.Sometimes(0.5, iaa.GammaContrast(gamma=(0.25, 1.25)))]))
            # ]),
            # iaa.OneOf([
            #     iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.5, 3.0))),
            #     iaa.Sometimes(0.3, iaa.MotionBlur(k=(5, 15))),
            #     iaa.AverageBlur(k=(3, 7))
            # ]),
            # iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.0, 1.0))),
            # iaa.Sometimes(0.2, iaa.JpegCompression(compression=(10, 80)))
        ])

        self.aug_all = iaa.Sequential([
            # iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.7, 1.2), "y": (0.7, 1.2)})),
            # iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            # iaa.Affine(scale=(0.5, 1.0)),
            iaa.Sometimes(0.5, iaa.Affine(translate_percent={'x': (-0.3, 0.3), 'y': (-0.3, 0.3)})),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-20, 20)))
        ])

    def __call__(self, img, mask):
        img = np.asarray(img)
        mask = mask2numpy(mask)
        img = self.img_resize(image=img)
        mask = self.img_resize(image=mask)

        aug_img_det = self.aug_img.to_deterministic()
        aug_all_det = self.aug_all.to_deterministic()

        img_aug = aug_img_det.augment_image(img)
        img_aug = aug_all_det.augment_image(img_aug)
        mask_aug = aug_all_det.augment_image(mask)
        mask_aug[mask_aug < 0] = 0
        return img_aug, mask_aug


def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target=0.8):
    # modifies boundary of the given mask.
    # remove consecutive vertice of the boundary by regional sample rate
    # ->
    # remove any vertice by sample rate
    # ->
    # move vertice by distance between vertice and center of the mask by move rate.
    # input: np array of size [H,W] image
    # output: same shape as input

    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # only modified contours is needed actually.
    sampled_contours = []
    modified_contours = []

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)

        # remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)

        idx_dist = []
        for i in range(number_of_vertices - number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i + number_of_removes]) ** 2)])

        idx_dist = sorted(idx_dist, key=lambda x: x[1])

        remove_start = random.choice(idx_dist[:math.ceil(0.1 * len(idx_dist))])[0]

        # remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
        new_contour = np.concatenate([contour[:remove_start], contour[remove_start + number_of_removes:]], axis=0)
        contour = new_contour

        # sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            # modify contours
            for idx, coor in enumerate(modified_contour):
                change = np.random.normal(0, move_rate)  # 0.1 means change position of vertex to 10 percent farther from center
                x, y = coor[0]
                new_x = x + (x - center[0]) * change
                new_y = y + (y - center[1]) * change

                modified_contour[idx] = [new_x, new_y]
        modified_contours.append(modified_contour)

    # draw boundary
    gt = np.copy(image)
    image = np.zeros_like(image)

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    image = perturb_seg(image, iou_target)

    return image
