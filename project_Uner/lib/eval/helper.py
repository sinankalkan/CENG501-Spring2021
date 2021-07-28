import numpy as np
from PIL import Image


def green_background(image, mask):
    green_image = np.zeros_like(image)
    green_image[:, :, :] = [0, 255, 0]
    out_img = (image * mask + (green_image * (1 - mask))).astype("uint8")
    out_img = Image.fromarray(out_img)
    return out_img
