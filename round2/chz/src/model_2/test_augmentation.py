import math
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import numbers
import types
import collections
import warnings

class FlipAndRotateCrop(object):
    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return flip_and_rotate_crop(img, self.size, self.vertical_flip)

def crop(img, i, j, h, w):
    if not isinstance(img, Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))
    
def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)
    
def special_crop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    # tl = img.crop((0, 0, crop_w, crop_h))
    # tr = img.crop((w - crop_w, 0, w, crop_h))
    # bl = img.crop((0, h - crop_h, crop_w, h))
    # br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    
    # return (tl, tr, bl, br, center)
    return (center,)

def flip_and_crop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_imgs = special_crop(img, size)

    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    second_imgs = special_crop(img, size)
    
    return first_imgs + second_imgs

def flip_and_rotate_crop(img, size, angles):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
    if isinstance(angles, numbers.Number):
        angles = [angles]
    
    all_imgs = flip_and_crop(img, size) * 2
    
    for angle in angles:
        all_imgs += flip_and_crop(img.rotate(angle), size)
        all_imgs += flip_and_crop(img.rotate(-angle), size)
    
    return all_imgs

