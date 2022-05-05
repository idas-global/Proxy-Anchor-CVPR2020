from __future__ import print_function
from __future__ import division

import torchvision
from torchvision import transforms
import PIL.Image
import torch
import random


def std_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).std(dim=1)


def mean_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).mean(dim=1)


class Identity():  # used for skipping transforms
    def __call__(self, im):
        return im


class print_shape():
    def __call__(self, im):
        print(im.size)
        return im


class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


class pad_shorter():
    def __call__(self, im):
        h, w = im.size[-2:]
        s = max(h, w)
        new_im = PIL.Image.new("RGB", (s, s))
        new_im.paste(im, ((s - h) // 2, (s - w) // 2))
        return new_im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
                         tensor - self.in_range[0]
                 ) / (
                         self.in_range[1] - self.in_range[0]
                 ) * (
                         self.out_range[1] - self.out_range[0]
                 ) + self.out_range[0]
        return tensor


class Transform:
    def __init__(self, is_inception=False):
        self.sz_resize = 256
        self.sz_crop = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if is_inception:
            self.sz_resize = 256
            self.sz_crop = 224
            self.mean = [104, 117, 128]
            self.std = [1, 1, 1]

    def make_transform(self, is_train=True, is_inception=False, crop=True):
        # Resolution Resize List : 256, 292, 361, 512
        # Resolution Crop List: 224, 256, 324, 448

        resnet_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.sz_crop) if is_train and crop else Identity(),
            transforms.RandomHorizontalFlip() if is_train else Identity(),
            transforms.RandomVerticalFlip() if is_train and not crop else Identity(),
            transforms.Resize(self.sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(self.sz_crop) if not is_train and crop else Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        inception_transform = transforms.Compose(
            [
                RGBToBGR(),
                transforms.RandomResizedCrop(self.sz_crop) if is_train else Identity(),
                transforms.RandomHorizontalFlip() if is_train else Identity(),
                transforms.Resize(self.sz_resize) if not is_train else Identity(),
                transforms.CenterCrop(self.sz_crop) if not is_train else Identity(),
                transforms.ToTensor(),
                ScaleIntensities([0, 1], [0, 255]),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

        return inception_transform if is_inception else resnet_transform
