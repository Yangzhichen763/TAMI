import numpy as np
import torch
import torchvision.transforms.functional as TF


class TransformBase:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if self.p < 1.0 and np.random.rand() > self.p:
            return img

        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)

        original_shape = img.shape
        if img.dim() == 3:
            img = img.unsqueeze(0)

        imgs = self.call(img)
        if isinstance(imgs, tuple):
            img = imgs[0]
            others = imgs[1:]
        else:
            img = imgs
            others = None

        if original_shape[0] == 3:
            img = img.squeeze(0)

        if others is not None:
            return img, *others
        else:
            return img

    def call(self, img):
        return img
