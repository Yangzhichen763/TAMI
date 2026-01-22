import torch
import numpy as np
import math
import lpips

from basic.utils.convert import apply
from basic.utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes
from .error_func import get_func
from basic.utils.convert import apply, standardize, numpy2tensor



__all__ = ['calculate_lpips', 'LPIPS']


@METRICS_REGISTRY.register()
class LPIPS:
    """
    Calculate MABD(Mean Absolute Brightness Difference) between two videos.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
    """
    metric_mode = 'FR'

    def __init__(self, reduction='mean', **metrics_kwargs):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.reduction = reduction
        self.net_name = metrics_kwargs.pop('net_name', 'alex')
        self.net = lpips.LPIPS(net=self.net_name, verbose=False)
        self.metrics_kwargs = metrics_kwargs

        self.metrics_kwargs.update(dict(lpips_net=self.net))


    def __call__(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (B, N, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, N, C, H, W). Ground truth tensor.
        """
        pred = torch.round(pred * 255) / 255
        target = torch.round(target * 255) / 255

        return calculate_lpips(pred, target, reduction=self.reduction, **self.metrics_kwargs)


# [⭐]计算两个图像（numpy 数组）的 LPIPS 值
# noinspection SpellCheckingInspection
@paired_reduce
def calculate_lpips(image_1, image_2, lpips_net, device='cpu'):
    """
    Calculate LPIPS between two images (numpy arrays or PyTorch tensors).

    Args:
        image_1 (torch.Tensor): Image A. A numpy array, with range [0, 1], and shape (H, W, C)
        image_2 (torch.Tensor): Image B. A numpy array, with range [0, 1], and shape (H, W, C)
        lpips_net (torch.nn.Module): The LPIPS network.
        device (str): The device to run the calculations on.

    Returns:
        np.ndarray or float: LPIPS value, if input images have 3 dimensions, return a float value; else, return a numpy.ndarray with shape (N,)
    """
    def get_lpips_np(image_1, image_2):
        image_1 = numpy2tensor(image_1, change_range=False, clip_range=False)
        image_2 = numpy2tensor(image_2, change_range=False, clip_range=False)
        image_1 = standardize(image_1, 255).to(device)  # image / 127.5 - 1
        image_2 = standardize(image_2, 255).to(device)

        lpips_net.to(device)
        lpips = lpips_net.forward(image_1, image_2)
        return lpips

    def get_lpips_tensor(image_1, image_2):
        image_1 = standardize(image_1, 1)  # image / 127.5 - 1
        image_2 = standardize(image_2, 1)

        lpips = lpips_net.forward(image_1, image_2)
        return lpips

    # 两种方法（np 和 tensor）得到结果都是一致的
    return apply(image_1, image_2, get_lpips_np, get_lpips_tensor, device=device)
