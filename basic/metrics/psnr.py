import numpy as np

import torch

from basic.utils.convert import apply
from basic.utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes
from .error_func import get_func


__all__ = ['calculate_psnr', 'PSNR']


error_func_tensor = get_func('mse', "tensor")
error_func_np = get_func('mse', "np")


@METRICS_REGISTRY.register()
class PSNR:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

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
        self.metrics_kwargs = metrics_kwargs

    def __call__(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
        """
        return calculate_psnr(pred, target, reduction=self.reduction, **self.metrics_kwargs)


# noinspection SpellCheckingInspection
@paired_reduce
def calculate_psnr(image_1, image_2, crop_border=0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        image_1 (np.ndarray or torch.Tensor): if input is numpy array, it should have range [0, 255], and shape (H, W, C);
            if input is PyTorch tensor, it should have range [0, 1], and shape (C, H, W) or (N, C, H, W)
        image_2 (np.ndarray or torch.Tensor): same as image_1
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.

    Returns:
        float or np.ndarray: PSNR value, if input images have shape (H, W, C), return a float value; else, return a numpy.ndarray with shape (N,)
    """
    return apply(image_1, image_2, get_psnr_np, get_psnr_tensor, crop_border=crop_border)


#region get_psnr_np() and get_psnr_tensor()
# 计算两个图像（numpy 数组）的 PSNR 值
# 输入为 numpy 数组，范围为 [0, 255]，形状为 (H, W, C)，如果要和 numpy 的计算结果一致，要先将输入张量使用 util.convert.to_uint8()
# noinspection SpellCheckingInspection
def get_psnr_np(image_1, image_2, crop_border=0):
    """
    Calculate PSNR between two images (numpy arrays), with range [0, 255], and shape (H, W, C)

    Args:
        image_1 (np.ndarray): Image A, with range [0, 255], and shape (H, W, C)
        image_2 (np.ndarray): Image B, with range [0, 255], and shape (H, W, C)
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
    Returns:
        np.ndarray or float: PSNR value
    """
    # 确保图像数据类型为浮点数
    image_1 = image_1.astype(np.float64)
    image_2 = image_2.astype(np.float64)

    # 剪裁图像边缘
    if crop_border != 0:
        image_1 = image_1[..., crop_border:-crop_border, crop_border:-crop_border, :]
        image_2 = image_2[..., crop_border:-crop_border, crop_border:-crop_border, :]

    # 计算 MSE
    error = error_func_np(image_1, image_2)
    mse = np.mean(error, axis=(-3, -2, -1))

    # 计算 PSNR
    # 如果 MSE 为 0，说明两幅图像完全相同，PSNR 为无穷大
    psnr = -20 * np.log10(np.sqrt(mse) / 255.0)

    '''
    # 或者可以使用下方现成的 psnr 计算函数
    from skimage.metrics import peak_signal_noise_ratio
    psnr = peak_signal_noise_ratio(image_1, image_2, data_range=255)
    '''

    return psnr


# 计算两个或两组图像（PyTorch 张量）的 PSNR 值
# noinspection SpellCheckingInspection
def get_psnr_tensor(image_1, image_2, crop_border=0):
    """
    Calculate PSNR between two images (PyTorch tensors), with range [0, 1], and shape (C, H, W) or (N, C, H, W)

    Args:
        image_1 (torch.Tensor): Image A, with range [0, 1], and shape (C, H, W) or (N, C, H, W)
        image_2 (torch.Tensor): Image B, with range [0, 1], and shape (C, H, W) or (N, C, H, W)
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
    Returns:
        torch.Tensor or float: PSNR value, if input images have shape (C, H, W), return a float value; else, return a numpy.ndarray with shape (N,)
    """
    # 确保图像数据类型为浮点数
    image_1 = image_1.type(torch.float64)
    image_2 = image_2.type(torch.float64)

    # 剪裁图像边缘
    if crop_border != 0:
        image_1 = image_1[crop_border:-crop_border, crop_border:-crop_border, ...]
        image_2 = image_2[crop_border:-crop_border, crop_border:-crop_border, ...]

    image_1 = torch.round(image_1 * 255) / 255
    image_2 = torch.round(image_2 * 255) / 255

    # 计算 MSE
    error = error_func_tensor(image_1, image_2)
    mse = torch.mean(error, dim=(-3, -2, -1))

    # 计算 PSNR
    # 如果 MSE 为 0，说明两幅图像完全相同，PSNR 为无穷大
    psnr = -20 * torch.log10(torch.sqrt(mse))
    psnr = psnr.detach().cpu()
    return psnr
#endregion