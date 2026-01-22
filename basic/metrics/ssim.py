import cv2
import numpy as np

import torch
import torch.nn.functional as F

from basic.utils.convert import apply
from basic.utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes


__all__ = ['calculate_ssim', 'SSIM']


GAUSSIAN_WINDOW = None  # 用于缓存高斯核


@METRICS_REGISTRY.register()
class SSIM:
    """Structural Similarity Index Measure (SSIM) .

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the SSIM calculation.
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
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return calculate_ssim(pred, target, reduction=self.reduction, **self.metrics_kwargs)


@paired_reduce
def calculate_ssim(image_1,
                   image_2,
                   crop_border=0,
                   gray_scale=False):
    """
    Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        image_1 (np.ndarray or torch.Tensor): if input is numpy array, it should have range [0, 255], and shape (H, W, C);
            if input is PyTorch tensor, it should have range [0, 1], and shape (C, H, W) or (N, C, H, W)
        image_2 (np.ndarray or torch.Tensor): same as image_1
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the SSIM calculation.
        gray_scale (bool): Whether to convert the images to grayscale before calculating SSIM.

    Returns:
        float or np.ndarray: SSIM value, if input images have shape (H, W, C), return a float value; else, return a numpy.ndarray with shape (N,)
    """
    return apply(image_1, image_2, get_ssim_np, get_ssim_tensor, crop_border=crop_border, gray_scale=gray_scale)


#region get_ssim_np and get_ssim_tensor
# 计算两个图像（numpy 数组）的 SSIM 值
def get_ssim_np(image_1, image_2, crop_border=0, gray_scale=False):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        image_1 (np.ndarray): Image A. with range [0, 255], and shape (H, W, C) or (N, H, W, C)
        image_2 (np.ndarray): Image B. with range [0, 255], and shape (H, W, C) or (N, H, W, C)
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the SSIM calculation.
        gray_scale (bool): Whether to convert the images to grayscale before calculating SSIM.

    Returns:
        np.ndarray or float: SSIM value, if input images have 3 dimensions, return a float value; else, return a numpy.ndarray with shape (N,)
    """
    # 剪裁图像边缘
    if crop_border != 0:
        image_1 = image_1[..., crop_border:-crop_border, crop_border:-crop_border, :]
        image_2 = image_2[..., crop_border:-crop_border, crop_border:-crop_border, :]

    # 处理单通道图像
    if image_1.ndim == 2:
        return ssim_np(image_1, image_2)

    # 处理多通道图像 (H, W, C)
    elif image_1.ndim == 3:
        _, _, C = image_1.shape
        # 如果是 RGB 图像
        if C == 3:
            # 如果选择先转化为 grayscale 图像再计算，则先转化为 grayscale 图像
            if gray_scale:
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
                return ssim_np(image_1, image_2)
            # 否则计算每个通道的 SSIM 平均值
            else:
                ssim_list = np.zeros(C)
                for i in range(3):
                    ssim_list[i] = ssim_np(image_1[:, :, i], image_2[:, :, i])
                return ssim_list.mean()
        # 如果是 grayscale 图像，则计算单通道的 SSIM
        elif C == 1:
            return ssim_np(np.squeeze(image_1), np.squeeze(image_2))

    # 处理 batched 多通道图像 (N, H, W, C)
    elif image_1.ndim == 4:
        N, _, _, _ = image_1.shape
        ssim_list = np.zeros(N)
        for i in range(N):
            ssim_list[i] = get_ssim_np(image_1[i], image_2[i])
        return ssim_list
    else:
        raise ValueError(f'Dimension of input image must be 2, 3 or 4, instead of {image_1.ndim}.')


# 计算两个图像（PyTorch 数组）的 SSIM 值
# noinspection SpellCheckingInspection
def get_ssim_tensor(image_1, image_2, crop_border=0, gray_scale=False):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        image_1 (torch.Tensor): Image A. with shape (C, H, W) or (N, C, H, W)
        image_2 (torch.Tensor): Image B. with shape (C, H, W) or (N, C, H, W)
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the SSIM calculation.
        gray_scale (bool): Whether to convert the images to grayscale before calculating SSIM.

    Returns:
        np.ndarray or float: SSIM value, if input images have 3 dimensions, return a float value; else, return a numpy.ndarray with shape (N,)
    """
    image_1 = torch.round(image_1 * 255) / 255
    image_2 = torch.round(image_2 * 255) / 255

    C = image_1.shape[-3]
    # 如果是 RGB 图像，并且选择先转化为 grayscale 图像再计算，则先转化为 grayscale 图像再计算 ssim
    if C == 3 and gray_scale:
        image_1 = torch.mean(image_1, dim=-3, keepdim=True)
        image_2 = torch.mean(image_2, dim=-3, keepdim=True)

    # 剪裁图像边缘
    if crop_border != 0:
        image_1 = image_1[..., crop_border:-crop_border, crop_border:-crop_border, :]
        image_2 = image_2[..., crop_border:-crop_border, crop_border:-crop_border, :]

    # 计算 SSIM
    ssim_value = ssim_tensor(image_1, image_2)
    return ssim_value


# 计算两个图像的 SSIM 值，图像的维度只能是 2，即 (H, W)
# noinspection SpellCheckingInspection,PyPep8Naming
def ssim_np(image_1, image_2):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        image_1 (np.ndarray): Image A. with range [0, 255], and shape (H, W)
        image_2 (np.ndarray): Image B. with range [0, 255], and shape (H, W)

    Returns:
        float: SSIM value
    """
    # 常数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    sigma = 1.5
    ws = 11         # window size
    hws = ws // 2   # half window size

    # 将图像数据类型转换为 float64
    image_1 = image_1.astype(np.float64)
    image_2 = image_2.astype(np.float64)

    # 创建高斯核窗口
    kernel = cv2.getGaussianKernel(ws, sigma)       # ws×ws 高斯核，标准差为 sigma
    window = np.outer(kernel, kernel.transpose())   # 生成 2D 高斯窗口

    # 计算图像的局部均值
    mu1 = cv2.filter2D(image_1, -1, window)[hws:-hws, hws:-hws]  # guassion 加权计算均值，使用 'valid' 模式，即裁剪边缘模式
    mu2 = cv2.filter2D(image_2, -1, window)[hws:-hws, hws:-hws]

    # 计算均方
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算图像的局部方差和协方差
    sigma1_sq = cv2.filter2D(image_1 ** 2, -1, window)[hws:-hws, hws:-hws] - mu1_sq
    sigma2_sq = cv2.filter2D(image_2 ** 2, -1, window)[hws:-hws, hws:-hws] - mu2_sq
    sigma12 = cv2.filter2D(image_1 * image_2, -1, window)[hws:-hws, hws:-hws] - mu1_mu2

    # 计算 SSIM 映射
    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

    return ssim_map.mean()


# 计算两个图像的 SSIM 值
# noinspection SpellCheckingInspection,PyPep8Naming
def ssim_tensor(image_1, image_2):
    """
    Calculates the Structural Similarity Index (SSIM) between two images using PyTorch.

    Args:
        image_1 (torch.Tensor): Image A with range [0, 1], and shape (N, C, H, W).
        image_2 (torch.Tensor): Image B with range [0, 1], and shape (N, C, H, W).

    Returns:
        float: SSIM value
    """
    # 常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ws = 11         # window size
    hws = ws // 2   # half window size
    sigma = 1.5

    input_dim = image_1.dim()
    if input_dim == 3:
        image_1 = image_1.unsqueeze(0)
        image_2 = image_2.unsqueeze(0)
    N, C, H, W = image_1.shape
    device = image_1.device

    # 创建高斯核窗口
    global GAUSSIAN_WINDOW
    if GAUSSIAN_WINDOW is None:
        def create_window(window_size, sigma):
            # 创建高斯核窗口
            x = torch.arange(window_size, dtype=torch.float32) - hws
            y = torch.arange(window_size, dtype=torch.float32) - hws

            # 下面这段代码和这段代码功能一致 x, y = torch.meshgrid(x, y, indexing='ij')
            x = x.view(window_size, 1).repeat(1, window_size)  # (window_size, window_size)
            y = y.view(1, window_size).repeat(window_size, 1)  # (window_size, window_size)

            window = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            window = window / window.sum()  # 归一化

            # 将 kernel 扩展为 4D 张量 (out_channels, in_channels, H, W)
            window = window.view(1, 1, ws, ws).repeat(C, 1, 1, 1)  # expend(C, 1, window_size, window_size).contiguous
            return window
        GAUSSIAN_WINDOW = create_window(ws, sigma)
    kernel = GAUSSIAN_WINDOW.to(device)

    # 计算图像的局部均值
    mu1 = F.conv2d(image_1, kernel, padding=hws, groups=C)
    mu2 = F.conv2d(image_2, kernel, padding=hws, groups=C)

    # 计算均方
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算图像的局部方差和协方差
    sigma1_sq = F.conv2d(image_1 ** 2, kernel, padding=hws, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(image_2 ** 2, kernel, padding=hws, groups=C) - mu2_sq
    sigma12 = F.conv2d(image_1 * image_2, kernel, padding=hws, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 计算 SSIM 值的均值
    ssim_values = ssim_map.mean(dim=(-3, -2, -1))
    ssim_values = ssim_values.detach().cpu()
    return ssim_values
#endregion