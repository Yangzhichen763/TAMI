import numpy as np

import torch

from basic.utils.convert import apply
from basic.utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes
from .error_func import get_func


__all__ = ['calculate_mabd', 'MABD']


error_func_tensor = get_func('mse', "tensor")
error_func_np = get_func('mse', "np")


#region ==[MABD]==
@METRICS_REGISTRY.register()
class MABD:
    """
    Calculate MABD(Mean Absolute Brightness Difference) between two videos.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
    """

    def __init__(self, reduction='mean', **metrics_kwargs):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.reduction = reduction
        self.metrics_kwargs = metrics_kwargs

    def __call__(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (B, N, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, N, C, H, W). Ground truth tensor.
        """
        return calculate_mabd(pred, target, reduction=self.reduction, **self.metrics_kwargs)


#region ==[基础函数]==
# 计算两个视频的 MABD 值，视频可以是 numpy 数组[0, 255](N, H, W, C)或 PyTorch 张量(N, C, H, W)
# noinspection SpellCheckingInspection
@paired_reduce
def calculate_mabd(video_1, video_2):
    """
    Calculate the Mean Absolute Brightness Difference (MABD) between two videos.
    video_2 can be considered as the predicted video, while video_1 represents the GT (ground truth) video.

    Ref: "Learning to See Moving Objects in the Dark"(ICCV 2019)

    Args:
        video_1 (np.ndarray or torch.Tensor): The first video. if it is a numpy array, with range [0, 255], and shape (N, H, W, C); if it is a PyTorch tensor, with shape (N, C, H, W)
        video_2 (np.ndarray or torch.Tensor): The second video. if it is a numpy array, with range [0, 255], and shape (N, H, W, C); if it is a PyTorch tensor, with shape (N, C, H, W)

    Returns:
        dict: A dictionary containing the MABD values of all videos in the directory.
        Contains the following keys:
            'value': The MABD value of the two videos.
            'vector': The MABD vector of the second video.
    Example:
    >>> # Generate two random videos
    >>> np.random.seed(0)
    >>> vid_1 = np.random.rand(10, 160, 240, 3)
    >>> vid_2 = vid_1 + 0.1 * np.random.rand(10, 160, 240, 3)
    ...
    >>> def to_uint8(video):
    ...     return (video * 255).round().astype(np.uint8)
    ...
    >>> # Calculate MABD between the two videos
    >>> mabd = get_mabd(to_uint8(vid_1), to_uint8(vid_2))
    >>> print(mabd)
    0.015627016386975365
    """
    return apply(video_1, video_2, get_mabd_np, get_mabd_tensor)


# [⭐]计算 RGB 值对应的 brightness
def get_brightness(R, G, B):
    """
    Calculate the brightness of input RGB values.

    Args:
        R: The red value of the input RGB values.
        G: The green value of the input RGB values.
        B: The blue value of the input RGB values.

    Returns:
        The brightness of the input RGB values.
    """
    return 0.2126 * R + 0.7152 * G + 0.0722 * B  # refer to https://en.wikipedia.org/wiki/Relative_luminance


# 计算两个视频序列帧（numpy 数组）的 Mean Absolute Brightness Difference (MABD)
# noinspection SpellCheckingInspection
def get_mabd_np(video_1, video_2):
    """
    Calculate the Mean Absolute Brightness Difference (MABD) between two videos.
    video_2 can be considered as the predicted video, while video_1 represents the GT (ground truth) video.

    Args:
        video_1 (np.ndarray): The first video, with range [0, 255], and shape (N, H, W, C).
        video_2 (np.ndarray): The second video, with range [0, 255], and shape (N, H, W, C).
    """
    # 计算一个视频序列帧的 brightness
    # noinspection PyPep8Naming
    def brightness(video):
        """
        Calculate the brightness of a video sequence frame.

        Args:
            video (np.ndarray): The video sequence frame, with range [0, 255], and shape (N, H, W, C).

        Returns:
            np.ndarray: The brightness of the video sequence frame.
        """
        R, G, B = video[..., 0], video[..., 1], video[..., 2]
        return get_brightness(R, G, B)

    # 计算视频序列帧的 brightness
    br_video_1 = brightness(video_1)                                # (N, H, W, C) -> (N, H, W)
    br_video_2 = brightness(video_2)

    # 对视频序列帧的 brightness 进行差分，计算 MABD
    mabd_1 = np.abs(np.diff(br_video_1, axis=0)).mean(axis=(1, 2))  # (N, H, W) -> (N-1, H, W) -> (N-1)
    mabd_2 = np.abs(np.diff(br_video_2, axis=0)).mean(axis=(1, 2))

    # 计算 ?-MABD 值
    mabd = error_func_np(mabd_1, mabd_2)                          # (N-1) -> scalar
    return mabd
    # {
    #     'value': mabd.item(),
    #     'vector': mabd_2
    # }


# [⭐]计算两个视频序列帧（PyTorch 张量）的 MABD
# noinspection SpellCheckingInspection
def get_mabd_tensor(video_1, video_2):
    """
    Calculate the Mean Absolute Brightness Difference (MABD) between two videos using PyTorch.
    video_2 can be considered as the predicted video, while video_1 represents the GT (ground truth) video.

    Args:
        video_1 (torch.Tensor): The first video, with range [0, 1], and shape (N, C, H, W).
        video_2 (torch.Tensor): The second video, with range [0, 1], and shape (N, C, H, W).
    """
    # 计算一个视频序列帧的 brightness
    # noinspection PyPep8Naming
    def brightness(video):
        """
        Calculate the brightness of a video sequence frame.

        Args:
            video (torch.Tensor): The video sequence frame, with range [0, 1], and shape (N, C, H, W).

        Returns:
            torch.Tensor: The brightness of the video sequence frame.
        """
        R, G, B = video[..., 0, :, :], video[..., 1, :, :], video[..., 2, :, :]
        return get_brightness(R, G, B)

    video_1 = torch.round(video_1 * 255) / 255
    video_2 = torch.round(video_2 * 255) / 255

    # 计算视频序列帧的 brightness
    br_video_1 = brightness(video_1)                                        # (N, C, H, W) -> (N, H, W)
    br_video_2 = brightness(video_2)

    # 对视频序列帧的 brightness 进行差分，计算 MABD
    mabd_1 = torch.abs(br_video_1[1:] - br_video_1[:-1]).mean(dim=(1, 2))   # (N-1, H, W) -> (N-1)
    mabd_2 = torch.abs(br_video_2[1:] - br_video_2[:-1]).mean(dim=(1, 2))

    # 计算 ?-MABD 值
    mabd = error_func_tensor(mabd_1, mabd_2)                                 # (N-1) -> scalar
    return mabd


# noinspection SpellCheckingInspection
def get_mabd_vector(video):
    """
    Calculate the Mean Absolute Brightness Difference (MABD) vector of a single video using PyTorch.

    Args:
        video (torch.Tensor): The first video, with range [0, 1], and shape (N, C, H, W).
    """
    # 计算一个视频序列帧的 brightness
    # noinspection PyPep8Naming
    def brightness(video):
        """
        Calculate the brightness of a video sequence frame.

        Args:
            video (torch.Tensor): The video sequence frame, with range [0, 1], and shape (N, C, H, W).

        Returns:
            torch.Tensor: The brightness of the video sequence frame.
        """
        R, G, B = video[..., 0, :, :], video[..., 1, :, :], video[..., 2, :, :]
        return get_brightness(R, G, B)

    video = torch.round(video * 255) / 255

    # 计算视频序列帧的 brightness
    br_video = brightness(video)                                        # (N, C, H, W) -> (N, H, W)

    # 对视频序列帧的 brightness 进行差分，计算 MABD
    mabd_vector = torch.abs(br_video[1:] - br_video[:-1]).mean(dim=(1, 2))   # (N-1, H, W) -> (N-1)
    return mabd_vector
#endregion
#endregion


#region ==[Peak MABD]==
@METRICS_REGISTRY.register()
class PMABD:
    """
    Calculate PMABD(Peak Mean Absolute Brightness Difference) between two videos.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
    """

    def __init__(self, reduction='mean', **metrics_kwargs):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.reduction = reduction
        self.metrics_kwargs = metrics_kwargs

    def __call__(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (B, N, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, N, C, H, W). Ground truth tensor.
        """
        return calculate_pmabd(pred, target, reduction=self.reduction, **self.metrics_kwargs)


@paired_reduce
def calculate_pmabd(video_1, video_2):
    return apply(video_1, video_2, get_pmabd_np, get_pmabd_tensor)


def get_pmabd_np(video_1, video_2, eps=1e-10):
    mabd = get_mabd_np(video_1, video_2)
    pmabd = -10 * np.log10(np.sqrt(mabd) + eps)
    return pmabd


def get_pmabd_tensor(video_1, video_2, eps=1e-10):
    mabd = get_mabd_tensor(video_1, video_2)
    pmabd = -10 * torch.log10(torch.sqrt(mabd) + eps)
    return pmabd
#endregion