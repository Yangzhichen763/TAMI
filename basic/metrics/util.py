import numpy as np
import functools
from torch.nn import functional as F

import torch.nn as nn
import torch
from typing import Optional, Union, Callable, List
from basic.utils.console.log import highlight_diff
from copy import deepcopy



_reduction_modes = ['none', 'mean', 'sum']


# the same as basic/archs/util
def clone_module(
        module: nn.Module,
        n: Optional[int] = None,
        reset_parameters: Union[bool, Callable] = True,
        init_fn: Optional[Callable] = None,
        param_names: Optional[List[str]] = None,
) -> Union[nn.Module, nn.ModuleList]:
    """Deep clone a module with optional parameter re-initialization.

    Args:
        module: The module to clone.
        n: If None or 1, return a single cloned module.
           If >1, return a ModuleList of cloned modules.
        reset_parameters:
           - If True, call module.reset_parameters() if it exists.
           - If False, skip initialization.
           - If a function (e.g., `lambda m: m.weight.data.normal_(0, 0.01)`), use it to initialize.
        init_fn: A function to initialize parameters (deprecated if reset_parameters is a function).
        param_names: Only initialize parameters whose name contains any of these strings (e.g., ['weight']).
    """

    def _deep_copy_module(module: nn.Module, memo=None) -> nn.Module:
        if memo is None:
            memo = {}

        if not isinstance(module, torch.nn.Module):
            return module

        # check if module has already been cloned
        if id(module) in memo:
            return memo[id(module)]

        # create a new module without __init__
        clone = module.__new__(type(module))
        memo[id(module)] = clone

        # shallow copy basic attributes
        clone.__dict__ = {
            k: v for k, v in module.__dict__.items()
        }

        # deep copy parameters, buffers, and child modules
        clone._parameters = {}
        for name, param in module._parameters.items():
            if param is not None:
                param_ptr = param.data_ptr()
                if param_ptr in memo:
                    clone._parameters[name] = memo[param_ptr]
                else:
                    cloned_param = param.clone()
                    clone._parameters[name] = cloned_param
                    memo[param_ptr] = cloned_param

        clone._buffers = {}
        for name, buffer in module._buffers.items():
            if buffer is not None:
                buffer_ptr = buffer.data_ptr()
                if buffer_ptr in memo:
                    clone._buffers[name] = memo[buffer_ptr]
                else:
                    cloned_buffer = buffer.clone()
                    clone._buffers[name] = cloned_buffer
                    memo[buffer_ptr] = cloned_buffer

        clone._modules = {}
        for name, child in module._modules.items():
            if child is not None:
                clone._modules[name] = _deep_copy_module(child, memo)

        clone.training = module.training
        return clone


    def _detach_module(module: nn.Module):
        if not isinstance(module, torch.nn.Module):
            return

        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                module._parameters[param_key] = module._parameters[param_key].detach_()

        for buffer_key in module._buffers:
            if module._buffers[buffer_key] is not None and \
                    module._buffers[buffer_key].requires_grad:
                module._buffers[buffer_key] = module._buffers[buffer_key].detach_()

        for module_key in module._modules:
            _detach_module(module._modules[module_key])


    def _check_shared_params_or_buffers(module1, module2):
        # 检查参数
        for (name1, param1), (name2, param2) in zip(module1.named_parameters(), module2.named_parameters()):
            if param1.data_ptr() == param2.data_ptr():
                print(f"Parameter '{name1}' and '{name2}' share memory (potential reference)!")

        # 检查缓冲区
        for (name1, buf1), (name2, buf2) in zip(module1.named_buffers(), module2.named_buffers()):
            if buf1.data_ptr() == buf2.data_ptr():
                print(f"Buffer '{name1}' and '{name2}' share memory (potential reference)!")

        # 递归检查子模块
        for (name1, child1), (name2, child2) in zip(module1.named_children(), module2.named_children()):
            _check_shared_params_or_buffers(child1, child2)


    def _are_modules_fully_independent(module1, module2):
        # 检查参数是否独立
        for (name1, param1), (name2, param2) in zip(module1.named_parameters(), module2.named_parameters()):
            if param1.data_ptr() == param2.data_ptr():
                return False

        # 检查缓冲区是否独立
        for (name1, buf1), (name2, buf2) in zip(module1.named_buffers(), module2.named_buffers()):
            if buf1.data_ptr() == buf2.data_ptr():
                return False

        # 递归检查子模块
        for (name1, child1), (name2, child2) in zip(module1.named_children(), module2.named_children()):
            if not _are_modules_fully_independent(child1, child2):
                return False

        return True


    def _clone_module(module: nn.Module) -> nn.Module:
        # Plan A: Deep copy the module
        _module = deepcopy(module)
        if module.__repr__() != _module.__repr__():
            print(f"{highlight_diff(module.__repr__(), _module.__repr__())}\n"
                  f"Cloned module has different parameters.")
        _check_shared_params_or_buffers(module, _module)
        if not _are_modules_fully_independent(module, _module):
            print("Cloned module shares parameters or buffers with the original module.")
        # Plan B: Shallow copy the module and manually copy parameters and buffers
        # _module = type(module)(*module.args, **module.kwargs)
        # _module.load_state_dict(deepcopy(module.state_dict()))
        # Plan C: Deep copy the module and detach all parameters and buffers
        # _module = _deep_copy_module(module)
        # _detach_module(_module)

        # Case 1: reset_parameters is a custom function
        if callable(reset_parameters):
            reset_parameters(_module)

        # Case 2: reset_parameters is True and module has .reset_parameters()
        elif reset_parameters and hasattr(_module, 'reset_parameters'):
            _module.reset_parameters()

        # Case 3: Use init_fn if provided (fallback)
        elif init_fn is not None:
            _apply_init_fn(_module, init_fn, param_names)

        return _module


    def _apply_init_fn(
            module: nn.Module,
            init_fn: Callable,
            param_names: Optional[List[str]] = None,
    ) -> None:
        """
        Helper to apply init_fn to parameters recursively.
        """
        for name, param in module.named_parameters(recurse=False):
            if param_names is None or any(p in name for p in param_names):
                init_fn(param)

        # Recursively apply to child modules
        for child in module.children():
            _apply_init_fn(child, init_fn, param_names)

    # Validate n
    if n is not None and n <= 0:
        raise ValueError(f"n must be positive or None, got {n}")

    # Clone single module
    if n is None:
        module = _clone_module(module)
        return module

    # Clone into ModuleList
    module_list = nn.ModuleList([_clone_module(module) for _ in range(n)])
    return module_list


'''
Modified from BasicSR(https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py)
'''
def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


#region ==[Color Utils]==
'''
Modified from BasicSR: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/color_util.py
'''
def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                  [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    """Convert a YCbCr image to RGB image.

    This function produces the same results as Matlab's ycbcr2rgb function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted RGB image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [
                                  -222.921, 135.576, -276.836
                              ]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    """Convert a YCbCr image to BGR image.

    The bgr version of ycbcr2rgb.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted BGR image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [
                                  -276.836, 135.576, -222.921
                              ]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img
#endregion


def reduce(tensor, reduction='mean'):
    """Reduce tensor as specified.

    Args:
        tensor (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return tensor
    elif reduction_enum == 1:
        return tensor.mean()
    else:
        return tensor.sum()


def paired_reduce(metrics_func):
    """
    Create a reduction version for metrics function.
    """

    @functools.wraps(metrics_func)
    def wrapper(pred, target, reduction='mean', **kwargs):
        loss = metrics_func(pred, target, **kwargs)
        loss = reduce(loss, reduction)
        return loss

    return wrapper


def unpaired_reduce(metrics_func):
    """
    Create a reduction version for metrics function.
    """

    @functools.wraps(metrics_func)
    def wrapper(pred, reduction='mean', **kwargs):
        loss = metrics_func(pred, **kwargs)
        loss = reduce(loss, reduction)
        return loss

    return wrapper