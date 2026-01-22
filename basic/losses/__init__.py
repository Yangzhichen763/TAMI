from copy import deepcopy
import torch.nn as nn

from .basic_loss import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty,PSNRLoss,SSIMLoss)
from basic.losses.pixel_wise_losses.perceptual_loss import PerceptualLoss

from basic.options.options import parse_params, parse_arguments
from basic.utils.registry import LOSS_REGISTRY

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'PerceptualLoss', 'GANLoss',
    'PSNRLoss', 'SSIMLoss',
    'get_loss_func'
]


from basic.utils.console.log import get_root_logger
logger = get_root_logger()


# 将每个计算损失的模块都导入进来，同时会激活模块中的 register 装饰器，将损失函数注册到 MODEL_REGISTRY 中。
# automatically scan and import losses modules
# scan all the files under the 'losses' folder and collect files
import importlib
import os.path as osp
from basic.utils.path import scandir
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder, suffix='.py')
]
# import all the model modules
_model_modules = []
for file_name in model_filenames:
    try:
        _model_modules.append(importlib.import_module(f'basic.losses.{file_name}'))
    except Exception as e:
        logger.warning(f'Failed to import {file_name} because of {e}')


def get_loss_func(option):
    """
    Build loss from options.

    Args:
        option (dict): Configuration. It must contain:
            - type (str): the type of loss.

    Returns:
        nn.Module: The constructed loss.
    """
    option = deepcopy(option)  # deepcopy the option to avoid modification

    loss_type, loss_params = parse_params(option)

    # dynamic instantiation
    loss_class = LOSS_REGISTRY.try_get(loss_type)
    if loss_class is None:
        if getattr(nn, loss_type, None) is None:
            raise ValueError(f'Loss {loss_type} is not supported.')
        loss_class = getattr(nn, loss_type)
    args, kwargs = parse_arguments(loss_class, loss_params)
    loss_func = loss_class(*args, **kwargs)

    return loss_func


def try_get_loss_func(option_list, required_loss_type):
    """
    Build loss from options.

    Args:
        option_list (list): Configuration list. Each element must contain:
            - type (str): the type of loss.

    Returns:
        nn.Module: The constructed loss.
    """
    for option in option_list:
        option = deepcopy(option)  # deepcopy the option to avoid modification

        loss_type, loss_params = parse_params(option)
        if required_loss_type != loss_type:
            continue
        if 'name' in loss_params and required_loss_type != loss_params['name']:
            continue

        # dynamic instantiation
        loss_class = LOSS_REGISTRY.try_get(loss_type)
        if loss_class is None:
            if getattr(nn, loss_type, None) is None:
                raise ValueError(f'Loss {loss_type} is not supported.')
            loss_class = getattr(nn, loss_type)
        loss_func = loss_class(**loss_params)

        return loss_func
    return None


def get_all_loss_func(option_list):
    """
    Build loss from options.

    Args:
        option_list (list): Configuration list. Each element must contain:
            - type (str): the type of loss.

    Returns:
        nn.Module: The constructed loss.
    """
    loss_funcs = {}
    for option in option_list:
        option = deepcopy(option)  # deepcopy the option to avoid modification

        loss_type, loss_params = parse_params(option)
        loss_name =loss_type
        if 'name' in loss_params:
            loss_name = loss_params.pop('name')

        # dynamic instantiation
        loss_class = LOSS_REGISTRY.try_get(loss_type)
        if loss_class is None:
            if getattr(nn, loss_type, None) is None:
                raise ValueError(f'Loss {loss_type} is not supported.')
            loss_class = getattr(nn, loss_type)
        loss_func = loss_class(**loss_params)

        loss_funcs[loss_name] = loss_func
    return loss_funcs