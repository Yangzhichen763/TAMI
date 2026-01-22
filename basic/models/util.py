
from copy import deepcopy
import torch.optim as optim
import torch.optim.lr_scheduler as torch_lr_scheduler

from . import lr_scheduler  # import lr_scheduler to activate registry

from basic.options.options import parse_params
from basic.utils.registry import SCHEDULER_REGISTRY


def get_scheduler(optimizer, option):
    """
    Build scheduler from options.

    Args:
        optimizer (Optimizer): Optimizer of the model.
        option (dict): Configuration. It must contain:
            - type (str): the type of scheduler.

    Returns:
        _LRScheduler: Created scheduler.
    """
    option = deepcopy(option)  # deepcopy the option to avoid modification

    scheduler_type, scheduler_params = parse_params(option)

    # dynamic instantiation
    scheduler_class = SCHEDULER_REGISTRY.try_get(scheduler_type)
    if scheduler_class is None:
        if getattr(torch_lr_scheduler, scheduler_type, None) is None:
            raise ValueError(f'Loss {scheduler_type} is not supported.')
        scheduler_class = getattr(torch_lr_scheduler, scheduler_type)

    # recursively create scheduler for nested schedulers
    for key, value in scheduler_params.items():
        if isinstance(value, dict) and 'type' in value:
            scheduler_params[key] = get_scheduler(optimizer, value)

    scheduler = scheduler_class(optimizer, **scheduler_params)
    return scheduler


def get_named_scheduler(params, options, name=None):
    options = deepcopy(options)  # deepcopy the option to avoid modification

    if isinstance(options, list):
        for i, option in enumerate(options):
            if name is None or option['name'] == name:
                option.pop('name')
                return get_scheduler(params, option)
    elif isinstance(options, dict):
        return get_scheduler(params, options)

    raise ValueError(f'Scheduler {name} is not found.')


def get_schedulers(params, options):
    """
    Build schedulers from options.

    Args:
        params (list): List of model parameters to optimize.
        options (list): List of configurations. Each configuration must contain:
            - type (str): the type of optimizer.

    Returns:
        list: Created optimizers.
    """
    schedulers = {}
    for option in options:
        scheduler_name = option.pop('name')
        scheduler = get_scheduler(params, option)
        schedulers[scheduler_name] = scheduler
    return schedulers


def get_optimizer(params, option):
    """
    Build optimizer from options.

    Args:
        params (list): List of model parameters to optimize.
        option (dict): Configuration. It must contain:
            - type (str): the type of optimizer.

    Returns:
        Optimizer: Created optimizer.
    """
    option = deepcopy(option)  # deepcopy the option to avoid modification

    optimizer_type, optimizer_params = parse_params(option)

    # dynamic instantiation
    if getattr(optim, optimizer_type, None) is None:
        raise ValueError(f'Optimizer {optimizer_type} is not supported.')
    optimizer_class = getattr(optim, optimizer_type)
    optimizer = optimizer_class(params, **optimizer_params)
    return optimizer


def get_optimizers(params, options):
    """
    Build optimizers from options.

    Args:
        params (list): List of model parameters to optimize.
        options (list): List of configurations. Each configuration must contain:
            - type (str): the type of optimizer.

    Returns:
        list: Created optimizers.
    """
    optimizers = {}
    for option in options:
        optimizer_name = option.pop('name')
        optimizer = get_optimizer(params, option)
        optimizers[optimizer_name] = optimizer
    return optimizers



def set_all_params_lr_to_zero(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0

def set_first_params_lr_to_zero(optimizer):
    optimizer.param_groups[0]['lr'] = 0