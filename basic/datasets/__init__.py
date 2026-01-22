import numpy as np
import random
import os
from functools import partial
from copy import deepcopy

import torch
import torch.utils.data

from .prefetch_dataloader import PrefetchDataLoader
from .data_sampler import DistIterSampler
from .util import create_transforms
from basic.utils.dist import get_dist_info, is_dist_available, is_dist_initialized
from basic.utils.console.log import ColorPrefeb as CP

from basic.options.options import parse_params, parse_arguments
from basic.utils.registry import DATASET_REGISTRY, SAMPLER_REGISTRY


__all__ = ['create_dataset', 'create_dataloader']


from basic.utils.console.log import get_root_logger
logger = get_root_logger()


'''
Modified from Retinexformer(https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/data/__init__.py)
'''


# 将每个带有 _dataset 文件的模块都导入进来，同时会激活模块中的 register 装饰器，将 dataset 注册到 MODEL_REGISTRY 中。
# automatically scan and import dataset modules
# scan all the files under the 'datasets' folder and collect files ending with
# '_dataset.py'
import importlib
import os.path as osp
from basic.utils.path import scandir
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(dataset_folder, suffix='_dataset.py')
]
# import all the dataset modules
_dataset_modules = []
for file_name in dataset_filenames:
    try:
        _dataset_modules.append(
            importlib.import_module(f'basic.datasets.{file_name}')
        )
    except Exception as e:
        logger.warning(f'Failed to import {file_name} because of {e}')


def create_dataset(option):
    """Create dataset.

    Args:
        option (dict): Configuration for dataset. It must contain:
            - name (str): Dataset name.
            - type (str): Dataset type.
    """
    option = deepcopy(option)  # deepcopy the option to avoid modification

    dataset_type, dataset_params = parse_params(option)

    # dynamic instantiation
    dataset_class = DATASET_REGISTRY.get(dataset_type)
    for key, value in dataset_params.items():   # recursively instantiate all the sub-modules
        if key == 'transforms':
            dataset_params[key] = create_transforms(value)
    args, kwargs = parse_arguments(dataset_class, dataset_params)
    dataset = dataset_class(*args, **kwargs)

    logger.info(f'Dataset [{dataset.__class__.__name__}] - {CP.keyword(option["name"])} ({CP.number(len(dataset))} its) is created.')
    return dataset


def create_dataloader(
        dataset,
        dataset_option,
        num_gpu=1,
        dist=False,
        seed=None
):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_option (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_option['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:    # distributed training
            batch_size = dataset_option['batch_size_per_gpu']
            num_workers = dataset_option['num_workers_per_gpu']
        else:       # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_option['batch_size_per_gpu'] * multiplier
            num_workers = dataset_option['num_workers_per_gpu'] * multiplier
        shuffle = dataset_option.get('use_shuffle', True)
        num_workers = min(num_workers, os.cpu_count())

        # sampler
        sampler = None
        if 'sampler' in dataset_option:
            sampler_option = dataset_option['sampler']
            sampler = create_sampler(dataset, sampler_option, seed)
            if hasattr(sampler, 'shuffle') and sampler.shuffle is True:
                shuffle = False
        if sampler is None and not shuffle:
            from basic.utils.console.log import dict_to_str
            print(dict_to_str(dataset_option))
            logger.warning(f'The shuffle option is set to False, and the sampler is None. '
                           'This may cause the data loading to be consistent.')

        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            worker_init_fn=partial(
                    worker_init_fn, num_workers=num_workers, rank=rank, seed=seed
                ) if seed is not None else None
        )

    elif phase == 'test':
        # sampler
        sampler = None
        if 'sampler' in dataset_option:
            sampler_option = dataset_option['sampler']
            sampler = create_sampler(dataset, sampler_option, dist)

        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
        )

    elif phase == 'val':
        if 'num_workers_per_gpu' in dataset_option:
            num_workers = dataset_option['num_workers_per_gpu']
            num_workers = min(num_workers, os.cpu_count())
        else:
            logger.warning(f'`num_workers_per_gpu` is not found in {dataset_option}. '
                           f'The data loading latencies may be high. '
                           'Use default value 0.')
            num_workers = 0

        # sampler
        sampler = None
        if 'sampler' in dataset_option:
            sampler_option = dataset_option['sampler']
            sampler = create_sampler(dataset, sampler_option, dist)

        dataloader_args = dict(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers, sampler=sampler)

    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_option.get('pin_memory', False)

    prefetch_mode = dataset_option.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPU Pre-fetcher
        num_prefetch_queue = dataset_option.get('num_prefetch_queue', 1)
        logger.info(f'Use {prefetch_mode} prefetch dataloader: '
                    f'num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDA Pre-fetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def create_sampler(dataset, sampler_option, seed):
    """Create sampler.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        sampler_option (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_samples_per_gpu (int): Number of samples for each GPU.
    """
    sampler_option = deepcopy(sampler_option)  # deepcopy the option to avoid modification

    # distributed settings
    if not is_dist_available() or not is_dist_initialized():
        num_replicas, rank = 1, 0
    else:
        num_replicas, rank = get_dist_info()
    sampler_option.update(num_replicas=num_replicas)
    sampler_option.update(rank=rank)

    # dynamic instantiation
    sampler_type, sampler_params = parse_params(sampler_option)
    sampler_class = SAMPLER_REGISTRY.get(sampler_type)
    args, kwargs = parse_arguments(sampler_class, sampler_params)
    sampler = sampler_class(dataset, *args, **kwargs)

    return sampler


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
