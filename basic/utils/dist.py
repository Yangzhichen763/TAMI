import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from basic.utils.console.log import get_root_logger


try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass


'''
Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
'''


#region ==[Initialization]==
def init_dist(launcher, backend='nccl', **kwargs):
    logger = get_root_logger()

    # 设置全局共享策略，防止出现文件共享错误
    if mp.get_start_method(allow_none=True) is None:
        # Start method hasn't been set yet, we can set it to 'spawn'
        # 如果导入了 multiprocessing，则 start method 很有可能被设置为了 folk
        logger.info(f"FileSystem - Change start method from {CP.keyword(mp.get_start_method(allow_none=True))} to {CP.keyword('spawn')}.")
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        logger.warning(
            f"FileSystem - Cannot change start method from {CP.keyword(mp.get_start_method())} to {CP.keyword('spawn')} "
            "because it has already been set. Some functionality may not work as expected."
        )
    mp.set_sharing_strategy('file_system')
    logger.info(f"FileSystem - Set sharing strategy to {CP.keyword('file_system')}.")

    if launcher == 'none' or launcher is None:
        logger.info(f"Distributed - Distributed init on {CP.keyword('localhost')}.")
        return

    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)
#endregion

#region ==[Information]==
def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_dist_available():
    return dist.is_available()


def is_dist_initialized():
    return dist.is_initialized()


def is_master():
    rank, _ = get_dist_info()
    return rank == 0
#endregion


#region ==[Wrapper]==
def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)
    return wrapper
#endregion
