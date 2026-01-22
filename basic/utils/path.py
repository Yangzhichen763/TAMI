import os
import os.path as osp
from basic.utils.console.log import get_striped_time_str

from basic.utils.console.log import get_root_logger
logger = get_root_logger()


try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass


'''
Modified from BasicSR: https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/utils/misc.py
'''


root = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir, osp.pardir))


def mkdir_and_rename(path, rename=True):
    """
    mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if rename and osp.exists(path):
        new_name = f'{path}_archived_{get_striped_time_str()}'
        logger.info(f'Path already exists. Rename it to {new_name}')
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def make_exp_dirs(opt, rename=True):
    """
    Make dirs for experiments.
    """
    from basic.utils.dist import is_master
    if not is_master():
        return

    path_opt = opt['path'].copy()

    # 根据 rename 和是否已经有 directory，决定是否重命名，并创建新的文件夹
    def move_or_overwrite(root_name: str, directory_name: str):
        directory_path = osp.join(path_opt[root_name], directory_name)
        # 如果文件夹中拥有了冲突文件，则重命名旧的文件夹，创建新的文件夹用于存储新的实验结果
        opt['rename_flag'] = (
                rename
                and (osp.exists(directory_path)
                     and (len(os.listdir(directory_path)) != 0)
                     and not all(['val_G.pth' in file or 'val.state' in file for file in os.listdir(directory_path)]))
        )

        exp_root = path_opt.pop(root_name)
        mkdir_and_rename(exp_root, opt["rename_flag"])

    if opt['is_train']:
        move_or_overwrite('experiments_root', 'models')
    else:
        if 'results_root' not in path_opt:
            logger.warning('results_root is not specified in the options. '
                           'If is training phase, please set "is_train" to True.')
        else:
            move_or_overwrite('results_root', 'test_images')

    # 创建 path 下的所有文件夹（除了不是文件夹的）
    for key, path in path_opt.items():
        if isinstance(path, str) and '.' not in osp.basename(path):
            os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

