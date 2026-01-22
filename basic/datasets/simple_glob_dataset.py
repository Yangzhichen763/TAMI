import os.path as osp

import random

import torch
import torch.utils.data as data

import basic.utils.io as io
from basic.utils.convert import numpy2tensor
from basic.utils.registry import DATASET_REGISTRY
from basic.utils.general import try_fill_default_dict, obsolete

from .base_dataset import TransformsDatasetBase
from .util import generate_random_indices_in_a_sequence


def read_image_to_tensor(file_path, transform=None, seed=None):
    """
    Reads image file to tensor.

    The illustration below shows the order of the random state applied:
    if seed is not None:
        gt:[b], ..., [a], [c]
        lq:[b], ..., [a], [c]
        (a, b, c represents the different transforms applied to the image)
    """
    if seed is not None:
        random.seed(seed)           # apply this seed to img transforms
        torch.manual_seed(seed)     # needed for torchvision 0.7

    if transform is not None:
        image_pil = io.read_image_as_pil(file_path)
        image_tensor = transform(image_pil)
    else:
        image_np = io.read_image_as_numpy(file_path)
        image_tensor = numpy2tensor(image_np, expand_dim=False)
    return image_tensor


def read_images_to_tensor(file_paths, transform=None, seed=None):
    """
    Reads image sequence to tensor.

    The illustration below shows the order of the random state applied:
    if seed is not None:
        seq_gt:[b] -> [e] -> [a] -> [d] -> [c], ..., [i] -> [g] -> [j] -> [f] -> [h]
        seq_lq:[b] -> [e] -> [a] -> [d] -> [c], ..., [i] -> [g] -> [j] -> [f] -> [h]
        (a, b, ..., j represents the different transforms applied to the image)
    """
    if seed is not None:
        random.seed(seed)           # apply this seed to img transforms
        torch.manual_seed(seed)     # needed for torchvision 0.7

    tensors = []
    for file_path in file_paths:
        image_tensor = read_image_to_tensor(file_path, transform)
        tensors.append(image_tensor)
    return torch.stack(tensors)


def read_seq_images_to_tensor(file_paths, transform=None, seed=None):
    """
    Reads image sequence to tensor.

    The illustration below shows the order of the random state applied:
    if seed is not None:
        seq_gt:[a] -> [a] -> [a], ..., [c] -> [c] -> [c], [b] -> [b] -> [b]
        seq_lq:[a] -> [a] -> [a], ..., [c] -> [c] -> [c], [b] -> [b] -> [b]
        (a, b, c represents the different transforms applied to the image)
    """
    tensors = []
    for file_path in file_paths:
        image_tensor = read_image_to_tensor(file_path, transform, seed)
        tensors.append(image_tensor)
    return torch.stack(tensors)


# noinspection SpellCheckingInspection
@obsolete
class GlobSingleDatasetBase(data.Dataset):
    def __init__(self, files_glob_func=None):
        super(GlobSingleDatasetBase, self).__init__()
        self.files_glob_func = files_glob_func
        self.names = []
        self.common_dirs = {}

    def get_all_file_paths(self, **kwargs):
        dir_names = []
        attr_names = []
        name_prefixs = []
        if "dataroot" in kwargs:
            # dataroot, xx_dir, xx_ext
            for key, value in kwargs.items():
                if key.endswith("_dir") and value is not None:
                    name_prefix = key[:-len("_dir")]

                    dir_name = f"{name_prefix}_dir"
                    ext_name = f"{name_prefix}_ext"
                    attr_name = f"{name_prefix}_file_paths"
                    if ext_name not in kwargs:
                        ext_name = None
                    self.dir_attr(attr_name, dir_name, ext_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)
        else:
            # dataroot_xx
            for key, value in kwargs.items():
                if key == "dataroot":
                    continue
                if key.startswith("dataroot_") and value is not None:
                    name_prefix = key[len("dataroot_"):]

                    dir_name = f"dataroot_{name_prefix}"  # 所有图片的文件夹路径
                    attr_name = f"{name_prefix}_file_paths"  # 在 Dataset 类中的属性名称，用来记录所有图片的文件路径
                    self.dir_attr(attr_name, dir_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)

        # 检查文件对应性
        if len(attr_names) > 0:
            def get_log_str(_attr_names, _dir_names):
                return ', '.join([f"{attr_name}({dir_name})" for attr_name, dir_name in zip(_attr_names, _dir_names)])
            # 检查文件数据量是否为 0
            if any(len(getattr(self, attr_name)) == 0 for attr_name in attr_names):
                empty_attr_names = [attr_name for attr_name in attr_names if len(getattr(self, attr_name)) == 0]
                empty_dir_names = [dir_name for dir_name in dir_names if dir_name in empty_attr_names]
                raise ValueError(f"No image files found in {get_log_str(empty_attr_names, empty_dir_names)}")
            # 检查文件数量是否不同
            if len({len(getattr(self, attr_name)) for attr_name in attr_names}) != 1:
                raise ValueError(f"The number of image files in {get_log_str(attr_names, dir_names)} are not the same")
            # 检查所有的 self.xx_file_paths 中的图片是否都一一对应
            attr_lists = [getattr(self, n) for n in attr_names]
            for i, group in enumerate(zip(*attr_lists)):
                basenames = [osp.basename(p) for p in group]
                if len(set(basenames)) != 1:
                    raise ValueError(f"Files do not correspond at index {i}: {basenames}")


        # 保存属性名称
        self.names = name_prefixs
        self.check_file_paths()

        self.common_dirs = {
            name: self.get_file_paths_common_dir(name)
            for name in self.names
        }

    def dir_attr(self, attr_name, dir_key, ext_key=None, **kwargs):
        if dir_key not in kwargs or (ext_key is not None and ext_key not in kwargs):
            return

        root_dir = ""
        if 'dataroot' in kwargs:
            root_dir = kwargs['dataroot']   # 如果在配置文件中有指定文件的根目录，就使用根目录

        dir_path = osp.join(root_dir, kwargs[dir_key])
        assert isinstance(dir_path, str), f"{dir_key} should be a string, but got {type(dir_path)}"
        # 确保路径是有效的
        if not osp.exists(dir_path):
            any_found = False

            if not any_found:
                # simple prefix check
                for prefix in ["/mnt/", "/data/", "/home/"]:
                    if dir_path.startswith(prefix):
                        for _prefix in ["/mnt/", "/data/", "/home/"]:
                            if prefix == _prefix:
                                continue

                            _dir_path = dir_path.replace(prefix, _prefix, 1)
                            if osp.exists(_dir_path):

                                from basic.utils.console.log import get_root_logger
                                logger = get_root_logger()
                                logger.warning(f"The directory {dir_path} does not exist, but found {_dir_path} instead.")
                                dir_path = _dir_path

                                any_found = True
                                break
                        break

            if not any_found:
                raise ValueError(f"The directory {dir_path} does not exist.")

        assert osp.exists(dir_path), f"{dir_path} does not exist"
        assert osp.isdir(dir_path), f"{dir_path} is not a directory"
        self.__setattr__(dir_key, dir_path)

        exts = kwargs[ext_key] if ext_key is not None in kwargs else io.IMG_EXTENSIONS
        if isinstance(exts, str):
            exts = [exts]
        elif not isinstance(exts, (list, tuple)):
            raise ValueError(f"{ext_key} should be a string or a list or tuple of extensions")
        if isinstance(exts, (list, tuple)) and len(exts) <= 0:
            raise ValueError(f"{ext_key} should be a non-empty list or tuple of extensions")

        dirs = self.files_glob_func(dir_path, exts) # 所有图片的文件路径
        self.__setattr__(attr_name, dirs)


    def get_file_paths(self, name, without_notified=True):
        """
        Returns:
            list[str]: A list of file paths.
        """
        file_paths_name = f"{name}_file_paths"
        if not without_notified:
            if not hasattr(self, file_paths_name):
                raise ValueError(f"{file_paths_name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")
        return getattr(self, file_paths_name)


    def get_file_paths_case_insensitive(self, name, without_notified=True):
        names = [name.lower(), name.upper(), name.capitalize()]
        for name in names:
            file_paths_name = f"{name}_file_paths"
            if hasattr(self, file_paths_name):
                return getattr(self, file_paths_name)
        if not without_notified:
            raise ValueError(f"{name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")


    def check_file_paths(self):
        if len(self.names) == 0:
            raise ValueError(
                "No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. "
                "Or check the path of the dataset."
            )


    def get_file_paths_common_dir(self, name, without_notified=True):
        file_paths = self.get_file_paths(name, without_notified)
        if len(file_paths) == 0:
            return "/"

        def flatten(_list):
            result = []
            for item in _list:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        common_dir = osp.commonpath(flatten(file_paths))
        return common_dir


# noinspection SpellCheckingInspection
class GlobDatasetBase(data.Dataset):
    def __init__(self, files_glob_func=None):
        super(GlobDatasetBase, self).__init__()
        self.files_glob_func = files_glob_func
        self.names = []
        self.common_dirs = {}

    def get_all_file_paths(self, **kwargs):
        dir_names = []
        attr_names = []
        name_prefixs = []
        if "dataroot" in kwargs:
            # dataroot, xx_dir, xx_ext
            for key, value in kwargs.items():
                if key.endswith("_dir") and value is not None:
                    name_prefix = key[:-len("_dir")]

                    dir_name = f"{name_prefix}_dir"
                    ext_name = f"{name_prefix}_ext"
                    attr_name = f"{name_prefix}_file_paths"
                    if ext_name not in kwargs:
                        ext_name = None
                    self.dir_attr(attr_name, dir_name, ext_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)
        else:
            # dataroot_xx
            for key, value in kwargs.items():
                if key == "dataroot":
                    continue
                if key.startswith("dataroot_") and value is not None:
                    name_prefix = key[len("dataroot_"):]

                    dir_name = f"dataroot_{name_prefix}"  # 所有图片的文件夹路径
                    attr_name = f"{name_prefix}_file_paths"  # 在 Dataset 类中的属性名称，用来记录所有图片的文件路径
                    self.dir_attr(attr_name, dir_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)

        # 检查文件对应性
        if len(attr_names) > 0:
            def get_log_str(_attr_names, _dir_names):
                return ', '.join([f"{attr_name}({dir_name})" for attr_name, dir_name in zip(_attr_names, _dir_names)])

            # 检查文件数据量是否为 0
            if any(len(getattr(self, attr_name)) == 0 for attr_name in attr_names):
                empty_attr_names = [attr_name for attr_name in attr_names if len(getattr(self, attr_name)) == 0]
                empty_dir_names = [dir_name for dir_name in dir_names if dir_name in empty_attr_names]
                raise ValueError(f"No image files found in {get_log_str(empty_attr_names, empty_dir_names)}")
            # 检查文件数量是否不同
            if len(set(len(getattr(self, attr_name)) for attr_name in attr_names)) != 1:
                raise ValueError(f"The number of image files in {get_log_str(attr_names, dir_names)} are not the same")
            # 检查所有的 self.xx_file_paths 中的图片是否都一一对应
            def move_first_dim_to_last(data, index_prefix=()):
                """
                data: shape [k, a, b, ...]
                yield: (index, [obj0, obj1, ..., objk])
                """
                if not isinstance(data[0], list):
                    yield index_prefix, data
                    return
                for i, sub_items in enumerate(zip(*data)):
                    yield from move_first_dim_to_last(
                        list(sub_items),
                        index_prefix + (i,)
                    )

            attr_lists = [getattr(self, n) for n in attr_names]
            for idx, group in move_first_dim_to_last(attr_lists):
                basenames = [osp.basename(p) for p in group]
                if len(set(basenames)) != 1:
                    raise ValueError(f"Files do not correspond at index {idx}: {basenames}")

        # 保存属性名称
        self.names = name_prefixs
        self.check_file_paths()

        self.common_dirs = {
            name: self.get_file_paths_common_dir(name)
            for name in self.names
        }

    def dir_attr(self, attr_name, dir_key, ext_key=None, **kwargs):
        if dir_key not in kwargs or (ext_key is not None and ext_key not in kwargs):
            return

        root_dir = ""
        if 'dataroot' in kwargs:
            root_dir = kwargs['dataroot']  # 如果在配置文件中有指定文件的根目录，就使用根目录

        # ==[多数据集支持]==
        if isinstance(kwargs[dir_key], list):
            dir_base_paths = kwargs[dir_key]
        else:
            dir_base_paths = [kwargs[dir_key]]

        dir_paths = []
        all_file_paths = []
        for dir_base_path in dir_base_paths:
            dir_path = osp.join(root_dir, dir_base_path)
            assert isinstance(dir_path, str), f"{dir_key} should be a string, but got {type(dir_path)}"
            # 确保路径是有效的
            if not osp.exists(dir_path):
                any_found = False

                # simple prefix check
                if not any_found:
                    for prefix in ["/mnt/", "/data/", "/home/"]:
                        if dir_path.startswith(prefix):
                            for _prefix in ["/mnt/", "/data/", "/home/"]:
                                if prefix == _prefix:
                                    continue

                                _dir_path = dir_path.replace(prefix, _prefix, 1)
                                if osp.exists(_dir_path):
                                    from basic.utils.console.log import get_root_logger
                                    logger = get_root_logger()
                                    logger.warning(f"The directory {dir_path} does not exist, but found {_dir_path} instead.")
                                    dir_path = _dir_path

                                    any_found = True
                                    break
                            break

                if not any_found:
                    raise ValueError(f"The directory {dir_path} does not exist.")

            assert osp.exists(dir_path), f"{dir_path} does not exist"
            assert osp.isdir(dir_path), f"{dir_path} is not a directory"
            dir_paths.append(dir_path)

            exts = kwargs[ext_key] if ext_key is not None in kwargs else io.IMG_EXTENSIONS
            if isinstance(exts, str):
                exts = [exts]
            elif not isinstance(exts, (list, tuple)):
                raise ValueError(f"{ext_key} should be a string or a list or tuple of extensions")
            if isinstance(exts, (list, tuple)) and len(exts) <= 0:
                raise ValueError(f"{ext_key} should be a non-empty list or tuple of extensions")

            file_paths = self.files_glob_func(dir_path, exts)  # 所有图片的文件路径
            all_file_paths.extend(file_paths)

        self.__setattr__(dir_key, dir_paths)
        self.__setattr__(attr_name, all_file_paths)

    def get_file_paths(self, name, without_notified=True):
        """
        Returns:
            list[str]: A list of file paths.
        """
        file_paths_name = f"{name}_file_paths"
        if not without_notified:
            if not hasattr(self, file_paths_name):
                raise ValueError(
                    f"{file_paths_name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")
        return getattr(self, file_paths_name)

    def get_file_paths_case_insensitive(self, name, without_notified=True):
        names = [name.lower(), name.upper(), name.capitalize()]
        for name in names:
            file_paths_name = f"{name}_file_paths"
            if hasattr(self, file_paths_name):
                return getattr(self, file_paths_name)
        if not without_notified:
            raise ValueError(
                f"{name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")

    def check_file_paths(self):
        if len(self.names) == 0:
            raise ValueError(
                "No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. "
                "Or check the path of the dataset."
            )

    def get_file_paths_common_dir(self, name, without_notified=True):
        file_paths = self.get_file_paths(name, without_notified)
        if len(file_paths) == 0:
            return "/"

        def flatten(_list):
            result = []
            for item in _list:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        common_dir = osp.commonpath(flatten(file_paths))
        return common_dir


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class ImageDataset(GlobDatasetBase, TransformsDatasetBase):
    def __init__(self, **option):
        """
        Dataset for image-to-image tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., but not allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        GlobDatasetBase.__init__(self, io.glob_single_files)
        TransformsDatasetBase.__init__(self, **option)
        self.get_all_file_paths(**option)


    def __len__(self):
        return len(self.get_file_paths_case_insensitive(self.names[0], without_notified=False))


    def __getitem__(self, index):
        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            file_paths = self.get_file_paths_case_insensitive(name)[index]
            file = read_image_to_tensor(file_paths, self.transforms)
            datas[name] = {
                "image": file,
                "path": file_paths,
                "common_dir": self.common_dirs[name],
            }
            return datas

        raise ValueError("No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. Or check the path of the dataset.")


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class PairedImageDataset(ImageDataset):
    def __init__(self, **option):
        """
        Dataset for image-to-image tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        super(PairedImageDataset, self).__init__(**option)

    def __getitem__(self, index):
        """
        dataset = PairedImageDataset(dataroot_pred=folder1, dataroot_gt=folder2)
        for data i dataset:
            pred, gt = data['pred']['image'], data['gt']['image']
            path = data['pred']['path'], data['gt']['path']
            common_dir = data['pred']['common_dir'], data['gt']['common_dir']
        """
        seed = random.randint(1, 2**32) # ensure the same transform for paired images

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            file_paths = self.get_file_paths_case_insensitive(name)[index]
            file = read_image_to_tensor(file_paths, self.transforms, seed)
            datas[name] = {
                "image": file,
                "path": file_paths,
                "common_dir": self.common_dirs[name],
            }
        return datas


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class VideoDataset(GlobDatasetBase, TransformsDatasetBase):
    def __init__(self, pad_s=True, **option):
        """
        Dataset for video-to-video tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        GlobDatasetBase.__init__(self, io.glob_packed_files)
        TransformsDatasetBase.__init__(self, **option)
        self.get_all_file_paths(**option)
        self.random_clip = try_fill_default_dict(
            option.get('random_clip', None),
            seq_length=30,
        )

        self.pad_s = pad_s


    def __len__(self):
        return len(self.get_file_paths_case_insensitive(self.names[0], without_notified=False))

    def __getitem__(self, index):
        total_seq_length = len(self.get_file_paths_case_insensitive(self.names[0])[index])
        if self.random_clip:
            if isinstance(self.random_clip['seq_length'], tuple):
                seq_length = random.randint(*self.random_clip['seq_length'])
            elif isinstance(self.random_clip['seq_length'], list):
                seq_length = random.choice(self.random_clip['seq_length'])
            else:
                seq_length = self.random_clip['seq_length']
            padding_mode = self.random_clip.get('padding_mode', 'reflect')
            indices = generate_random_indices_in_a_sequence(total_seq_length, seq_length, padding_mode=padding_mode)
        else:
            indices = list(range(total_seq_length))

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            key_name = name
            if self.pad_s:
                key_name = f"{name}s"   # 'lqs', 'gts', etc.

            frame_paths = self.get_file_paths_case_insensitive(name)[index]
            frame_paths = [frame_paths[i] for i in indices]
            frames = read_seq_images_to_tensor(frame_paths, self.transforms)
            datas[key_name] = {
                "images": frames,
                "paths": frame_paths,
                "common_dir": self.common_dirs[name],
            }
            return data

        raise ValueError("No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config.")

    def reprepare_data(self):
        pass


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class PairedVideoDataset(VideoDataset):
    def __init__(self, pad_s=True, **option):
        """
        Dataset for video-to-video tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        super(PairedVideoDataset, self).__init__(pad_s=pad_s, **option)

    def __getitem__(self, index):
        seed = random.randint(1, 2**32) # ensure the same transform for paired images

        total_seq_length = len(self.get_file_paths_case_insensitive(self.names[0])[index])
        if self.random_clip:
            if isinstance(self.random_clip['seq_length'], tuple):
                seq_length = random.randint(*self.random_clip['seq_length'])
            elif isinstance(self.random_clip['seq_length'], list):
                seq_length = random.choice(self.random_clip['seq_length'])
            else:
                seq_length = self.random_clip['seq_length']
            padding_mode = self.random_clip.get('padding_mode', 'reflect')
            indices = generate_random_indices_in_a_sequence(total_seq_length, seq_length, padding_mode=padding_mode)
        else:
            indices = list(range(total_seq_length))

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            key_name = name
            if self.pad_s:
                key_name = f"{name}s"   # 'lqs', 'gts', etc.

            frame_paths = self.get_file_paths_case_insensitive(name)[index]
            frame_paths = [frame_paths[i] for i in indices]
            frames = read_seq_images_to_tensor(frame_paths, self.transforms, seed)
            datas[key_name] = {
                "images": frames,   # [N, C, H, W]
                "paths": frame_paths,
                "common_dir": self.common_dirs[name],
            }
        return datas


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class DynamicVideoDataset(GlobDatasetBase, TransformsDatasetBase):
    def __init__(self, **option):
        """
        Dataset for video-to-video tasks. Instead of loading all frames into memory, it loads only one frame at a time.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        GlobDatasetBase.__init__(self, io.glob_packed_files)
        TransformsDatasetBase.__init__(self, **option)
        self.get_all_file_paths(**option)

        ### preparation
        frames_paths = None
        for name in self.names: # 'lq', 'gt', etc.
            frames_paths = self.get_file_paths_case_insensitive(name)
            break
        assert frames_paths is not None, (
            "No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. "
            "Or check the path of the dataset."
        )
        self.frames_paths = frames_paths

        self.video_count = len(self.frames_paths)
        self.frame_counts = [len(frame_paths) for frame_paths in self.frames_paths]
        self.all_frame_count = sum(len(frame_paths) for frame_paths in self.frames_paths)

        self.seeds = []
        self.reprepare_data()

    def __len__(self):
        return self.all_frame_count

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            raise ValueError(
                "Index should be a tuple of (video_index, frame_index, end)."
                "Or use VideoClipSampler in the dataloader."
                f"Got {index} with type {type(index)}."
            )

        video_index, frame_index, end = index
        seed = self.seeds[video_index] # ensure the same transform for paired and video-same frames

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            frame_path = self.get_file_paths_case_insensitive(name)[video_index][frame_index]
            frame = read_image_to_tensor(frame_path, self.transforms, seed)
            datas[name] = {
                "image": frame,
                "path": frame_path,
                "common_dir": self.common_dirs[name],
            }
            datas["end"] = end,
            return datas
        raise ValueError("No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config.")

    def reprepare_data(self):
        ### the seeds for each video (used for random transform)
        self.seeds = [random.randint(1, 2 ** 32) for _ in range(len(self.frames_paths))]


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class PairedDynamicVideoDataset(DynamicVideoDataset):
    def __init__(self, **option):
        """
        Dataset for video-to-video tasks. Instead of loading all frames into memory, it loads only one frame at a time.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        super(PairedDynamicVideoDataset, self).__init__(**option)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            raise ValueError(
                f"Index should be a tuple of (video_index, frame_index, end)."
                f"Or use VideoClipSampler in the dataloader."
                f"Got {index} with type {type(index)}."
            )

        video_index, frame_index, end = index
        seed = self.seeds[video_index] # ensure the same transform for paired and video-same frames

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            frame_path = self.get_file_paths_case_insensitive(name)[video_index][frame_index]
            frame = read_image_to_tensor(frame_path, self.transforms, seed)
            datas[name] = {
                "image": frame,
                "path": frame_path,
                "common_dir": self.common_dirs[name],
            }
        datas["end"] = end,
        return datas



if __name__ == '__main__':
    image_dataset = PairedImageDataset(dataroot_gt='/path/to/Datasets/LLIE/LOL-v1/eval15/high', dataroot_lq='/path/to/Datasets/LLIE/LOL-v1/eval15/low')
    print(len(image_dataset))

    image_dataset = PairedImageDataset(dataroot='/path/to/Datasets/LLIE/LOL-v1/eval15', gt_dir='high', lq_dir='low')
    print(len(image_dataset))

    image_dataset = ImageDataset(dataroot_gt='/path/to/Datasets/LLIE/LOL-v1/eval15/high')
    print(len(image_dataset))

    image_dataset = ImageDataset(dataroot='/path/to/Datasets/LLIE/LOL-v1/eval15', gt_dir='high')
    print(len(image_dataset))