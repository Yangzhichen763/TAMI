from multiprocessing import current_process

import numpy as np


try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass


class SharedPool:
    # 不同的子进程有不同的 shared pool，否则同时使用时会出现干扰
    _process_pools = {}

    def __init__(self, process_id, name):
        self._process_id = process_id
        self._name = name
        self._obj_map = {}

    @classmethod
    def __class_getitem__(cls, pool_name) -> 'SharedPool':
        """Get a shared pool by name for the current thread."""
        return cls.get(pool_name)

    @staticmethod
    def get(pool_name) -> 'SharedPool':
        """
        Returns:
            pool: {
                'key_1': (value,)
                'key_2': (tensor_1, tensor_2)
                ...
            }
        """
        process = current_process()
        process_id = process.pid

        if process_id not in SharedPool._process_pools:
            SharedPool._process_pools[process_id] = {}

            # from basic.utils.log import get_root_logger
            # logger = get_root_logger()
            # logger.info(f'Create shared pool for process {process_id}')

        process_pool = SharedPool._process_pools[process_id]

        if pool_name not in process_pool:
            process_pool[pool_name] = SharedPool(process_id, pool_name)

        return process_pool[pool_name]

    def append(self, name, *obj):
        if name not in self._obj_map:
            self._obj_map[name] = []
        self._obj_map[name].append(obj)

    @staticmethod
    def _reduce(name, objs, reduction=None):
        # objs: (a1, b1), (a2, b2), ...
        # -> (a1+a2+..., b1+b2+...)
        if reduction is not None and reduction in ["mean", "sum"]:
            assert all(
                all(isinstance(x, (torch.Tensor, np.ndarray, float)) for x in obj)
                for obj in objs
            ), f"Reduction only supports tuples of torch.Tensor, but got {'; '.join(np.unique([', '.join([str(type(x)) for x in obj]) for obj in objs]))} for {name}"

        if isinstance(reduction, int):
            objs = objs[reduction]
        elif isinstance(reduction, str):
            if reduction == "mean":
                objs = list(zip(*objs))  # [(a1, b1, ...), (a2, b2, ...), ...] -> [(a1, a2, ...), (b1, b2, ...), ...]
                objs = tuple(
                    torch.mean(torch.stack(x, dim=0), dim=0) if isinstance(x[0], torch.Tensor) else np.mean(x, axis=0)
                    for x in objs
                )
            elif reduction == "sum":
                objs = list(zip(*objs))
                objs = tuple(
                    torch.sum(torch.stack(x, dim=0), dim=0) if isinstance(x[0], torch.Tensor) else np.sum(x, axis=0)
                    for x in objs
                )
            elif reduction == "last":
                objs = objs[-1]
            elif reduction == "first":
                objs = objs[0]
        return objs

    def try_pop(self, name):
        """
        Returns:
            (value,) or (tensor_1, tensor_2), a tuple value
        """
        if name not in self._obj_map:
            return None

        ret = self._obj_map[name].pop()
        if not self._obj_map[name]:
            del self._obj_map[name]

        return ret

    def try_pop_all(self, name, reduction=None):
        """
        Returns:
            (value,) or (tensor_1, tensor_2), a tuple value if reduction;
            else returns a list of tuple values.
        """
        if name not in self._obj_map:
            return None

        ret = self._reduce(name, self._obj_map[name], reduction)
        del self._obj_map[name]
        return ret

    def pop_all(self, reduction=None):
        for name, objs in self._obj_map.items():
            objs = self._reduce(name, objs, reduction)
            yield name, objs
        self._obj_map.clear()

    def try_get(self, name, reduction="last"):
        """
        Get the last object of the given name.
        If the name is not in the pool, return None.

        If append (name, a, b, c) to the pool, and then try_get(name), it will return (a, b, c).
        If append (name, d) to the pool, and then try_get(name), it will return (d,).

        Returns:
            tuple or None
        """
        if name not in self._obj_map:
            return None

        objs = self._reduce(name, self._obj_map[name], reduction)
        return objs

    def any(self, name):
        return name in self._obj_map

    def any_obj(self, name):
        return self.any(name) and len(self._obj_map[name]) > 0


    def clear(self, name):
        if name in self._obj_map:
            del self._obj_map[name]

    def clear_and_append(self, name, *obj):
        self.clear(name)
        self.append(name, *obj)

    def __contains__(self, name):
        return name in self._obj_map

    def keys(self):
        return self._obj_map.keys()

    @staticmethod
    def print_all():
        for p_id, shared_pool in SharedPool._process_pools.items():
            print(f"Process {p_id} - Pool {shared_pool._name}:")

            shared_pool.print()

    def print(self):
        def shape_to_str(shape):
            if len(shape) == 0:
                return "()"
            elif len(shape) == 1:
                return f"({shape[0]},)"
            else:
                return f"({', '.join([str(s) for s in shape])})"

        # _obj_map: dict[str, list[tuple]]
        for name, obj_list in self._obj_map.items():
            all_types = np.array([str(type(obj)) for obj in obj_list[0]])
            all_types = np.unique(all_types)
            type_log_str = ", ".join(all_types)
            log_str = f"{name}: {len(obj_list)} pairs w/ {len(obj_list[0])} items & type: {type_log_str}"

            import torch
            if isinstance(obj_list[0][0], np.ndarray) or isinstance(obj_list[0][0], torch.Tensor):
                all_shapes = [obj_list[i][0].shape for i in range(len(obj_list))]
                log_str += f" (shape: {', '.join([shape_to_str(shape) for shape in all_shapes])})"

            print(log_str)


# Example usage
if __name__ == "__main__":
    import torch
    pool = SharedPool.get("example_pool")
    pool.append("key1", 1, 2)
    pool.append("key1", 3, 4)
    pool.append("key2", "itemA1", "itemA2")
    pool.append("key2", "itemB1", "itemB2")
    pool.append("key2", "itemC1", "itemC2")
    pool.append("key3", torch.Tensor([1, 2, 3]))
    pool.append("key3", torch.Tensor([1, 2, 3, 4]))
    pool.append("key4", np.array([1, 2, 3]))

    pool.append("value", torch.Tensor([1, 2, 3]))
    pool.append("value", torch.Tensor([4, 5, 6]))
    pool.append("value", torch.Tensor([7, 8, 9]))
    value = pool.try_pop_all("value", reduction="mean")
    print(value)

    pool.print()

