from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from torch import Tensor
from einops import rearrange


class ScalableSoftmax(nn.Module):
    def __init__(self, *dims):
        """
        e.g. usually dims = (1, ) or (heads, 1)
        """
        super().__init__()
        self.dims = dims

        self.scale = nn.Parameter(torch.ones(dims))

    def forward(self, x, dim=-1):
        n = x.shape[dim]
        x = F.softmax(self.scale * torch.log(n) * x, dim=-1)
        return x


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = module(*inputs)
        return inputs


class AdaptiveInputModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.num_args = module.forward.__code__.co_argcount - 1

    def forward(self, *args, **kwargs):
        return self.module(*args[:self.num_args], **kwargs)


class CustomIdentity(nn.Identity):
    def __init__(self, forward_fn):
        super().__init__()

        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(x)


class NoneModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def register_function(self, name: str, function: callable) -> None:
        setattr(self, name, function)

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        pass

    def __getattr__(self, name):
        if name == "_module":
            return super().__getattr__(name)

        if self._module is not None and hasattr(self._module, name):
            attr = getattr(self._module, name)
            if callable(attr):
                # retain the signature of the original method
                from functools import wraps
                @wraps(attr)
                def method(*args, **kwargs):
                    return attr(*args, **kwargs)

                return method
            return attr
        if hasattr(super(), name):
            return super().__getattr__(name)
        return self.do_nothing

    def __deepcopy__(self, memo):
        return NoneModule(deepcopy(self._module, memo))

    def do_nothing(self, *args, **kwargs):
        return None

    def extra_repr(self):
        return f"module={self._module}"


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return rearrange(x, self.pattern)


class ShapeGetter(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x
