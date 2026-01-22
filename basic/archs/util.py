from itertools import repeat
import collections.abc
from copy import deepcopy
from typing import Optional, Callable, Union, List
import enum
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.utils.console.log import highlight_diff
from basic.archs.modules.feed_forward import SwiGLU, swiglu

'''
Modified from 
- SAM2(https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_utils.py)
- ConvNeXt(https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py)
'''


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


def zero_module(module: nn.Module) -> nn.Module:
    """Set all parameters in a module to zero. And return the modified module.

    Args:
        module: The module to modify.
    """
    for param in module.parameters():
        param.data.zero_()
    return module


def set_requires_grad(module: nn.Module, requires_grad: bool = True, exclude = None) -> None:
    """Set the `requires_grad` attribute of all parameters in a module.

    Args:
        module: The module to modify.
        requires_grad: The new value of `requires_grad`.
        exclude (nn.Module, list[nn.Module]): A list of modules to exclude from the `requires_grad` attribute.
    """
    # TODO: 添加排除模块的功能
    if exclude is not None and not isinstance(exclude, list):
        exclude: list = [exclude]

    assert exclude is None, "exclude is not supported yet."

    for param in module.parameters():
        # if exclude is not None and any(param in e.parameters() for e in exclude):
        #     continue
        param.requires_grad = requires_grad


@contextmanager
def no_grad_if(condition: bool=True):
    with torch.no_grad() if condition else nullcontext():
        yield


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self, *,
            input_dim: int,
            hidden_dim: Optional[int] = None,
            output_dim: Optional[int] = None,
            num_layers: int = 2,
            bias: bool =False,
            activation=nn.ReLU,
            sigmoid_output: bool = False,
            drop_out: float = 0.0,
    ):
        """
        Args:
            activation: [Callable[..., nn.Module, Activation, str]
        """
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        self.num_layers = num_layers

        hidden_layers = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k, bias=bias) for n, k in zip([input_dim] + hidden_layers, hidden_layers + [output_dim])
        )

        # 每层（除了最后一层）都有的激活函数
        self.sigmoid_output = sigmoid_output
        if isinstance(activation, str) or isinstance(activation, Activation):
            self.act = Activation.get_function(activation)
        else:
            self.act = activation()

        self.drop = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.drop(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Activation(enum.Enum):
    ReLU        = "relu"
    LeakyReLU   = "leaky_relu"
    ELU         = "elu"
    GELU        = "gelu"
    GLU         = "glu"
    SiLU        = "silu"
    SwiGLU      = "swi_glu"

    @staticmethod
    def get_function(activation):
        if isinstance(activation, str):
            activation = activation.lower()
            for a in Activation:
                if activation.lower() == a.value.lower() or activation.lower() == a.name.lower():
                    return Activation.get_function(a)
            raise ValueError(f"Invalid activation function. "
                             f"Allowed values are: {[a.name for a in Activation]}. "
                             f"Got {activation}")
        elif isinstance(activation, Activation):
            if activation == Activation.ReLU:
                return F.relu
            elif activation == Activation.LeakyReLU:
                return F.leaky_relu
            elif activation == Activation.ELU:
                return F.elu
            elif activation == Activation.GELU:
                return F.gelu
            elif activation == Activation.GLU:
                return F.glu
            elif activation == Activation.SiLU:
                return F.silu
            elif activation == Activation.SwiGLU:
                return swiglu
            else:
                raise NotImplementedError(f"Invalid activation function. "
                                          f"Allowed values are: {[a.name for a in Activation]}. ")
        else:
            raise ValueError(f"Invalid activation function parameter type. "
                             f"Allowed types are: str, Activation. "
                             f"Got {type(activation)}")

    @staticmethod
    def get_module(activation, **kwargs):
        if isinstance(activation, str):
            activation = activation.lower()
            for a in Activation:
                if activation.lower() == a.value.lower() or activation.lower() == a.name.lower():
                    return Activation.get_module(a)
            raise ValueError(f"Invalid activation function. "
                             f"Allowed values are: {[a.name for a in Activation]}. "
                             f"Got {activation}")
        elif isinstance(activation, Activation):
            if activation == Activation.ReLU:
                return nn.ReLU(**kwargs)
            elif activation == Activation.LeakyReLU:
                return nn.LeakyReLU(**kwargs)
            elif activation == Activation.ELU:
                return nn.ELU(**kwargs)
            elif activation == Activation.GELU:
                return nn.GELU(**kwargs)
            elif activation == Activation.GLU:
                return nn.GLU(**kwargs)
            elif activation == Activation.SiLU:
                return nn.SiLU(**kwargs)
            elif activation == Activation.SwiGLU:
                return SwiGLU(**kwargs)
            else:
                raise NotImplementedError(f"Invalid activation function. "
                                          f"Allowed values are: {[a.name for a in Activation]}. ")
        else:
            raise ValueError(f"Invalid activation function parameter type. "
                             f"Allowed types are: str, Activation. "
                             f"Got {type(activation)}")


#region ==[Tensor]==
def extract_ref(a, t, x):
    """
    Slice `a` according to `t` and flatten the result into shape=[batch_size, 1, 1, 1, ...], where the number of 1s is determined by `x.shape`.
    Args:
        a (torch.Tensor):
        t (torch.Tensor | int):
        x (torch.Tensor):
    """
    x_shape = x.shape
    if isinstance(t, torch.Tensor):
        out = a.gather(dim=-1, index=t)
        return out.view(x_shape[0], *([1] * (len(x_shape) - 1)))
    elif isinstance(t, int):
        out = a[t]
        return out.repeat(x_shape[0], *([1] * (len(x_shape) - 1)))
    else:
        raise ValueError("t must be int or tensor")


def get_expend_dims(as_x, at_dim, dims, default_value=-1):
    d = [default_value] * as_x.dim()
    d[at_dim] = dims
    return tuple(d)


def get_extract_dims(x, value=1):
    """
    Returns:
        tuple: A tuple of integers representing the dimensions to extract from `x`
        e.g. for a tensor of shape [batch_size, height, width, channels], the output would be (1, 1, 1)
    """
    return [value] * (len(x.shape) - 1)


def extract_as(a, x):
    """
    Expand `a` into shape=[batch_size, 1, 1, 1, ...], where the number of 1s is determined by `x.shape`.
    Args:
        a (torch.Tensor):
        x (torch.Tensor):
    """
    assert len(a.shape) == 1, f"a must be 1-D tensor, instead of {a.shape}"
    return a.view(x.shape[0], *((1,) * (len(x.shape) - 1)))


def expand_as(a, x):
    """
    Expand `a` into shape=[batch_size, *x.shape[1:]], where the number of 1s is determined by `x.shape`.
    Args:
        a (torch.Tensor):
        x (torch.Tensor):
    """
    return a.expand(*x.shape)
#endregion


def to_n_tuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2_tuple = to_n_tuple(2)
to_3_tuple = to_n_tuple(3)
to_4_tuple = to_n_tuple(4)