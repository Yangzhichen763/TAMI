import functools
import torch
import torch.nn as nn
from torch.nn import functional as F


_reduction_modes = ['none', 'mean', 'sum']


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    eps = 1e-6

    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(0) == 1 or weight.size(0) == loss.size(0)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(0) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(0)
        loss = loss.sum(0) / (weight + eps)
        loss = reduce_loss(loss, reduction)

    return loss


def supervised_weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @supervised_weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


def unsupervised_weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(inputs, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(inputs, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


def contrastive_weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(*args, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(*args, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


"""
Adapted from 
- LightningDiT (https://github.com/hustvl/LightningDiT/blob/main/vavae/ldm/modules/losses/contperceptual.py#L56-L67) 
- Taming Transformers (VQGAN) (https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L63-L74)
"""
def calculate_adaptive_weight(main_loss, aux_loss, last_layer_params):
    """
    Calculate the adaptive weight for the auxiliary loss.

    Usage:
        aux_weight = calculate_adaptive_weight(main_loss, aux_loss, last_layer)
        aux_loss = aux_weight * aux_loss

    Args:
        main_loss (Tensor): The main loss tensor.
        aux_loss (Tensor): The auxiliary loss tensor.
        last_layer_params (list): The last layer parameters of the model.

    Returns:
        Tensor: The adaptive weight for the auxiliary loss.
    """
    main_grads = torch.autograd.grad(main_loss, last_layer_params, retain_graph=True)[0]
    aux_grads = torch.autograd.grad(aux_loss, last_layer_params, retain_graph=True)[0]

    aux_weight = torch.norm(main_grads) / (torch.norm(aux_grads) + 1e-4)
    aux_weight = torch.clamp(aux_weight, 0.0, 1e8).detach()
    return aux_weight


def get_last_layer_params(model):
    """
    Get the last layer parameters of the model.

    Args:
        model (nn.Module): The model.

    Returns:
        list: The last layer parameters of the model.
    """
    for name, layer in reversed(list(model.named_modules())):
        params = list(layer.parameters())
        if len(params) > 0:
            return params
    return None


# 这个可学习损失包装类没啥用，反向传播的时候会把自己先降到最低
# class LearnableWeightedLoss(nn.Module):
#     def __init__(self, loss_func, weight_init=0.0):
#         super(LearnableWeightedLoss, self).__init__()
#         self.loss_func = loss_func
#         self.weight = nn.Parameter(torch.tensor(weight_init, dtype=torch.float32))
#
#     def forward(self, *args, **kwargs):
#         loss = self.loss_func(*args, **kwargs)
#         weighted_loss = loss * torch.sigmoid(self.weight)
#         return weighted_loss