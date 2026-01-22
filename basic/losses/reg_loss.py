import torch
from torch import nn as nn
from torch.nn import functional as F

from basic.utils.registry import LOSS_REGISTRY

from .util import unsupervised_weighted_loss, _reduction_modes


@unsupervised_weighted_loss
def l1_reg_loss(target):
    return F.l1_loss(torch.zeros_like(target), target)


@unsupervised_weighted_loss
def l2_reg_loss(target):
    return F.mse_loss(torch.zeros_like(target), target) / 2


@LOSS_REGISTRY.register()
class RegL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(RegL1Loss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, net: nn.Module, weight=None, **kwargs):
        """
        Args:
            net (nn.Module): Network to regularize.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        nets = [net] if isinstance(net, nn.Module) else net
        device = next(nets[0].parameters()).device

        losses = torch.scalar_tensor(0.0, device=device)
        for net in nets:
            for name, param in net.named_parameters():
                if 'weight' in name:
                    loss = l1_reg_loss(param, weight=weight, reduction=self.reduction)
                    losses += loss
        return self.loss_weight * losses


@LOSS_REGISTRY.register()
class RegL2Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(RegL2Loss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, net, weight=None, **kwargs):
        """
        Args:
            net (list of nn.Module or nn.Module): Network to regularize.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        nets = [net] if isinstance(net, nn.Module) else net
        device = next(nets[0].parameters()).device

        losses = torch.scalar_tensor(0.0, device=device)
        for net in nets:
            for name, param in net.named_parameters():
                if 'weight' in name:
                    loss = l2_reg_loss(param, weight=weight, reduction=self.reduction)
                    losses += loss
        return self.loss_weight * losses
