
import torch
from torch import nn as nn
from torch.nn import functional as F

from basic.losses.basic_loss import mse_loss, BaseLoss
from basic.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class EdgeLoss(BaseLoss):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).cuda()

        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def loss_func(self, x, y, weight=None, **kwargs):
        assert weight is None, 'EdgeLoss does not support weight'

        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight

