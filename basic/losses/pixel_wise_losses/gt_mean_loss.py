import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from basic.losses.basic_loss import BaseLoss, l1_loss
from basic.utils.registry import LOSS_REGISTRY


"""
Adapted from GT-mean Loss (https://github.com/jingxiLiao/GT-mean-loss/blob/main/basicsr/models/losses/losses.py)
"""


@LOSS_REGISTRY.register()
class GTMeanLoss(BaseLoss):
    """GTMean loss.

    Args:
        loss_weight (float): Loss weight for GTMean loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight='dou_KL_2', reduction='mean', sigma=0.1):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.loss_weight_unclip = None
        self.sigma = sigma
        if loss_weight == 'dou_KL_2':
            self.iter_weight = self.double_KL_div_2
        else:
            raise ValueError(f'Currently weight is undefined for {loss_weight}')

        self.transform = transforms.Grayscale(num_output_channels=1)

    @staticmethod
    def KL_div(mu_1, mu_2, sigma_1, sigma_2, eps=1e-6):
        return torch.log((sigma_2 + eps) / (sigma_1 + eps)) + 0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2 ** 2 + eps) - 0.5

    def double_KL_div_2(self, mu_1, mu_2, sigma):
        sigma_1 = sigma * mu_1
        sigma_2 = sigma * mu_2
        mu_M = 0.5 * (mu_1 + mu_2)
        sigma_M = torch.sqrt((sigma_1 ** 2 + sigma_2 ** 2) / 2)
        KL_2_weight = 0.5 * self.KL_div(mu_1, mu_M, sigma_1, sigma_M) + 0.5 * self.KL_div(mu_2, mu_M, sigma_2, sigma_M)

        return KL_2_weight

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        E_y = torch.mean(self.transform(target), dim=(1, 2, 3))
        E_x = torch.mean(self.transform(pred), dim=(1, 2, 3))

        self.loss_weight_unclip = self.iter_weight(torch.abs(E_y), torch.abs(E_x), self.sigma)
        self.loss_weight = torch.clip(self.loss_weight_unclip, 0, 1).detach()

        eps = 1e-6
        m = E_y / (E_x + eps)

        pred_clip = torch.clip(m[:, None, None, None] * pred, 0, 1)

        L1_loss = l1_loss(pred, target, weight=self.loss_weight[:, None, None, None], reduction=self.reduction)
        GT_loss = l1_loss(pred_clip, target, weight=(1 - self.loss_weight)[:, None, None, None], reduction=self.reduction)

        total_loss = GT_loss + L1_loss

        return total_loss.mean()
