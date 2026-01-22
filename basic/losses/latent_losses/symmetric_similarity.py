import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basic.utils.registry import LOSS_REGISTRY
from basic.losses.util import supervised_weighted_loss, _reduction_modes


def norm(x, eps=1e-12):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    x_norm = (x - mean) / (std + eps)
    return x_norm


@supervised_weighted_loss
def symmetric_cosine_similarity_loss(x, y, normalize=False):
    if normalize:
        x = norm(x)
        y = norm(y)

    x = rearrange(x, 'b c h w -> b (h w) c')
    y = rearrange(y, 'b c h w -> b (h w) c')

    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)(x.unsqueeze(2), y.unsqueeze(1))
    loss = nn.MSELoss()(cos_sim, cos_sim.transpose(-1, -2))
    return loss


@supervised_weighted_loss
def symmetric_dist_loss(x, y, p=2, normalize=False):
    if normalize:
        x = norm(x)
        y = norm(y)

    x = rearrange(x, 'b c h w -> b (h w) c')
    y = rearrange(y, 'b c h w -> b (h w) c')

    l2_distance = torch.cdist(x, y, p=p)
    loss = nn.MSELoss()(l2_distance, l2_distance.transpose(-1, -2))
    return loss


@LOSS_REGISTRY.register()
class SymmetricCosineSimilarityLoss(nn.Module):
    def __init__(
            self,
            normalize=False,
            loss_weight=1.0, reduction='mean',
    ):
        super(SymmetricCosineSimilarityLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.normalize = normalize

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * symmetric_cosine_similarity_loss(
            pred, target,
            normalize=self.normalize,
            weight=weight, reduction=self.reduction,
        )


@LOSS_REGISTRY.register()
class SymmetricDistLoss(nn.Module):
    def __init__(
            self,
            p=2, normalize=False,
            loss_weight=1.0, reduction='mean',
    ):
        super(SymmetricDistLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.p = p
        self.normalize = normalize

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * symmetric_dist_loss(
            pred, target,
            p=self.p, normalize=self.normalize,
            weight=weight, reduction=self.reduction,
        )