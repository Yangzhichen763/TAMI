import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basic.utils.registry import LOSS_REGISTRY

from basic.losses.util import contrastive_weighted_loss, _reduction_modes


# KoLeo 正则化损失，用于鼓励 embedding 在空间中分散
@contrastive_weighted_loss
def koleo_loss(embeddings, p: float = 2.0, eps: float = 1e-6):
    """
    Compute the KoLeo (Kozachenko–Leonenko) regularization loss.

    This loss encourages the embeddings to be spread apart in the feature space
    by penalizing small nearest-neighbor distances. It is typically used as a
    regularizer to prevent feature collapse in self-supervised or contrastive learning.

    Args:
        embeddings (Tensor): Input tensor of shape (B, D), where
            B is the batch size and D is the feature dimension.
        p (float): Norm degree used to compute pairwise distances. Default: 2.0 (Euclidean).
        eps (float): Small constant added for numerical stability. Default: 1e-6.

    Returns:
        Tensor: KoLeo loss of shape (B,) or scalar depending on reduction.
    """
    assert embeddings.dim() == 2, "Input embeddings must be 2D (B, D)"
    B = embeddings.shape[0]
    if B <= 1:
        return embeddings.new_tensor(0.0)

    # Compute pairwise distances between embeddings
    dist = torch.cdist(embeddings, embeddings, p=p)

    # Mask out self-distances (diagonal entries = 0)
    mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
    dist = dist.masked_fill(mask, float('inf'))

    # Find the nearest-neighbor distance for each sample
    nearest_dist, _ = dist.min(dim=1)  # (B,)

    # KoLeo loss: -log(nearest_dist + eps)
    loss = -torch.log(nearest_dist + eps)
    return loss


@LOSS_REGISTRY.register()
class KoLeoLoss(nn.Module):
    """KoLeo regularization loss module.

    This module applies the KoLeo loss to encourage embeddings to occupy
    a larger volume of the embedding space by penalizing small pairwise distances.
    It helps reduce representation collapse in unsupervised learning.

    Args:
        loss_weight (float): Weight multiplier for the loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
        p (float): Norm degree for distance computation. Default: 2.0.
        eps (float): Small constant for numerical stability. Default: 1e-8.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', p=2.0, eps=1e-8):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. '
                f'Supported ones are: {_reduction_modes}'
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.p = p
        self.eps = eps

    def forward(self, inputs, weight=None, **kwargs):
        """
        Args:
            inputs (Tensor): Embeddings of shape (B, D).
            weight (Tensor, optional): Unused parameter, kept for API consistency.
        Returns:
            Tensor: Weighted KoLeo loss value.
        """
        return self.loss_weight * koleo_loss(
            inputs, p=self.p, eps=self.eps, weight=weight, reduction=self.reduction
        )