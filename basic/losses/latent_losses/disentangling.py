import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basic.utils.registry import LOSS_REGISTRY
from basic.losses.util import contrastive_weighted_loss, unsupervised_weighted_loss, _reduction_modes
from basic.losses.basic_loss import BaseLoss, scale_loss


@contrastive_weighted_loss
def orthogonal_loss(*deg_feats, centering=True):
    """
    Calculate orthogonal loss for degree features.

    Args:
        deg_feats (Tensor): List of degree feature tensors, each of shape (B, C, H, W).

    Returns:
        Tensor: Orthogonal loss value.
    """
    if len(deg_feats) == 1 and isinstance(deg_feats[0], list):
        deg_feats = deg_feats[0]
    deg_feats = torch.stack(deg_feats, dim=1)                   # (B, m, C, H, W)

    # centering
    if centering:
        deg_feats = deg_feats - deg_feats.mean(dim=(-2, -1), keepdim=True)

    # calculate cosine similarity
    deg_feats = rearrange(deg_feats, 'b n c h w -> b n (c h w)')        # (B, m, C, H, W) -> (B, m, N)
    deg_feats = F.normalize(deg_feats, p=2, dim=-1)
    deg_feats_mat = torch.einsum('bic,bjc->bij', deg_feats, deg_feats)  # (B, m, m)

    # orthogonal loss
    orthx_loss = deg_feats_mat - torch.eye(deg_feats_mat.size(1)).to(deg_feats.device)
    orthx_loss = orthx_loss ** 2

    return orthx_loss


@unsupervised_weighted_loss
def zero_loss(feat):
    """
    Calculate zero loss for a feature tensor.

    Args:
        feat (Tensor): Feature tensor of shape (B, C, H, W).

    Returns:
        Tensor: Zero loss value.
    """
    return feat ** 2


@LOSS_REGISTRY.register()
class OrthogonalLoss(BaseLoss):
    """
    orthogonal loss
    """

    def __init__(
            self,
            centering=True,
            loss_weight=1.0,
            reduction="mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.centering = centering

    @scale_loss
    def forward(self, *feats, weight=None, **kwargs):
        """
        Args:
            feats (Tensor): (B, C, H, W)
        """
        return orthogonal_loss(
            *feats,
            centering=self.centering,
            weight=weight,
        )


class ZeroLoss(BaseLoss):
    """
    zero loss
    """

    def __init__(
            self,
            loss_weight=1.0,
            reduction="mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

    @scale_loss
    def forward(self, feat, weight=None, **kwargs):
        """
        Args:
            feat (Tensor): (B, C, H, W)
        """
        return zero_loss(
            feat,
            weight=weight,
        )

