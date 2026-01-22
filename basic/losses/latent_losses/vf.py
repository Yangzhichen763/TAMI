import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basic.utils.registry import LOSS_REGISTRY
from basic.losses.util import supervised_weighted_loss, _reduction_modes
from basic.losses.basic_loss import BaseLoss, scale_loss


"""
Adapted from LightningDiT(https://github.com/hustvl/LightningDiT/blob/main/vavae/ldm/modules/losses/contperceptual.py)
"""


@supervised_weighted_loss
def visual_foundational_loss(z, aux_feature, distmat_margin=0.25, cos_margin=0.5, distmat_weight=1.0, cos_weight=1.0, detach_aux_feature=True):
    # distmat: distance matrix

    z_flat = rearrange(z, 'b c h w -> b c (h w)')
    aux_feature_flat = rearrange(aux_feature, 'b c h w -> b c (h w)')

    z_norm = F.normalize(z_flat, dim=1)
    aux_feature_norm = F.normalize(aux_feature_flat, dim=1)

    # Cosine similarity
    z_cos_sim = torch.einsum('bci,bcj->bij', z_norm, z_norm)
    aux_feature_cos_sim = torch.einsum('bci,bcj->bij', aux_feature_norm, aux_feature_norm)

    # Loss computation
    if detach_aux_feature:
        diff = torch.abs(z_cos_sim - aux_feature_cos_sim.detach())
        vf_loss_1 = F.relu(diff - distmat_margin).mean()
        vf_loss_2 = F.relu(1 - F.cosine_similarity(aux_feature.detach(), z) - cos_margin).mean()
    else:
        diff = torch.abs(z_cos_sim - aux_feature_cos_sim)
        vf_loss_1 = F.relu(diff - distmat_margin).mean()
        vf_loss_2 = F.relu(1 - F.cosine_similarity(aux_feature, z) - cos_margin).mean()
    vf_loss = vf_loss_1 * distmat_weight + vf_loss_2 * cos_weight

    return vf_loss

@LOSS_REGISTRY.register()
class VFLoss(BaseLoss):
    def __init__(
            self,
            distmat_margin=0.25, cos_margin=0.5, distmat_weight=1.0, cos_weight=1.0, detach_aux_feature=True,
            loss_weight=1.0, reduction='mean',
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.distmat_margin = distmat_margin
        self.cos_margin = cos_margin
        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight
        self.detach_aux_feature = detach_aux_feature

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        return visual_foundational_loss(
            pred, target,
            distmat_margin=self.distmat_margin, cos_margin=self.cos_margin,
            distmat_weight=self.distmat_weight, cos_weight=self.cos_weight,
            detach_aux_feature=self.detach_aux_feature,
            weight=weight, reduction=self.reduction,
        )
