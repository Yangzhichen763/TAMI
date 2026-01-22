import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basic.utils.registry import LOSS_REGISTRY
from basic.losses.util import supervised_weighted_loss, _reduction_modes
from basic.losses.basic_loss import BaseLoss, scale_loss


@supervised_weighted_loss
def kl_loss(z, aux_feature, z_temperature=1.0, aux_feature_temperature=1.0):
    x_prob = F.log_softmax(z / z_temperature, dim=1)
    y_prob = F.softmax(aux_feature / aux_feature_temperature, dim=1)

    kl_loss = F.kl_div(x_prob, y_prob, reduction='none')
    return kl_loss


@supervised_weighted_loss
def symmetric_kl_loss(z, aux_feature, z_temperature=1.0, aux_feature_temperature=1.0):
    x_prob = F.softmax(z / z_temperature, dim=1)
    y_prob = F.softmax(aux_feature / aux_feature_temperature, dim=1)

    # D_KL(P || Q) and D_KL(Q || P)
    log_x_prob = torch.log(x_prob)
    log_y_prob = torch.log(y_prob)

    kl_p_q = F.kl_div(log_x_prob, y_prob, reduction='none')
    kl_q_p = F.kl_div(log_y_prob, x_prob, reduction='none')

    symmetric_kl_loss_value = 0.5 * (kl_p_q + kl_q_p)
    return symmetric_kl_loss_value


@supervised_weighted_loss
def js_loss(z, aux_feature, z_temperature=1.0, aux_feature_temperature=1.0):
    x_prob = F.softmax(z / z_temperature, dim=1)
    y_prob = F.softmax(aux_feature / aux_feature_temperature, dim=1)

    m_prob = 0.5 * (x_prob + y_prob)

    # D_KL(P || M) 和 D_KL(Q || M)
    kl_p_m = F.kl_div(F.log_softmax(x_prob, dim=1), m_prob, reduction='none')
    kl_q_m = F.kl_div(F.log_softmax(y_prob, dim=1), m_prob, reduction='none')

    js_loss_value = 0.5 * (kl_p_m + kl_q_m).mean()

    return js_loss_value


@LOSS_REGISTRY.register()
class KLLoss(BaseLoss):
    def __init__(
            self,
            z_temperature=1.0, aux_feature_temperature=1.0,
            loss_weight=1.0, reduction='mean',
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.z_temperature = z_temperature
        self.aux_feature_temperature = aux_feature_temperature

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * kl_loss(
            pred, target,
            z_temperature=self.z_temperature,
            aux_feature_temperature=self.aux_feature_temperature,
            weight=weight, reduction=self.reduction,
        )


@LOSS_REGISTRY.register()
class SymmetricKLLoss(BaseLoss):
    def __init__(
            self,
            z_temperature=1.0, aux_feature_temperature=1.0,
            loss_weight=1.0, reduction='mean',
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.z_temperature = z_temperature
        self.aux_feature_temperature = aux_feature_temperature

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * symmetric_kl_loss(
            pred, target,
            z_temperature=self.z_temperature,
            aux_feature_temperature=self.aux_feature_temperature,
            weight=weight, reduction=self.reduction,
        )


@LOSS_REGISTRY.register()
class JSLoss(BaseLoss):
    def __init__(
            self,
            z_temperature=1.0, aux_feature_temperature=1.0,
            loss_weight=1.0, reduction='mean',
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.z_temperature = z_temperature
        self.aux_feature_temperature = aux_feature_temperature

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * js_loss(
            pred, target,
            z_temperature=self.z_temperature,
            aux_feature_temperature=self.aux_feature_temperature,
            weight=weight, reduction=self.reduction,
        )
