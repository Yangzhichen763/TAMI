import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basic.utils.registry import LOSS_REGISTRY
from basic.losses.util import supervised_weighted_loss, _reduction_modes
from basic.losses.basic_loss import BaseLoss, scale_loss


"""
Adapted from Spatial-Forcing(https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/openvla-SF/prismatic/models/projectors.py#L52-L115)
"""


@supervised_weighted_loss
def spatial_forcing_loss(
        pred, target, projector, loss_func, interpolate=False
):
    pred = projector(pred)

    if interpolate:
        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

    if isinstance(loss_func, str):
        if loss_func == "cosine":
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)
            sim = (pred * target).sum(dim=-1)
            align_loss = 1 - sim
        elif loss_func == "mse":
            align_loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # mean over last dim
        else:
            raise ValueError(f"Unsupported loss type: {loss_func}. Supported ones are: ['cosine', 'mse']")
    else:
        align_loss = loss_func(pred, target)

    return align_loss


@LOSS_REGISTRY.register()
class SpatialForcingLoss(BaseLoss):
    """
    spatial alignment loss
    """

    def __init__(
            self,
            in_dim, out_dim,
            loss_func,
            loss_weight=1.0,
            reduction="mean",

            **mlp_kwargs
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.loss_func = loss_func
        self.projector = self._make_mlp(style='conv', in_dim=in_dim, out_dim=out_dim, **mlp_kwargs)

    def _make_mlp(self, style='conv', *args, **kwargs):
        if style.lower() == 'conv':
            return ConvMLP(*args, **kwargs)
        elif style.lower() == 'linear':
            return LinearMLP(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported MLP style: {style}. Supported ones are: ['conv', 'linear']")

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (B, C, H, W)
            target (Tensor): (B, C, H, W)
        """
        return spatial_forcing_loss(
            pred,
            target,
            projector=self.projector,
            loss_func=self.loss_func,
            weight=weight,
        )


@LOSS_REGISTRY.register()
class ArbitraryResolutionSpatialForcingLoss(BaseLoss):
    def __init__(
            self,
            in_dim, out_dim,
            loss_func,
            loss_weight=1.0,
            reduction="mean",

            **mlp_kwargs
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.loss_func = loss_func
        self.projector = self._make_mlp(style='conv', in_dim=in_dim, out_dim=out_dim, **mlp_kwargs)

    def _make_mlp(self, style='conv', *args, **kwargs):
        if style.lower() == 'conv':
            return ConvMLP(*args, **kwargs)
        elif style.lower() == 'linear':
            return LinearMLP(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported MLP style: {style}. Supported ones are: ['conv', 'linear']")

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (B, C, H, W)
            target (Tensor): (B, C, H, W)
        """
        return spatial_forcing_loss(
            pred,
            target,
            projector=self.projector,
            loss_func=self.loss_func,
            interpolate=True,
            weight=weight,
        )


class ConvMLP(nn.Module):
    def __init__(
            self, in_dim, out_dim, hidden_dim=None,
            kernel_size=1, stride=1,
            use_norm=False, act_func=nn.GELU()
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if hidden_dim is None:
            hidden_dim = out_dim
        self.hidden_dim = hidden_dim

        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(in_dim)
        else:
            self.norm = nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2, stride=stride),
            act_func,
            nn.Conv2d(hidden_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x):
        h, w = x.shape[-2:]

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x = self.net(x)
        return x


class LinearMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, use_norm=False, act_func=nn.GELU()):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if hidden_dim is None:
            hidden_dim = out_dim
        self.hidden_dim = hidden_dim

        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(in_dim)
        else:
            self.norm = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_func,
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x):
        h, w = x.shape[-2:]

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = self.net(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
