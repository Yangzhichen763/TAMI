import functools
import math
from typing import Optional

import lpips
import numpy as np

import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision.models as models

from basic.utils.registry import LOSS_REGISTRY

from .util import supervised_weighted_loss, _reduction_modes, weight_reduce_loss


def scale_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(self, *args, **kwargs):
        loss = loss_func(self, *args, **kwargs)
        loss = self.loss_weight * loss
        return loss

    return wrapper


class BaseLoss(nn.Module):
    """
    Base class for loss functions.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(
            self,
            reduction: str = 'mean',
            loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction


class UnsupervisedBaseLoss(nn.Module):
    """
    Base class for single loss functions.
    """
    def __init__(
            self,
            reduction: str = 'mean',
            loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, weight: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Args:
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = self.loss_func(inputs, weight=weight, **kwargs)
        return loss * self.loss_weight

    def loss_func(self, inputs: torch.Tensor, weight: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError


'''
Adapted from BasicSR(https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/losses/basic_loss.py)
'''


@supervised_weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@supervised_weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@supervised_weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    return F.smooth_l1_loss(pred, target, reduction='none', beta=beta)


@supervised_weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

@supervised_weighted_loss
def cosine_similarity_loss(pred, target, dim=1, eps=1e-8):
    """
    Args:
        pred: of shape (N, C, H, W). Predicted tensor.
        target: of shape (N, C, H, W). Ground truth tensor.
        dim: int. Dimension along which to compute the cosine similarity. Default: 1.
        eps: float. A value used to control the curvature near zero. Default: 1e-8.
    """
    # pred = F.normalize(pred, p=2, dim=dim, eps=eps)
    # target = F.normalize(target, p=2, dim=dim, eps=eps)
    # return 1 - (pred * target).sum(dim=dim)
    loss = 1.0 - F.cosine_similarity(pred, target, dim=dim, eps=eps)
    return loss

@supervised_weighted_loss
def sqr_cosine_similarity_loss(pred, target, dim=1, eps=1e-8):
    """
    Args:
        pred: of shape (N, C, H, W). Predicted tensor.
        target: of shape (N, C, H, W). Ground truth tensor.
        dim: int. Dimension along which to compute the cosine similarity. Default: 1.
        eps: float. A value used to control the curvature near zero. Default: 1e-8.
    """
    # pred = F.normalize(pred, p=2, dim=dim, eps=eps)
    # target = F.normalize(target, p=2, dim=dim, eps=eps)
    # return 1 - (pred * target).sum(dim=dim).pow(2)
    loss = 1.0 - F.cosine_similarity(pred, target, dim=dim, eps=eps).pow(2)
    return loss


'''
More losses see:
https://github.com/visionxiang/ZSCOS-CaMF/tree/master/src/Models/losses
'''


@LOSS_REGISTRY.register()
class L1Loss(BaseLoss):
    """L1 (mean absolute error, MAE) loss."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        return l1_loss(pred, target, weight=weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class SmoothL1Loss(BaseLoss):
    """Smooth L1 loss.

    Args:
        beta (float): The threshold in the piecewise function. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', beta=1.0):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.beta = beta

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        return smooth_l1_loss(pred, target, weight=weight, beta=self.beta, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(BaseLoss):
    """MSE (L2) loss."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        return mse_loss(pred, target, weight=weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(BaseLoss):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-24):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.eps = eps

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        return charbonnier_loss(pred, target, weight=weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CosineSimilarityLoss(BaseLoss):
    """Cosine similarity loss.

    Args:
        dim (int): Dimension along which to compute the cosine similarity. Default: 1.
        eps (float): A value used to control the curvature near zero. Default: 1e-8.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', dim=1, eps=1e-8):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

        self.dim = dim
        self.eps = eps

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        return cosine_similarity_loss(pred, target, weight=weight, dim=self.dim, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class LPIPSLoss(BaseLoss):
    def __init__(
            self,
            loss_weight=1.0,
            reduction='mean',
            use_input_norm=True,
            range_norm=False
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return weight_reduce_loss(lpips_loss, weight=weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class GANLoss(BaseLoss):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight, reduction='mean')
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class PSNRLoss(BaseLoss):

    def __init__(self, loss_weight=1.0, toY=False):
        super().__init__(loss_weight=loss_weight, reduction='mean')
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    @scale_loss
    def forward(self, pred, target, weight=None, **kwargs):
        assert weight is None, 'PSNRLoss does not support weight'
        assert len(pred.size()) == 4

        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@LOSS_REGISTRY.register()
class SSIMLoss(BaseLoss):
    def __init__(self, loss_weight=1.0, window_size=11, size_average=True, val_range=None):
        super().__init__(loss_weight=loss_weight, reduction='mean')
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    @scale_loss
    def forward(self, img1, img2, weight=None, **kwargs):
        assert weight is None, 'SSIMLoss does not support weight'

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        ssim_value = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

        return 1 - ssim_value


"""
Adapted from URWKV(https://github.com/FZU-N/URWKV/tree/main/model)
"""
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


@LOSS_REGISTRY.register()
class VGGLoss(BaseLoss):
    def __init__(self, loss_weight=1.0, conv_index='54', rgb_range=1):
        super().__init__(loss_weight=loss_weight, reduction='mean')

        vgg_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
            self.vgg.cuda()
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
            self.vgg.cuda()

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std).cuda()
        self.vgg.requires_grad = False

    @scale_loss
    def forward(self, sr, hr, weight=None, **kwargs):
        assert weight is None, 'VGGLoss does not support weight'

        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


def gaussian(window_size, sigma):
    """
    Create a 1-D gaussian kernel.

    Args:
        window_size (int): The size of window.
        sigma (float): The standard deviation of the gaussian kernel.

    Returns:
        Tensor: The 1-D gaussian kernel.
    """
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    """
    Create a window with given size and channel. The window is cross-correlation of two 1D gaussians.

    Args:
        window_size (int): The size of window.
        channel (int): The number of channels. Default: 1.

    Returns:
        Tensor: The window tensor.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    """
    Calculate SSIM (Structural Similarity Index Measure) for a pair of images.
    The formulas Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y] are used in the calculation of variance and covariance.
    And the expectation is calculated using the Gaussian kernel convolution.

    Args:
        img1 (Tensor): The first image.
        img2 (Tensor): The second image.
        window_size (int): The size of window. Default: 11.
        window (Tensor): The window tensor. Default: None.
        size_average (bool): Whether to average the SSIM results across the batch. Default: True.
        full (bool): Whether to return the cs score. Default: False.
        val_range (float): The value range of input images. Default: None.

    Returns:
        Tensor: The SSIM results.
    """
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
