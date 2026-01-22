import math
import lpips
import numpy as np
from scipy.spatial import KDTree

import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basic.archs.vgg_arch import VGGFeatureExtractor
from .util import supervised_weighted_loss
from .basic_loss import _reduction_modes
from basic.utils.registry import LOSS_REGISTRY


@supervised_weighted_loss
def lp_loss(pred, targe, p=2):
    """
    Lp loss
    """
    diff = pred.unsqueeze(dim=-2) - targe.unsqueeze(dim=-1)  # [B, N, C] - [B, M, C] -> [B, N, M, C]
    distances = torch.norm(diff, p=p, dim=2)                 # [B, N, M, C] -> [B, N, M]
    return distances


@LOSS_REGISTRY.register()
class LpLoss(nn.Module):
    """Lp loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LpLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * lp_loss(pred, target, weight, reduction=self.reduction, **kwargs)


@supervised_weighted_loss
def point_min_distance_loss(points_pred, points_gt, distance_func=lp_loss):
    """
    Calculate the minimum distance between each point in points_pred and the nearest point in points_gt.
    """
    distances = distance_func(points_pred, points_gt)   # [B, N, M, C] -> [B, N, M]
    min_dist = distances.min(dim=-1)                    # [B, N, M] -> [B, N]
    return min_dist


def point_count_density(points, k=10, eps=1e-10):
    """
    Compute the density of a point cloud.

    Args:
        points (torch.Tensor): Point cloud data, shape [B, N, C], where B is the batch size,
                              N is the number of points, and C is the dimensionality.
        k (int): Number of nearest neighbors to consider.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Density of each point, shape [B, N].
    """
    distances = lp_loss(points, points, p=2)                                # [B, N, C] -> [B, N, N]

    # Find the k+1 nearest neighbors (including itself)
    topk_values, _ = torch.topk(distances, k=k + 1, dim=-1, largest=False)  # [B, N, N] -> [B, N, k+1]
    avg_distances = topk_values[:, :, 1:].mean(dim=-1)                      # [B, N, k+1] -> [B, N, k] -> [B, N]
    densities = 1.0 / (avg_distances + eps)                                 # [B, N]
    return densities


def density_gradient(points, point_idx, k=10):
    """
    Compute the density gradient of a point in the point cloud.

    Args:
        points (torch.Tensor): Point cloud data, shape [B, N, C].
        point_idx (int): Index of the target point.
        k (int): Number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Density gradient of the target point, shape [B, C].
    """
    B, N, C = points.shape
    assert 0 <= point_idx < N, "point_idx is out of range"

    # Set the target point as a variable requiring gradient computation,
    # and replace the target point in the point cloud:
    target_point = points[:, point_idx, :].clone().requires_grad_(True) # [B, C]
    modified_points = points.clone()
    modified_points[:, point_idx, :] = target_point                     # [B, N, C]

    # Compute the density
    densities = point_count_density(modified_points, k=k)                     # [B, N, C] -> [B, N]
    target_density = densities[:, point_idx].sum()                      # [B] -> scalar

    # Compute the gradient
    target_density.backward()
    gradient = target_point.grad                                        # [B, C]
    return gradient


def density_gradient_external(points, external_point, k=10):
    """
    Compute the density gradient of a point that is not in the point cloud.

    Args:
        points (torch.Tensor): Point cloud data, shape [B, N, C].
        external_point (torch.Tensor): The external point, shape [B, C].
        k (int): Number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Density gradient of the external point, shape [B, C].
    """
    B, N, C = points.shape
    assert external_point.shape == (B, C), "external_point must have shape [B, C]"

    # Set the external point as a variable requiring gradient computation,
    # and add the external point to the point cloud
    external_point = external_point.clone().requires_grad_(True)                        # [B, C]
    modified_points = torch.cat(tensors=[points, external_point.unsqueeze(1)], dim=1)   # [B, N, C] -> [B, N+1, C]

    # Compute the density
    densities = point_count_density(modified_points, k=k)                                     # [B, N+1, C] -> [B, N+1]
    # Get the density of the external point
    external_density = densities[:, -1].sum()                                           # [B] -> scalar

    # Compute the gradient
    external_density.backward()
    gradient = external_point.grad                                                      # [B, C]

    return gradient


def point_distance_density(points, radius):
    dists = torch.cdist(points, points)

    density = (dists < radius).float().sum(dim=1) - 1  # 减去自身
    return density


def point_distance_density_batch(points, radius, batch_size=2048):
    n_points = points.shape[0]
    density = torch.zeros(n_points, device=points.device)

    for i in range(0, n_points, batch_size):
        batch = points[i:i + batch_size]
        dists = torch.cdist(batch, points)
        density[i:i + batch_size] = (dists < radius).float().sum(dim=1) - 1

    return density