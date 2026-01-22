import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basic.utils.registry import LOSS_REGISTRY

from basic.losses.util import contrastive_weighted_loss, _reduction_modes
from basic.losses.basic_loss import UnsupervisedBaseLoss, scale_loss


# 熵损失，用于计算 attn_scores 的熵，越小越好，鼓励 attn_scores 尽可能地远离均匀分布
@contrastive_weighted_loss
def entropy_loss(attn_scores):
    prob = torch.softmax(attn_scores, dim=-1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
    return -entropy


@LOSS_REGISTRY.register()
class EntropyLoss(UnsupervisedBaseLoss):
    """Entropy loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

    @scale_loss
    def forward(self, inputs, weight=None, **kwargs):
        """
        Args:
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return entropy_loss(inputs, weight=weight, reduction=self.reduction)


# 多样性损失，越小越好，鼓励 selected_features 尽可能地多样
@contrastive_weighted_loss
def diversity_loss(selected_features):
    """
    Args:
        selected_features: (B, k, dim)
    """
    k = selected_features.shape[-2]

    # similarity matrix
    normalized_frames = F.normalize(selected_features, p=2, dim=-1)
    similarity = torch.bmm(normalized_frames, normalized_frames.transpose(-1, -2))  # (B, k, k)

    # set self similarity to 0
    eye = torch.eye(k, device=selected_features.device).unsqueeze(0)
    similarity = similarity * (1 - eye)

    # lower better
    return similarity  # (B, k, k)


@LOSS_REGISTRY.register()
class DiversityLoss(UnsupervisedBaseLoss):
    """Diversity loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

    @scale_loss
    def forward(self, inputs, weight=None, **kwargs):
        """
        Args:
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return diversity_loss(inputs, weight=weight, reduction=self.reduction)


# 关键帧覆盖率损失，越小越好，鼓励 selected_features 覆盖所有帧
@contrastive_weighted_loss
def coverage_loss(selected_features, all_features):
    """
    Args:
        selected_features: (B, k, dim)
        all_features: (B, N, dim)
    """
    # compute distance between selected and all features
    dist_matrix = torch.cdist(all_features, selected_features, p=2) # (B, N, dim) o (B, k, dim) -> (B, N, k)
    min_dist = dist_matrix.min(dim=-1)[0]                           # (B, N)

    # lower better
    return min_dist


@LOSS_REGISTRY.register()
class CoverageLoss(UnsupervisedBaseLoss):
    """Coverage loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__(loss_weight=loss_weight, reduction=reduction)

    @scale_loss
    def forward(self, selected_features, all_features, weight=None, **kwargs):
        """
        Args:
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * coverage_loss(selected_features, all_features, weight=weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class ClusterLoss(nn.Module):
    def __init__(self, feature_dim, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, feature_dim))

    def forward(self, selected_frames):
        # selected_frames: (B, k, dim)
        batch_size, k, _ = selected_frames.shape

        # calculate distances between selected frames and cluster centers
        dists = torch.cdist(selected_frames, self.cluster_centers)  # [batch_size, k, n_clusters]

        # assign each selected frame to the closest cluster center
        assignments = dists.argmin(dim=-1)  # [batch_size, k]

        # encourage each cluster to be covered by at least one selected frame
        coverage = torch.zeros(batch_size, self.n_clusters, device=selected_frames.device)
        coverage.scatter_(1, assignments, 1)
        coverage = coverage.sum(dim=0)  # [n_clusters]

        # encourage uniform distribution of cluster centers
        prob = coverage / (batch_size * k)
        entropy = -(prob * (prob + 1e-10).log()).sum()

        return entropy
