import torch

from basic.losses.rank_losses import PADDED_Y_VALUE
from basic.losses.rank_losses import DEFAULT_EPS

"""
Modified from (https://github.com/allegro/allRank/blob/master/allrank/models/losses/neuralNDCG.py)
"""

import torch.nn as nn
from basic.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ApproxNDCGLoss(nn.Module):
    def __init__(self, loss_weight=1.0, **kwargs):
        """
        Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
        Information Retrieval Measures". Please note that this method does not implement any kind of truncation.

        Args:
            y_pred (torch.Tensor): predictions from the model, shape [batch_size, slate_length]
            y_true (torch.Tensor): ground truth labels, shape [batch_size, slate_length]
            eps: epsilon value, used for numerical stability
            padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
            alpha: score difference weight used in the sigmoid function
        Returns:
            torch.Tensor: loss value, a torch.Tensor
        """
        super(ApproxNDCGLoss, self).__init__()
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Approximate NDCG loss.

        Args:
            y_pred (torch.Tensor): predictions from the model, shape [batch_size, slate_length]
            y_true (torch.Tensor): ground truth labels, shape [batch_size, slate_length]
        Returns:
            torch.Tensor: loss value, a torch.Tensor
        """
        return self.loss_weight * (1 + approxNDCGLoss(y_pred, y_true, **self.kwargs))


def approxNDCGLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)
