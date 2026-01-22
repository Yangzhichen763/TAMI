import torch
import torch.nn as nn
import torch.nn.functional as F
from basic.losses.basic_loss import BaseLoss, scale_loss


def l2_normalize(x, eps=1e-8):
    """L2-normalize a tensor along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def cosine_sim(a, b):
    """Compute cosine similarity between two tensors."""
    return (a * b).sum(dim=-1)


class TripletMarginLoss(BaseLoss):
    """
    Triplet margin loss with hardest negative mining.
    The loss encourages:
      - anchor <-> positive distance to be small
      - anchor <-> hardest negative distance to be large

    Args:
        margin:   margin value (typically 0.1–0.5)
        distance: 'cosine' or 'euclidean'
    """
    def __init__(self, margin=0.2, distance='sqr_cosine', loss_weight=1.0):
        super().__init__(loss_weight=loss_weight, reduction="mean")
        self.margin = margin
        assert distance in ['cosine', 'sqr_cosine', 'euclidean']
        self.distance = distance

    @scale_loss
    def forward(self, a, p=None, n=None, normalize=True):
        """
        Args:
            a: (B, C)  - anchor features
            p: (B, C)  - positive features
            n: list[(B, C)] or tensor (B, K, C) - negative features
            normalize: whether to L2 normalize inputs
        Returns:
            scalar loss
        """
        if a.dim() == 4:
            a = a.reshape(a.shape[0], -1)

        if n is not None:
            if isinstance(n, list) and len(n) > 0:
                n = torch.stack(n, dim=1)
        if normalize:
            a = l2_normalize(a)
            if p is not None:
                p = l2_normalize(p)
            if n is not None:
                n = l2_normalize(n)

        d_ap = torch.zeros_like(a[:, 0])  # (B,)
        d_an = torch.zeros_like(a[:, 0])  # (B,)
        if self.distance == 'cosine':
            # Distance = 1 - cosine similarity
            if p is not None:
                d_ap = 1.0 - cosine_sim(a, p)                                   # (B,)
            if n is not None:
                d_an = 1.0 - cosine_sim(a.unsqueeze(1), n).min(dim=1).values    # (B,)
        elif self.distance == 'sqr_cosine':
            # Distance = torch.abs(1 - cosine similarity)
            if p is not None:
                d_ap = 1 - cosine_sim(a, p).pow(2)                                      # (B,)
            if n is not None:
                d_an = 1 - cosine_sim(a.unsqueeze(1), n).min(dim=1).values.pow(2)       # (B,)
        elif self.distance == 'euclidean':
            # Euclidean distance
            if p is not None:
                d_ap = (a - p).pow(2).sum(dim=-1).sqrt()                                    # (B,)
            if n is not None:
                d_an = (a.unsqueeze(1) - n).pow(2).sum(dim=-1).sqrt().min(dim=1).values     # (B,)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")


        # Margin ranking loss: max(0, margin + d_ap - d_an)
        loss = F.relu(self.margin + d_ap - d_an).mean()
        return loss


class InfoNCELoss(BaseLoss):
    """
    Contrastive InfoNCE / NT-Xent loss.
    Encourages the anchor to be similar to its positive
    and dissimilar to all negatives.

    Args:
        temperature: scaling factor (typical 0.05–0.2)
        label_smoothing: optional smoothing to improve stability
    """
    def __init__(self, temperature=0.07, label_smoothing=0.0, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight, reduction="mean")
        self.t = temperature
        self.ls = label_smoothing

    @scale_loss
    def forward(self, a, p, n, normalize=True):
        """
        Args:
            a: (B, C)  - anchor features
            p: (B, C)  - positive features
            n: list[(B, C)] or tensor (B, K, C) - negative features
            normalize: whether to L2 normalize inputs
        Returns:
            loss: scalar
            logits: (B, 1+K) similarity scores
        """
        if isinstance(n, list) and len(n) > 0:
            n = torch.stack(n, dim=1)
        if normalize:
            a = l2_normalize(a)
            p = l2_normalize(p)
            n = l2_normalize(n)

        B, K, C = n.shape

        # Cosine similarities (scaled by temperature)
        sim_pos = cosine_sim(a, p) / self.t                  # (B,)
        sim_neg = cosine_sim(a.unsqueeze(1), n) / self.t     # (B, K)

        # Concatenate positive and negatives -> (B, 1+K)
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)

        # Positive sample is at index 0
        target = torch.zeros(B, dtype=torch.long, device=a.device)

        if self.ls > 0:
            # Label smoothing version of cross-entropy
            eps = self.ls
            smooth_target = torch.full_like(logits, fill_value=eps / (K + 1))
            smooth_target[:, 0] = 1.0 - eps
            log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
            loss = -(smooth_target * log_prob).sum(dim=1).mean()
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(logits, target)

        return loss, logits


class NPairLoss(BaseLoss):
    """
    N-pair loss (multi-class N-pair loss)
    Reference: 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'
               (Sohn, NIPS 2016)

    Each (anchor_i, positive_i) forms a positive pair.
    All other positives in the batch are used as negatives for that anchor.
    """

    def __init__(self, l2_reg=0.0, normalize=True, loss_weight=1.0):
        """
        Args:
            l2_reg: Optional L2 regularization term on embeddings (default: 0)
            normalize: Whether to apply L2 normalization to embeddings
        """
        super().__init__(loss_weight=loss_weight, reduction="mean")
        self.l2_reg = l2_reg
        self.normalize = normalize

    @scale_loss
    def forward(self, anchor, positive):
        """
        Args:
            anchor: (B, C) - anchor embeddings
            positive: (B, C) - positive embeddings (same batch order)
        Returns:
            Scalar loss
        """
        if self.normalize:
            anchor = l2_normalize(anchor)
            positive = l2_normalize(positive)

        # Similarity matrix between anchors and all positives: (B, B)
        sim_matrix = torch.matmul(anchor, positive.T)

        # Diagonal: anchor <-> its own positive
        pos_sim = torch.diag(sim_matrix).unsqueeze(1)  # (B, 1)

        # Compute difference with all other positives
        # Equivalent to exp(sim(a_i, p_j) - sim(a_i, p_i))
        diff = sim_matrix - pos_sim  # (B, B)

        # Mask out self-similarity (j == i)
        mask = ~torch.eye(anchor.size(0), dtype=torch.bool, device=anchor.device)
        diff = diff[mask].view(anchor.size(0), -1)  # (B, B-1)

        # Log-sum-exp over negatives
        loss_per_anchor = torch.log1p(torch.exp(diff).sum(dim=1))
        loss = loss_per_anchor.mean()

        # Optional L2 regularization term on embeddings
        if self.l2_reg > 0:
            l2_loss = 0.25 * (anchor.pow(2).sum(dim=1) + positive.pow(2).sum(dim=1)).mean()
            loss += self.l2_reg * l2_loss

        return loss


class MagnetLoss(BaseLoss):
    """
    Magnet Loss implementation in PyTorch.

    Based on:
    "The Magnet Loss: Metric Learning with Adaptive Density Discrimination"
    (Rippel et al., CVPR 2016)

    Args:
        alpha: margin controlling separation between clusters.
        eps: small value to avoid division by zero.
    """
    def __init__(self, alpha=1.0, eps=1e-6, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight, reduction="mean")
        self.alpha = alpha
        self.eps = eps

    @scale_loss
    def forward(self, embeddings, labels, cluster_centers, cluster_labels):
        """
        Args:
            embeddings: (B, D) tensor of sample embeddings.
            labels: (B,) tensor of class labels for each sample.
            cluster_centers: (M, D) tensor of cluster centroids (from training set or online update).
            cluster_labels: (M,) tensor of class labels for each cluster.
        Returns:
            loss: scalar
        """
        # Ensure normalized embeddings (optional, but often helps)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        cluster_centers = F.normalize(cluster_centers, p=2, dim=-1)

        B, D = embeddings.shape
        M = cluster_centers.shape[0]

        # Compute pairwise squared distances: (B, M)
        dists = torch.cdist(embeddings, cluster_centers, p=2).pow(2)

        # For each sample, find its own cluster
        mask_same_class = (labels.unsqueeze(1) == cluster_labels.unsqueeze(0))  # (B, M)
        same_class_dists = dists.clone()
        same_class_dists[~mask_same_class] = float('inf')

        # Get distance to closest cluster of the same class (assigned cluster)
        assigned_dists, _ = same_class_dists.min(dim=1)  # (B,)

        # Compute global variance \sigma^2 (Rippel used across all distances)
        sigma_sq = torch.mean(dists) + self.eps

        # Compute numerator and denominator
        # Numerator: exp( - (||x - \miu_c||^2) / (2\sigma^2) - \alpha )
        num = torch.exp(-0.5 * assigned_dists / sigma_sq - self.alpha)

        # Denominator: sum over all clusters
        denom = torch.sum(torch.exp(-0.5 * dists / sigma_sq), dim=1) + self.eps

        # Magnet loss for each sample
        loss_i = -torch.log(num / denom)

        # Average over batch
        loss = loss_i.mean()
        return loss
