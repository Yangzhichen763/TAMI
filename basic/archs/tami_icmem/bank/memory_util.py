from typing import Optional

import math


import torch
import torch.nn.functional as F
from functools import wraps

def pool_in_and_out(scale=4, mode='avg'):
    def decorator(func):
        @wraps(func)
        def wrapper(mk, ms, qk, qe, *args, **kwargs):
            def pool(x, scale):
                if x is None:
                    return None
                if x.dim() == 3:  # (B, C, L)
                    L = x.shape[2]
                    s = scale * scale
                    pooled_L = max(1, L // s)
                    if mode == 'avg':
                        return F.avg_pool1d(x, kernel_size=s) if L % s == 0 else F.adaptive_avg_pool1d(x, pooled_L)
                    elif mode == 'max':
                        return F.max_pool1d(x, kernel_size=s) if L % s == 0 else F.adaptive_max_pool1d(x, pooled_L)
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")
                elif x.dim() == 4:  # (B, C, H, W)
                    H, W = x.shape[-2:]
                    pooled_H, pooled_W = max(1, H // scale), max(1, W // scale)
                    if mode == 'avg':
                        return F.avg_pool2d(x, kernel_size=scale) if H % scale == 0 and W % scale == 0 else F.adaptive_avg_pool2d(x, (pooled_H, pooled_W))
                    elif mode =='max':
                        return F.max_pool2d(x, kernel_size=scale) if H % scale == 0 and W % scale == 0 else F.adaptive_max_pool2d(x, (pooled_H, pooled_W))
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")
                else:
                    raise ValueError(f"Unsupported dimension: {x.dim()}")

            mk_pooled = pool(mk, scale)
            ms_pooled = pool(ms, scale)
            qk_pooled = pool(qk, scale)
            qe_pooled = pool(qe, scale)

            out_pooled = func(mk_pooled, ms_pooled, qk_pooled, qe_pooled, *args, **kwargs)

            if out_pooled.dim() == 3:
                out = F.interpolate(
                    out_pooled.unsqueeze(0),
                    size=(mk.shape[2], math.prod(qk.shape[2:])),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                raise ValueError(f"Unsupported dimension: {out_pooled.dim()}")
            return out
        return wrapper
    return decorator


"""
Modified from XMem
"""


# @pool_in_and_out(scale=4, mode='avg')
def get_similarity(mk, qk, ms=None, qe=None, mode='sa'):
    """
    used for training/inference and memory reading/memory potential
    Args:
        mk: (B, C_k, N   ) - Memory keys
        ms: (B,   1, N   ) - Memory shrinkage
        qk: (B, C_k, HW/P) - Query keys
        qe: (B, C_k, HW/P) - Query selection
        mode (bool): spatial attention ('sa') or channel attention ('ca')
    """
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    C_k = mk.shape[1]

    if mode == 'sa':
        if qe is not None:
            # See appendix for derivation
            # or you can just trust me ヽ(ー_ー )ノ
            mk = mk.transpose(1, 2)                 # (B, C_k, N) -> (B, N, C_k)
            a_sq = (mk.pow(2) @ qe)                 # (B, N, C_k) @ (B, C_k, HW/P) -> (B, N, HW/P)
            two_ab = 2 * (mk @ (qk * qe))           # (B, N, C_k) @ (B, C_k, HW/P) -> (B, N, HW/P)
            b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)    # (B, C_k, HW/P) -> (B, 1, HW/P)
            similarity = (-a_sq+two_ab-b_sq)        # (B, N, HW/P)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)    # (B, N, 1)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)  # (B, N, C_k) @ (B, C_k, HW/P) -> (B, N, HW/P)
            similarity = (-a_sq+two_ab)             # (B, N, HW/P)

        if ms is not None:
            similarity = similarity * ms / math.sqrt(C_k)   # (B, N, HW/P)
        else:
            similarity = similarity / math.sqrt(C_k)   # (B, N, HW/P)
    elif mode == 'ca':
        raise NotImplementedError("Channel attention not implemented yet")
    elif mode == 'cosine':
        x_norm = F.normalize(mk, dim=-1)  # (B, C_k, N)
        y_norm = F.normalize(qk, dim=-1)  # (B, C_k, HW/P)
        similarity = torch.matmul(x_norm.transpose(-1, -2), y_norm)  # (B, C_k, N)^T @ (B, C_k, HW/P) -> (B, N, HW/P)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return similarity


def get_similarity_chunked(mk, qk, ms=None, qe=None, chunk_size=256):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    B, CK, N = mk.shape
    _, _, HW = qk.shape

    similarity = torch.empty((B, N, HW), device=mk.device)

    if qe is not None:
        mk = mk.transpose(1, 2)  # B x N x CK
        for i in range(0, HW, chunk_size):
            chunk_end = min(i + chunk_size, HW)
            qk_chunk = qk[:, :, i:chunk_end]
            qe_chunk = qe[:, :, i:chunk_end]

            a_sq = (mk.pow(2) @ qe_chunk)
            two_ab = 2 * (mk @ (qk_chunk * qe_chunk))
            b_sq = (qe_chunk * qk_chunk.pow(2)).sum(1, keepdim=True)
            similarity[:, :, i:chunk_end] = (-a_sq + two_ab - b_sq)
    else:
        a_sq = mk.pow(2).sum(1).unsqueeze(2)  # B x N x 1
        for i in range(0, HW, chunk_size):
            chunk_end = min(i + chunk_size, HW)
            qk_chunk = qk[:, :, i:chunk_end]
            two_ab = 2 * (mk.transpose(1, 2) @ qk_chunk)
            similarity[:, :, i:chunk_end] = (-a_sq + two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)
    else:
        similarity = similarity / math.sqrt(CK)

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: (B, N, HW/P)
    # use inplace with care
    def get_affinity(x):
        x_exp = torch.exp(x)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp_sum = torch.where(x_exp_sum < 1e-6, torch.ones_like(x_exp_sum), x_exp_sum)
        affinity = x_exp / x_exp_sum
        return affinity

    if top_k is not None:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        similarity_values, indices = torch.topk(similarity, k=top_k, dim=1)

        if torch.mean(similarity_values).item() > 1e4:
            from basic.utils.console.log import get_root_logger
            logger = get_root_logger()
            logger.warning(f"Similarity values are too large, got avg {torch.mean(similarity_values).item():.0f} and max {torch.max(similarity_values).item():.0f}, you can ignore this warning if training is stable")

        # x_exp = similarity_values.exp_()
        # x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        affinity = get_affinity(similarity_values - maxes)
        if inplace:
            similarity.zero_().scatter_(1, indices, affinity) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, affinity) # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        # x_exp = torch.exp(similarity - maxes)
        # x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        # affinity = x_exp / x_exp_sum
        affinity = get_affinity(similarity - maxes)
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def masked_softmax(
        similarity, threshold=0.0, return_usage=False,
        **softmax_kwargs
):
    # similarity: (B, N, HW/P)
    # sim_max = torch.amax(similarity, dim=-1, keepdim=True)[0]
    # sim_mean = torch.mean(similarity, dim=-1, keepdim=True)
    # sim_threshold = (sim_max - sim_mean) * threshold + sim_mean
    # sim_threshold = torch.maximum(sim_threshold, torch.zeros_like(sim_threshold))
    x_mask = (similarity > threshold).float()

    x_masked = similarity.masked_fill(x_mask == 0, -1e9)
    return do_softmax(x_masked, return_usage=return_usage, **softmax_kwargs)

def weighted_softmax(
        similarity, weight: torch.Tensor, return_usage=False,
        **softmax_kwargs
):
    # similarity: (B, N, HW/P)
    if similarity.shape != weight.shape:
        def compute_repeat_factors(src_shape, tgt_shape):
            n1, c1 = src_shape
            _, n2, c2 = tgt_shape
            assert n2 % n1 == 0 and c2 % c1 == 0, f"x shape must be divisible by y shape, got {n2}/{n1} and {c2}/{c1}"
            return 1, n2 // n1, c2 // c1
        size = compute_repeat_factors(weight.shape, similarity.shape)
        weight = weight.repeat(*size)
    weighted_similarity = similarity * weight

    return do_softmax(weighted_similarity, return_usage=return_usage, **softmax_kwargs)


def get_affinity(mk, qk, ms=None, qe=None, softmax_func=None):
    """
    used for training/inference and memory reading/memory potential
    Args:
        mk: (B, C_k, N   ) - Memory keys
        ms: (B,   1, N   ) - Memory shrinkage
        qk: (B, C_k, HW/P) - Query keys
        qe: (B, C_k, HW/P) - Query selection
        softmax_func: function to apply softmax, default is do_softmax
    """
    if softmax_func is None:
        softmax_func = do_softmax

    # shorthand used in training with no top-k
    similarity = get_similarity(mk, qk, ms, qe)
    affinity = softmax_func(similarity)
    return affinity

def readout(affinity, mv):
    mem = torch.bmm(mv, affinity)   # (B, CV, n) @ (B, n, HW) -> (B, CV, HW)
    return mem



def generate_gaussian_kernels(h, w, sigma=2, normalize=False, device='cpu'):
    # coord grid
    y = torch.arange(h, device=device).repeat(w)                # (N)
    x = torch.arange(w, device=device).repeat_interleave(h)     # (N)
    coords = torch.stack(tensors=[x, y], dim=1).float()         # (N, 2)

    # calculate pairwise distance matrix: (N, N)
    diff = coords[None, :, :] - coords[:, None, :]              # (N, N, 2)
    dist2 = (diff ** 2).sum(dim=-1)  # (N, N)
    gaussian_weight = torch.exp(-dist2 / (2 * sigma ** 2))      # (N, N)

    if normalize:
        gaussian_weight = gaussian_weight / gaussian_weight.sum(dim=-1, keepdim=True)

    return gaussian_weight  # (N, N)
