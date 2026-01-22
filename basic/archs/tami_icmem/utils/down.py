
import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_downsample(x, size, mul=2, mode_1='bilinear', mode_2='bilinear'):
    """
    Downsamples an input image to a target size while preserving multi-scale details
    by stacking resized versions along the channel dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        size (tuple): Target spatial dimensions (h, w).
        mul (int): Channel multiplier. Output channels will be C * mul.

    Returns:
        torch.Tensor: Output tensor of shape (B, C*mul, h, w).
    """
    B, C, H, W = x.shape
    h, w = size

    # Generate multi-scale features
    features = []
    for i in range(mul):
        # Method 1: Vary interpolation scales (e.g., using different strides)
        scale = 1.0 - i * (0.5 / mul)  # Custom scaling factor
        scaled_h, scaled_w = int(H * scale), int(W * scale)

        # Resize to an intermediate scale
        x_resized = F.interpolate(
            x,
            size=(scaled_h, scaled_w),
            mode=mode_1,
            align_corners=False
        )

        # Downsample to target size
        x_down = F.interpolate(
            x_resized,
            size=(h, w),
            mode=mode_2,
            align_corners=False
        )
        features.append(x_down)

    # Concatenate along channel dimension
    out = torch.cat(features, dim=1)  # Shape: (B, C*mul, h, w)
    return out


def unshuffle_downsample(x, scale_factor):
    """
    Downsamples an image tensor while preserving spatial details by expanding local regions into channels.

    Args:
        x: Input tensor of shape (B, C, H, W)
        scale_factor: Integer scaling factor s (must evenly divide H and W)

    Returns:
        Tensor of shape (B, C*(s² + 1), H//s, W//s) containing:
        - The downscaled version (averaged)
        - All s×s local neighborhood details as additional channels
    """
    b, c, h, w = x.shape
    assert h % scale_factor == 0 and w % scale_factor == 0, \
        "Height and width must be divisible by scale_factor"

    # 1. Average downsampling (low-frequency component)
    x_down = nn.AvgPool2d(scale_factor)(x)  # Shape: (B, C, h//s, w//s)

    # 2. Unfold local s×s neighborhoods into channels (high-frequency details)
    # Unfold operation extracts all s×s patches with given stride
    x_unfold = nn.Unfold(scale_factor, stride=scale_factor)(x)  # Shape: (B, C*s*s, (h//s)*(w//s))

    # Reshape to separate spatial dimensions and neighborhood elements
    x_unfold = x_unfold.view(b, c, scale_factor * scale_factor, h // scale_factor, w // scale_factor)

    # Reorder dimensions to group spatial positions together
    x_unfold = x_unfold.permute(0, 1, 3, 4, 2).reshape(
        b,
        c * (scale_factor * scale_factor),
        h // scale_factor,
        w // scale_factor
    )

    # 3. Concatenate downsampled version with neighborhood details
    result = torch.cat([x_down, x_unfold], dim=1)  # Shape: (B, C*(s² + 1), h//s, w//s)

    return result
