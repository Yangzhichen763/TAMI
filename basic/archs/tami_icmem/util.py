import math
import torch
import torch.nn.functional as F
from einops import rearrange


def match_illumination(semantic_query, semantic_key, illumination_value, type="cosine"):
    """
    Args:
        semantic_query:     (B, 1, C, N)
        semantic_key:       (B, T, C, N)
        illumination_value: (B, T, d, N)
    """
    B, T, C, N = semantic_key.shape

    # 计算 semantic 之间的相似性
    if type == "cosine":
        query = semantic_query.repeat(1, T, 1, 1)               # (B, 1, C, N) -> (B, T, C, N)
        similarity = query.transpose(-1, -2) @ semantic_key     # (B, T, C, N)^T @ (B, T, C, N) -> (B, T, N, N)
    else:
        query = semantic_query.repeat(1, T, 1, 1)               # (B, 1, C, N) -> (B, T, C, N)
        a_sq = query.pow(2).sum(-2).unsqueeze(-1)               # (B, T, C, N) -> (B, T, N) -> (B, T, N, 1)
        two_ab = 2 * (query.transpose(-1, -2) @ semantic_key)   # (B, T, C, N)^T @ (B, T, C, N) -> (B, T, N, N)
        similarity = (-a_sq+two_ab)                             # (B, T, N, N)
    similarity = similarity / math.sqrt(C)

    # 提取相似 semantic 的 illumination
    value = illumination_value.transpose(-1, -2)                # (B, T, d, N) -> (B, T, N, d)
    illumination = (similarity @ value).transpose(-1, -2)       # (B, T, N, N) @ (B, T, N, d) -> (B, T, N, d) -> (B, T, d, N)

    # 得到的是具有相似语义的不同时间帧下的 illumination
    return illumination


#region ==[Illumination Map]== 获取光照图（灰度图）
def gaussian_kernel(kernel_size=3, sigma=1.0, channels=3):
    """ Generates a 2D Gaussian kernel. (Isotropic sigma)
    """
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0
    kernel = (1.0 / (2.0 * math.pi * variance)) * \
                     torch.exp(-torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance))

    kernel = kernel / torch.sum(kernel)

    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    return kernel


def gaussian_kernel_2d(kernel_size, sigma, channels=3):
    """ Generates a 2D Gaussian kernel. (Anisotropic sigma)
    """
    kx, ky = kernel_size
    sigma_x, sigma_y = sigma

    x_coord = torch.arange(kx)
    y_coord = torch.arange(ky)
    y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing='ij')
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()  # (ky, kx, 2)

    mean_x = (kx - 1) / 2.0
    mean_y = (ky - 1) / 2.0

    exponent = -((xy_grid[..., 0] - mean_x) ** 2 / (2 * sigma_x ** 2) +
                 (xy_grid[..., 1] - mean_y) ** 2 / (2 * sigma_y ** 2))
    gaussian_kernel = torch.exp(exponent)

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, ky, kx)  # (1, 1, ky, kx)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)  # (C, 1, ky, kx)

    return gaussian_kernel


def get_sized_image(image, size=(16, 16), mode="area", gaussian_factor=1):
    """
    Args:
        image: (B, 1, H, W)
        size:
    """
    B, C, H, W = image.shape

    if not isinstance(size, tuple):
        size = (size, size)

    # tip: sigma 是下采样倍率的一半，比如 image_size 是 (1024, 1024)，size 是 (256, 128)，则 sigma = (2, 4)
    image_size = image.shape[-2:]
    sigma = (image_size[0] / size[0] / 2, image_size[1] / size[1] / 2)

    # tip: 先使用 Gaussian 模糊再使用下采样的方式，可以有效防止摩尔纹现象的出现（同时会丢失一些细节）
    kernel_size = (round(gaussian_factor * sigma[0]) + 1, round(gaussian_factor * sigma[1]) + 1)  # 核大小 ≈ 4σ
    kernel = gaussian_kernel_2d(kernel_size, sigma, C).to(image.device)

    # Gaussian 模糊
    pad_x = kernel_size[0] // 2
    pad_y = kernel_size[1] // 2
    blurred = F.conv2d(
        image,
        kernel,
        stride=1,
        padding=(pad_x, pad_y),
        groups=C
    )

    # 下采样
    downsampled = F.interpolate(
        blurred,
        size=size,
        mode=mode # 使用 area 而不使用其他的原因为：计算简单
    )
    return downsampled


def get_scaled_image(image, down_scale=2, gaussian_factor=1):
    """
    Args:
        image: (B, 1, H, W)
        down_scale:
    """
    B, C, H, W = image.shape
    sigma = down_scale / 2
    image_size = image.shape[-2:]
    output_size = (image_size[0] // down_scale, image_size[1] // down_scale)

    # tip: 先使用 Gaussian 模糊再使用下采样的方式，可以有效防止摩尔纹现象的出现（同时会丢失一些细节）
    kernel_size = round(gaussian_factor * sigma) + 1
    kernel = gaussian_kernel(kernel_size, sigma, C).to(image.device)

    # Gaussian 模糊
    padding = kernel_size // 2
    blurred = F.conv2d(
        image,
        kernel,
        stride=1,
        padding=padding,
        groups=C
    )

    # 下采样
    downsampled = F.interpolate(
        blurred,
        size=output_size,
        mode='area'
    )
    return downsampled


#region [RGB to LAB]
def rgb_to_lab(image_rgb: torch.Tensor) -> torch.Tensor:
    """
    Input: image_rgb (shape [..., 3, H, W], range [0, 1])
    Output: image_lab (shape [..., 3], L∈[0,100], a/b∈[-128,127])
    """
    # sRGB to XYZ transformation matrix (D65 white point)
    rgb_to_xyz = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=torch.float32)

    # D65 white point (XYZ normalization reference values)
    D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)

    # Threshold and coefficients for LAB conversion
    epsilon = 0.008856  # (6/29)^3
    kappa = 903.3  # (29/3)^3

    # Ensure input is within [0, 1] range
    image_rgb = torch.clamp(image_rgb, 0.0, 1.0)
    image_rgb = rearrange(image_rgb, 'b c h w -> b h w c')

    # 1. Linearize RGB (remove gamma correction)
    mask = image_rgb > 0.04045
    image_linear = torch.where(
        mask,
        ((image_rgb + 0.055) / 1.055) ** 2.4,
        image_rgb / 12.92
    )

    # 2. RGB → XYZ (matrix multiplication)
    xyz = torch.einsum('...c,rc->...r', image_linear, rgb_to_xyz.to(image_rgb.device))

    # 3. XYZ → LAB
    # Normalize to D65 white point
    xyz_normalized = xyz / D65.to(image_rgb.device)

    # Nonlinear transform f(t)
    mask = xyz_normalized > epsilon
    f_xyz = torch.where(
        mask,
        xyz_normalized ** (1/3),
        (kappa * xyz_normalized + 16) / 116
    )

    # Calculate L*, a*, b*
    L = (116 * f_xyz[..., 1] - 16) / 100                        # L ∈ [0, 100]      -> [0, 1]
    a = (500 * (f_xyz[..., 0] - f_xyz[..., 1]) + 128) / 255     # a ≈ [-128, 127]   -> [0, 1]
    b = (200 * (f_xyz[..., 1] - f_xyz[..., 2]) + 128) / 255     # b ≈ [-128, 127]   -> [0, 1]

    # Combine channels
    return torch.stack([L, a, b], dim=-3)
#endregion


def get_illumination_map(image):
    """
    Args:
        image: (B, 3, H, W)
    """
    assert image.shape[1] == 3, "Image must be in RGB format"
    lab = rgb_to_lab(image)
    l = lab[:, 0:1, :, :]
    l = torch.clamp(l, 0, 1)

    return l
#endregion