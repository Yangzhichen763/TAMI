import torch
import torch.nn.functional as F
import math
import cv2
import numpy as np

from .base import TransformBase

"""
Modified from https://www.shadertoy.com/view/ltffzl
and https://github.com/yumayanagisawa/Unity-Raindrops/blob/master/Raindrop/Assets/Raindrop.shader
"""


class RainDropTransform(TransformBase):
    def __init__(
            self, p=1.0,
            time_second=0.0, rain_amount=0.8,
            gradient_method="sobel", normal_strength=1.0,
            with_blur=True, blur_strength=(3, 6),
            with_vign=False, with_cold_balance=False,
            return_uv=False,
    ):
        super().__init__(p)
        self.time_second = time_second
        self.rain_amount = rain_amount
        self.gradient_method = gradient_method
        self.normal_strength = normal_strength
        self.with_blur = with_blur
        self.blur_strength = blur_strength
        self.with_vign = with_vign
        self.with_cold_balance = with_cold_balance
        self.return_uv = return_uv

    def call(self, img):
        # 输入 img (B, C, H, W), 0..1
        out = render_raindrops(
            img, time_sec=self.time_second, rain_amount=self.rain_amount, seed=114514,

            gradient_method=self.gradient_method, normal_strength=self.normal_strength,
            with_blur=self.with_blur, blur_strength=self.blur_strength,
            return_uv=self.return_uv
        )
        return out


#region == [基础工具]==
def smooth_step(a, b, t):
    eps = 1e-6
    if isinstance(t, torch.Tensor):
        x = torch.clamp((t - a) / (b - a + eps), 0.0, 1.0)
    else:
        x = np.clip((t - a) / (b - a + eps), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def unbounded_smooth_step(a, b, t, strength=None):
    eps = 1e-6
    if strength is None:
        strength = math.e
    x = torch.clamp_min((t - a) / (b - a + eps), 0.0)
    return torch.tanh(torch.log(strength * x + eps) + 1) / (math.tanh(math.log(strength)) + 1 + eps)

def frac(x):
    if isinstance(x, torch.Tensor):
        return x - torch.floor(x)
    else:
        return np.modf(x)[0]

def gaussian_kernel2d(size, sigma, device="cpu", dtype=torch.float32):
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel   # (k, k)

def gaussian_blur(img, sigma=1.2, kernel_size=3):
    """
    Args:
        img: (B, C, H, W) or (C, H, W)
    """
    # 维度规范化
    if img.ndim == 3:
        # [C, H, W] → [1, C, H, W]
        img = img.unsqueeze(0)
    elif img.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected input shape: {img.shape}")

    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    # 生成高斯核
    g2d = gaussian_kernel2d(kernel_size, sigma, device=device, dtype=dtype)  # (k, k)

    # 卷积
    g2d = g2d.expand(C, 1, -1, -1)  # 每通道独立卷积
    radius = kernel_size // 2
    blurred = F.conv2d(img, g2d, padding=radius, groups=C)
    return blurred  # (B, C, H, W)

def make_gauss_pyramid(img, levels=6):
    """
    构建高斯金字塔
        img: torch.Tensor (1, 3, H, W), float32, [0,1]
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    pyr = [img]

    for _ in range(1, levels):
        blur = gaussian_blur(img, sigma=1.2, kernel_size=5)         # (B, C, H, W)
        down = F.interpolate(blur, scale_factor=0.5, mode="area")   # (B, C, H/2, W/2)
        img = down
        pyr.append(img)
    return pyr

def bilinear_sample(img, u, v):
    """

    Args:
        img: (B, C, H, W) or (C, H, W)
        u: (B, H, W) or (H, W)
        v: (B, H, W) or (H, W)
    """
    if img.ndim == 3:
        # [C, H, W] → [1, C, H, W]
        img = img.unsqueeze(0)
    elif img.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected input shape: {img.shape}")

    if u.ndim == 2:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)

    H, W = img.shape[-2:]
    uu = u * 2 - 1
    vv = v * 2 - 1
    grid = torch.stack([uu, vv], -1) # [B, H, W, 2]
    out = F.grid_sample(img, grid, align_corners=True)
    return out  # (B, C, H, W)

def sample_lod(pyramid, u, v, lod):
    """
    Args:
        pyramid: list of torch.Tensor (B, C, H, W), float32, [0,1]
        u, v: (B, H, W)  -- normalized coords in [0,1]
        lod:  (B, H, W)  -- fractional level-of-detail
    Returns:
        out: (B, C, H, W)
    """
    B, H, W = u.shape
    L = len(pyramid)

    lod = torch.clamp(lod, 0.0, L - 1 - 1e-6)
    lod0 = torch.floor(lod).long()
    lod1 = torch.clamp(lod0 + 1, 0, L - 1)
    w = (lod - lod0.float())  # (B, 1, H, W)

    # 输出初始化
    out = torch.zeros(
        (B, pyramid[0].shape[1], H, W),
        dtype=pyramid[0].dtype, device=pyramid[0].device
    )

    for lvl in range(L - 1):
        mask0 = (lod0 == lvl)  # (B, 1, H, W)
        mask1 = (lod1 == lvl)
        if mask0.any():
            c0 = bilinear_sample(pyramid[lvl], u, v)  # (B, C, H, W)
            out += c0 * (mask0.float() * (1 - w))
        if mask1.any():
            c1 = bilinear_sample(pyramid[lvl + 1], u, v)
            out += c1 * (mask1.float() * w)
    return out

def get_normal(h_map, method="central", strength=1.0):
    """
    Args:
        h_map: (B, 1, H, W) height map
        method: 'sobel', 'central', 'gradient'
        strength: scale factor for gradient strength
    Returns:
        normal: (B, 2, H, W)  # [dx, dy]
    """
    if h_map.ndim == 2:
        h_map = h_map.unsqueeze(0).unsqueeze(1) # -> (1, 1, H, W)
    elif h_map.ndim == 3:
        h_map = h_map.unsqueeze(0)              # -> (1, 1, H, W)
    elif h_map.ndim == 4:
        pass
    else:
        raise ValueError(f"Input h_map must be (B, H, W) or (H, W), but got {h_map.shape}")

    if method.lower() == "sobel":
        sobel_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=h_map.dtype, device=h_map.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)

        gx = F.conv2d(h_map, sobel_x, padding=1)
        gy = F.conv2d(h_map, sobel_y, padding=1)
    elif method.lower() == "central":
        gx = torch.roll(h_map, shifts=-1, dims=-1) - torch.roll(h_map, shifts=1, dims=-1)
        gy = torch.roll(h_map, shifts=-1, dims=-2) - torch.roll(h_map, shifts=1, dims=-2)
    elif method.lower() == "gradient":
        gy, gx = torch.gradient(h_map, dim=(-2, -1))
    else:
        raise ValueError(f"Unknown method: {method}")
    normal = torch.cat([gx, gy], dim=-3) * strength
    return normal   # (B, 2, H, W)
#endregion

#region ==[随机噪声（对应着色器里的 N13/N14）]==
def n13(p):
    # 近似 BigWings 的 N13：给定标量 p -> 3 维伪随机
    p3 = (np.modf(np.array([p, p, p]) * np.array([.1031, .11369, .13787]))[0]).astype(np.float32)
    p3 = p3 + np.dot(p3, (p3[[1, 2, 0]] + 19.19))
    r = (np.modf(np.array([(p3[0] + p3[1]) * p3[2],
                           (p3[0] + p3[2]) * p3[1],
                           (p3[1] + p3[2]) * p3[0]]))[0]).astype(np.float32)
    return r

def n_scalar(t):
    return torch.frac(torch.sin(t * 12345.564) * 7658.76)
#endregion

#region ==[单层动态雨滴（与 shader 对齐的形状/拖尾逻辑）]==
def drop_layer2(uv, t):
    """
    Args:
        uv: (B, 2, H, W) normalized coords in [0,1]
        t:  float or Tensor
    Returns:
        (B, 2, H, W): [mask, trail]
    """
    B, C, H, W = uv.shape
    device = uv.device

    # 为了对齐 shader，把[0,1]坐标系里的网格逻辑搬过来：
    UV = uv.clone()
    grid_mul = torch.tensor([6.0, 1.0], device=device)
    grid = grid_mul * 2.0

    u = uv[:, 0]
    v = uv[:, 1]

    uv_y_shift = v + t * 0.75
    id_x = torch.floor(u * grid[0]).to(torch.int32)
    id_y = torch.floor(uv_y_shift * grid[1]).to(torch.int32)

    # 每列随机相位（向量化的 n_scalar）
    # 原：colShift = vectorize(n_scalar)(id_x)；这里改为连续映射：sin/hash
    colShift = n_scalar(id_x)
    uv_y_shift2 = uv_y_shift + colShift
    id_x2 = torch.floor(u * grid[0]).to(torch.int32)
    id_y2 = torch.floor(uv_y_shift2 * grid[1]).to(torch.int32)

    key = id_x2 * 352 + id_y2
    kf = key.float()

    n0 = torch.frac(torch.sin(kf * 0.0131) * 43758.5453)
    n1 = torch.frac(torch.sin(kf * 0.11369) * 24634.6345)
    n2 = torch.frac(torch.sin(kf * 0.13787) * 36243.2342)
    n = torch.stack([n0, n1, n2], dim=1)  # (B, 3, H, W)

    st_x = frac(u * grid[0]) - 0.5
    st_y = frac(uv_y_shift2 * grid[1])
    st = torch.stack([st_x, st_y], dim=1)  # (B, 2, H, W)

    # 核心形状与摆动
    x = n[:, 0] - 0.5
    y = UV[:, 1] * 20.0
    wiggle = torch.sin(y + torch.sin(y))
    x = x + wiggle * (0.5 - torch.abs(x)) * (n[:, 2] - 0.5)
    x = x * 0.7
    ti = frac(t + n[:, 2])
    y = (smooth_step(0.85, 1.0, ti) - 0.5) * 0.9 + 0.5
    p = torch.stack([x, y], dim=1)  # (B, 2, H, W)

    # 距离 → 主滴
    a = torch.tensor([6.0, 1.0], device=device)
    diff = (st - p) * a.flip(0).view(2, 1, 1)  # (B,2,H,W)
    d = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    mainDrop = smooth_step(0.4, 0.0, d)

    # 拖尾
    r = torch.sqrt(smooth_step(1.0, y, st[:, 1]))
    cd = torch.abs(st[:, 0] - x)
    trail = smooth_step(0.23 * r, 0.15 * r * r, cd)
    trailFront = smooth_step(-0.02, 0.02, st[:, 1] - y)
    trail = trail * trailFront * r * r

    # 小水珠（层内）
    y2 = UV[:, 1]
    trail2 = smooth_step(0.2 * r, 0.0, cd)
    droplets = torch.clamp_min(torch.sin(y2 * (1.0 - y2) * 120.0) - st[:, 1], 0.0) * trail2 * trailFront * n[:, 2]
    y3 = frac(y2 * 10.0) + (st[:, 1] - 0.5)
    dd = torch.sqrt((st[:, 0] - x) ** 2 + (st[:, 1] - y3) ** 2)
    droplets = smooth_step(0.3, 0.0, dd)

    m = mainDrop + droplets * r * trailFront
    out = torch.stack([m, trail], dim=1)  # (B, 2, H, W)
    return out

def static_drops(uv, t):
    """
    静态小水珠层（对应 shader 的 40 倍频网格）
    Args:
        uv: (B, 2, H, W), normalized [0,1]
        t:  float 或 Tensor
    Returns:
        c: (B, 1, H, W)
    """
    B, C, H, W = uv.shape
    device = uv.device

    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=uv.dtype, device=device)
    t = t.view(-1, 1, 1)  # (B,1,1) 可广播

    uv40 = uv * 40.0
    u, v = uv40[:, 0], uv40[:, 1]

    id_x = torch.floor(u).to(torch.int64)
    id_y = torch.floor(v).to(torch.int64)

    cell_x = frac(u) - 0.5
    cell_y = frac(v) - 0.5
    cell = torch.stack([cell_x, cell_y], dim=1)  # (B, 2, H, W)

    key = id_x * 10745 + id_y * 3543654
    kf = key.float()

    nx = torch.frac(torch.sin(kf * 0.0745) * 24375.234)
    ny = torch.frac(torch.sin(kf * 0.0453) * 17354.543)
    nz = torch.frac(torch.sin(kf * 0.0651) * 16345.213)

    p = (torch.stack([nx, ny], dim=1) - 0.5) * 0.7
    diff = cell - p
    d = torch.sqrt((diff ** 2).sum(dim=1))  # (B, H, W)

    # 保留原 fade 逻辑：Saw(0.025, frac(t + nz))
    fade = smooth_step(0.025, 1.0, frac(t + nz))
    c = smooth_step(0.3, 0.0, d) * frac(nz * 10.0) * fade

    return c.unsqueeze(1).float()  # (B, 1, H, W)

def drops(uv, t, l0, l1, l2):
    """
    组合：静态小水珠 + 两层动态雨滴
    Args:
        uv: (B, 2, H, W)
        t: float 或 Tensor
        l0, l1, l2: float (各层权重)
    Returns:
        out: (B, 2, H, W)
    """
    # 静态小滴层
    s = static_drops(uv, t) * l0  # (B, 1, H, W)

    # 两层动态雨滴
    m1 = drop_layer2(uv, t) * l1  # (B, 2, H, W)
    m2 = drop_layer2(uv * 1.85, t) * l2  # (B, 2, H, W)

    # 主体mask叠加
    c = s + m1[:, 0:1] + m2[:, 0:1]  # 保持(B,1,H,W)
    c = unbounded_smooth_step(0.3, 1.0, c, strength=1.0)

    # 拖尾层叠加
    trail = torch.maximum(m1[:, 1:2] * l0, m2[:, 1:2] * l1)

    return torch.cat([c, trail], dim=1).float()  # (B, 2, H, W)
#endregion

#region ==[主渲染]==
def render_raindrops(
        bg, time_sec=0.0, rain_amount=0.8, seed=0,

        gradient_method="sobel", normal_strength=1.0,
        with_blur=True, blur_strength=(3, 6),
        with_vign=False, with_cold_balance=False,
        return_uv=False,
):
    """
    Args:
        bg (torch.Tensor): (B, 3, H, W), RGB in [0,1]
    Returns:
        color: (B, 3, H, W)
    """
    torch.manual_seed(seed)
    device = bg.device
    dtype = bg.dtype
    B, C, H, W = bg.shape

    # 构造网格坐标
    u = torch.linspace(0, 1, W, device=device, dtype=dtype)
    v = torch.linspace(0, 1, H, device=device, dtype=dtype)
    U, V = torch.meshgrid(u, v, indexing="xy")        # (W, H)
    UV = torch.stack([U, V], dim=0)                   # (2, H, W)
    UV = UV.unsqueeze(0).repeat(B, 1, 1, 1)           # (B, 2, H, W)

    # 构造以短边为尺度、中心(0,0)的“物理”uv（匹配 shader 的畸变/缩放）
    aspect = W / float(H)
    uv = (UV - 0.5)
    uv[:, 0] *= aspect
    # 轻微的时间变焦
    zoom = -math.cos(time_sec * 0.2)
    uv = uv * (0.7 + zoom * 0.3)
    # 还原回 [0,1] 用于采样
    uv01 = uv.clone()
    uv01[:, 0] = (uv[:, 0] / aspect) + 0.5
    uv01[:, 1] = uv[:, 1] + 0.5

    # 雨量 -> 三层权重
    staticDrops = smooth_step(-0.5, 1.0, rain_amount) * 2.0
    layer1 = smooth_step(0.25, 0.75, rain_amount)
    layer2 = smooth_step(0.0, 0.5, rain_amount)

    # 计算高度场 h_map 与拖尾 trail
    h_map_trail = drops(uv01, time_sec * 0.2, staticDrops, layer1, layer2)  # (B, 2, H, W)
    h_map = h_map_trail[:, 0:1]
    trail = h_map_trail[:, 1:2]

    # 法线（折射位移）
    # h_map: (B, 1, H, W)
    normal = get_normal(h_map, method=gradient_method, strength=normal_strength)  # (B, 2, H, W)

    if with_blur:
        # 焦距 LOD：trail 越大越虚化
        max_blur_factor = blur_strength[1] * rain_amount + blur_strength[0] * (1.0 - rain_amount)  # 3~6
        min_blur_factor = 1.0
        focus = max_blur_factor - trail
        # 有水滴的位置更加清晰
        focus = focus - smooth_step(0.1, 0.2, h_map) * (max_blur_factor - min_blur_factor)
        focus = torch.clamp(focus, 0.0, max_blur_factor)
        # 保持划痕区域清晰
        trail_mask = smooth_step(0.0, 1.0, trail)  # trail 强度阈值映射
        focus = focus * (1.0 - trail_mask) + min_blur_factor * trail_mask
    else:
        focus = torch.zeros_like(h_map)

    # 构建金字塔并 LOD 采样（对应 tex2Dlod(iChannel0, UV + normal, focus)）
    pyr = make_gauss_pyramid(bg, levels=6)
    # 折射坐标：在原始 UV 上加法线（系数略缩）
    refr_u = torch.clamp(UV[:, 0] + normal[:, 0] * 0.30, 0.0, 1.0)
    refr_v = torch.clamp(UV[:, 1] + normal[:, 1] * 0.30, 0.0, 1.0)
    color = sample_lod(pyr, refr_u, refr_v, focus / 2.0)  # 简单映射：越大越低分辨

    # 色调平衡
    if with_cold_balance:
        tint = torch.tensor([0.8, 0.9, 1.3], device=device, dtype=dtype).view(1, 3, 1, 1)
        color = color * tint

    # 暗角
    if with_vign:
        UVs = (UV - 0.5)
        vign = 1.0 - (UVs[:, 0] ** 2 + UVs[:, 1] ** 2)
        vign = torch.clamp(vign, 0.0, 1.0)
        color = color * vign.unsqueeze(1)

    color = torch.clamp(color, 0, 1)
    if return_uv:
        uv_vis = torch.cat([
            refr_u.unsqueeze(1),
            refr_v.unsqueeze(1),
            torch.zeros_like(refr_u).unsqueeze(1)
        ], dim=1)
        return color, uv_vis
    return color
#endregion


if __name__ == "__main__":
    import sys
    sys.path.append('.')

    import torchvision.transforms as T

    from basic.utils.io import read_image_as_numpy, read_image_as_pil, save_image
    from basic.utils.convert import numpy2tensor, tensor2numpy

    def calcu_psnr(a, b):
        a = a.detach().cpu()
        b = b.detach().cpu()
        return (10 * torch.log10(1 / torch.mean((a - b) ** 2))). item()

    image_path = "~/Dataset/LLVE/DID-1080/test/high/video102/001.jpg"
    image = read_image_as_pil(image_path)
    clean_image = numpy2tensor(read_image_as_numpy(image_path))
    save_image(tensor2numpy(clean_image), '.tmp/clean_image.jpg')

    # # ================================================
    # # 可视化雨滴效果
    # # ================================================
    # transform = T.Compose([
    #     T.ToTensor(),
    #     RainDropTransform(
    #         time_second=12.0, rain_amount=0.85,
    #         gradient_method='sobel', normal_strength=0.5,
    #         with_blur=True, blur_strength=(3, 6),
    #         return_uv=True,
    #     ),
    # ])
    # raindrop_image, uv_vis = transform(image)
    # save_image(tensor2numpy(raindrop_image), '.tmp/raindrop_image.jpg')
    # save_image(tensor2numpy(uv_vis, reverse_channels=False), '.tmp/raindrop_image_uv_vis.jpg')
    # print(f"Raindrop PSNR: {calcu_psnr(clean_image, raindrop_image)}")
    #
    # import matplotlib.pyplot as plt
    #
    # # 可视化雨滴效果
    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(tensor2numpy(raindrop_image))
    # axes[0].axis('off')
    # axes[0].set_title("Raindrop Image")
    # axes[1].imshow(tensor2numpy(uv_vis, reverse_channels=False))
    # axes[1].axis('off')
    # axes[1].set_title("UV Visualization")
    # plt.show()


    # ================================================
    # 可视化传感器噪声：不同曝光时间和读取时间的组合
    # ================================================
    import matplotlib.pyplot as plt

    # 定义参数范围
    normal_strength_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 1.9]  # 行
    rain_amount_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 列

    n_rows = len(normal_strength_values)
    n_cols = len(rain_amount_values)

    # 每个 cell 两张图（rain + uv）
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(4 * n_cols, 2 * n_rows))

    for i, normal_strength in enumerate(normal_strength_values):
        for j, rain_amount in enumerate(rain_amount_values):
            transform = T.Compose([
                T.ToTensor(),
                RainDropTransform(
                    time_second=12.0, rain_amount=rain_amount,
                    gradient_method='sobel', normal_strength=normal_strength,
                    with_blur=True, blur_strength=(3, 6),
                    return_uv=True,
                ),
            ])

            raindrop_image, uv_vis = transform(image)
            np_img = tensor2numpy(raindrop_image)
            np_uv = tensor2numpy(uv_vis, reverse_channels=False)

            # 左图：雨滴渲染结果
            ax1 = axes[i, j * 2] if n_rows > 1 else axes[j * 2]
            ax1.imshow(np_img)
            ax1.axis('off')
            ax1.set_title(f"n={normal_strength:.1f}, r={rain_amount:.1f}", fontsize=8)

            # 右图：UV 可视化
            ax2 = axes[i, j * 2 + 1] if n_rows > 1 else axes[j * 2 + 1]
            ax2.imshow(np_uv)
            ax2.axis('off')
            ax2.set_title("UV", fontsize=8)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.tight_layout()
    plt.savefig('.tmp/grid_raindrop_image.pdf', format='pdf', dpi=200, bbox_inches='tight')
    plt.show()
