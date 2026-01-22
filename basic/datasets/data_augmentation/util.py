import torch
import torch.nn.functional as F


'''
Modified from (https://github.com/timothybrooks/unprocessing)
相机参数是根据 Darmstadt Noise Dataset 数据集设置的，详细内容见论文 "Unprocessing Images for Learned Raw Denoising"
'''


"""
sRGB -> RAW
"""
def random_ccm(device=None):
    """
    Generates random RGB -> Camera color correction matrices.
    """
    xyz2cams = torch.tensor([
        [[1.0234, -0.2969, -0.2266],
         [-0.5625, 1.6328, -0.0469],
         [-0.0703, 0.2188, 0.6406]],
        [[0.4913, -0.0541, -0.0202],
         [-0.6130, 1.3513, 0.2906],
         [-0.1564, 0.2151, 0.7183]],
        [[0.8380, -0.2630, -0.0639],
         [-0.2887, 1.0725, 0.2496],
         [-0.0627, 0.1427, 0.5438]],
        [[0.6596, -0.2079, -0.0562],
         [-0.4782, 1.3016, 0.1933],
         [-0.0970, 0.1581, 0.5181]]
    ], dtype=torch.float32, device=device)

    weights = torch.empty(4, 1, 1, device=device).uniform_(1e-8, 1e8)
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights.sum(dim=0)

    rgb2xyz = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=torch.float32, device=device)

    rgb2cam = xyz2cam @ rgb2xyz
    rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdim=True)
    rgb2cam = rgb2cam.unsqueeze(0)
    return rgb2cam


def random_gains(device=None):
    # 调整这个 rgb gain 会改动图像的整体 RGB 值
    # rgb_gain = 1.0 / torch.normal(mean=0.8, std=0.1, size=(1,), device=device)
    rgb_gain = 1.0
    red_gain = torch.empty((1,), device=device).uniform_(1.9, 2.4)
    blue_gain = torch.empty((1,), device=device).uniform_(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
    image = torch.clamp(image, 0.0, 1.0)
    return 0.5 - torch.sin(torch.arcsin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion(image):
    return torch.pow(torch.clamp(image, 1e-8, 1.0), 2.2)


def apply_ccm(image, ccm):
    """
    image: [B, 3, H, W], ccm: [B, 3, 3]
    """
    B, C, H, W = image.shape
    flat = image.permute(0, 2, 3, 1).reshape(-1, 3)
    out = (flat @ ccm.mT).reshape(B, H, W, 3).permute(0, 3, 1, 2)
    return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """
    image: [B, 3, H, W]
    """
    gains = torch.tensor([1.0 / red_gain, 1.0, 1.0 / blue_gain],
                         dtype=image.dtype, device=image.device) / rgb_gain
    gains = gains.view(1, 3, 1, 1)
    image = image * gains

    """
    使用原论文的方式，高光区域会出现色彩斑块问题
    """
    # gray = image.mean(dim=1, keepdim=True)
    # inflection = 0.9
    # mask = ((torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0)
    # safe_gains = torch.maximum(mask + (1.0 - mask) * gains, gains)
    # image = image * safe_gains
    return image


def mosaic(image):
    """
    RGB -> RGGB 4-channel Bayer
    image: [B, 3, H, W]  => [B, 4, H/2, W/2]
    """
    B, C, H, W = image.shape
    r = image[:, 0, 0::2, 0::2]
    g_r = image[:, 1, 0::2, 1::2]
    g_b = image[:, 1, 1::2, 0::2]
    b = image[:, 2, 1::2, 1::2]
    return torch.stack([r, g_r, g_b, b], dim=1)


def unprocess_srgb_to_raw(image):
    """
    Convert sRGB image [B,3,H,W] to synthetic RAW [B,4,H/2,W/2]
    """
    assert image.shape[1] == 3
    rgb2cam = random_ccm(device=image.device)
    cam2rgb = torch.inverse(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains(device=image.device)

    image = inverse_smoothstep(image)
    image = gamma_expansion(image)
    image = apply_ccm(image, rgb2cam)
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    image = torch.clamp(image, 0.0, 1.0)
    raw = mosaic(image)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return raw, metadata


"""
RAW -> sRGB
"""
def apply_gains(bayer_images, red_gain, blue_gain):
    """
    Args:
        bayer_images: (B, 4, H, W)
        red_gains, blue_gains: (B, )
    """
    B = bayer_images.shape[0]
    green_gain = torch.ones_like(red_gain)
    gains = torch.stack([red_gain, green_gain, green_gain, blue_gain], dim=-1).view(B, 4, 1, 1)
    return bayer_images * gains


def demosaic(bayer_images):
    """
    Bilinearly demosaic a batch of RGGB Bayer images.

    Args:
        bayer_images: (B, H, W, 4)
        return: (B, 2H, 2W, 3)
    """
    B, C, H, W = bayer_images.shape
    target_size = (H * 2, W * 2)

    # red
    red = F.interpolate(bayer_images[:, 0:1], size=target_size, mode='bilinear', align_corners=False)

    # green at red
    g_r = torch.flip(bayer_images[:, 1:2], dims=[3])
    g_r = F.interpolate(g_r, size=target_size, mode='bilinear', align_corners=False)
    g_r = torch.flip(g_r, dims=[3])
    g_r = F.pixel_unshuffle(g_r, 2)

    # green at blue
    g_b = torch.flip(bayer_images[:, 2:3], dims=[2])
    g_b = F.interpolate(g_b, size=target_size, mode='bilinear', align_corners=False)
    g_b = torch.flip(g_b, dims=[2])
    g_b = F.pixel_unshuffle(g_b, 2)

    g_at_r = (g_r[:, 0] + g_b[:, 0]) / 2
    g_at_gr = g_r[:, 1]
    g_at_gb = g_b[:, 2]
    g_at_b = (g_r[:, 3] + g_b[:, 3]) / 2
    greens = torch.stack([g_at_r, g_at_gr, g_at_gb, g_at_b], dim=1)
    green = F.pixel_shuffle(greens, 2)

    # blue
    blue = torch.flip(bayer_images[:, 3:4], dims=[2, 3])
    blue = F.interpolate(blue, size=target_size, mode='bilinear', align_corners=False)
    blue = torch.flip(blue, dims=[2, 3])

    return torch.cat([red, green, blue], dim=1)


def apply_ccms(images, ccms):
    """
    Args:
        images: (B, H, W, 3)
        ccms: (B, 3, 3)
    """
    B, C, H, W = images.shape
    flat = images.permute(0, 2, 3, 1).reshape(B, -1, 3)
    out = torch.bmm(flat, ccms.transpose(1, 2))
    return out.view(B, H, W, 3).permute(0, 3, 1, 2)


def gamma_compression(images, gamma=2.2):
    return torch.pow(torch.clamp(images, 1e-8, 1.0), 1.0 / gamma)

def apply_smoothstep(image):
    return 3 * image ** 2 - 2 * image ** 3


def process_raw_to_srgb(bayer_images, red_gain, blue_gain, cam2rgb):
    """
    Process a batch of Bayer images to sRGB images.

    Args:
        bayer_images: (B, H, W, 4)
        red_gains, blue_gains: (B, )
        cam2rgb: (B, 3, 3)

    Returns:
        (B, 2H, 2W, 3)
    """
    # white balance
    bayer_images = apply_gains(bayer_images, red_gain, blue_gain)
    bayer_images = torch.clamp(bayer_images, 0.0, 1.0)

    # demosaic
    images = demosaic(bayer_images)

    # color correction
    images = apply_ccms(images, cam2rgb)
    images = torch.clamp(images, 0.0, 1.0)

    # gamma compression
    images = gamma_compression(images)

    # apply smoothstep
    images = apply_smoothstep(images)

    return images

