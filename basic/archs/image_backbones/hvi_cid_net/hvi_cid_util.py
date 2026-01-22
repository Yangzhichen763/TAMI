import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basic.archs.modules.norm import LayerNorm2d
from basic.utils.registry import MODULE_REGISTRY


'''
Adapted from HVI-CIDNet(https://github.com/Fediory/HVI-CIDNet/blob/master/net/HVI_transform.py, https://github.com/Fediory/HVI-CIDNet/blob/master/net/LCA.py)
'''


#region ==[HVI Transform]==
pi = 3.141592653589793


@MODULE_REGISTRY.register()
class RGB2HVI(nn.Module):
    def __init__(self, scaled_on_start = False, scaled_on_end = False, alpha_start = 1.3, alpha_end = 1.0):
        super(RGB2HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned

        self.scaled_on_start = scaled_on_start
        self.scaled_on_end = scaled_on_end
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

        self.k = 0

    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.scaled_on_start:
            s = s * self.alpha_start

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.scaled_on_end:
            rgb = rgb * self.alpha_end
        return rgb
#endregion


#region ==[LCA]==
# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: (B, C, H, W)
            y: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))        # (B, C, H, W) -> (B, C, H, W)
        kv = self.kv_dwconv(self.kv(y))     # (B, C, H, W) -> (B, 2*C, H, W)
        k, v = kv.chunk(2, dim=1)           # (B, 2*C, H, W) -> (B, C, H, W), (B, C, H, W)

        # (B, C, H, W) == (B, head*c, H, W) -> (B, head, c, H*W)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # (B, head, c, H*W)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # (B, head, c, H*W) @ (B, head, H*W, c) -> (B, head, c, c)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        # (B, head, c, c) @ (B, head, c, H*W) -> (B, head, c, H*W)
        out = (attn @ v)

        # (B, head, c, H*W) -> (B, head*c, H, W) == (B, C, H, W)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)         # (B, C, H, W) -> (B, C, H, W)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        x = self.project_in(x)                  # (B, C, H, W) -> (B, 2*dim, H, W)
        x1, x2 = self.dwconv(x).chunk(2, dim=1) # (B, 2*dim, H, W) -> (B, dim, H, W), (B, dim, H, W)
        x1 = self.Tanh(self.dwconv1(x1)) + x1   # (B, dim, H, W)
        x2 = self.Tanh(self.dwconv2(x2)) + x2   # (B, dim, H, W)
        x = x1 * x2
        x = self.project_out(x)                 # (B, dim, H, W) -> (B, C, H, W)
        return x


# HV Lighten Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.norm = LayerNorm2d(dim, 1, 1)
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.ffn = CAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: (B, C, H, W)
            y: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        x = x + self.ffn(self.norm(x), self.norm(y))     # (B, C, H, W)
        x = self.gdfn(self.norm(x))                      # (B, C, H, W)
        return x


# Intensity Lighten Cross Attention
class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm2d(dim, 1, 1)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: (B, C, H, W)
            y: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        x = x + self.ffn(self.norm(x), self.norm(y))     # (B, C, H, W)
        x = x + self.gdfn(self.norm(x))                  # (B, C, H, W)
        return x
#endregion


#region ==[Basic]==
class NormDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm2d(out_ch, 1, 1)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )

    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x


class NormUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm2d(out_ch, 1, 1)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x
#endregion