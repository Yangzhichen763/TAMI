"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
y - usually means features that also depends on the illumination map.

The trailing number of a variable usually denote the stride

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.metrics.summary import is_summary

from basic.utils.console.log import get_root_logger
from basic.archs.tami_icmem.util import get_illumination_map
from basic.archs.tami_icmem.module.ghostnet import (
    # GhostNetFeatureFusionBlock as FeatureFusionBlock,
    GhostBottleneck as Bottleneck,
    GhostConvBlock as ConvBlock,
)
from basic.archs.tami_icmem.module.resnet import (
    MobileFeatureFusionBlock as FeatureFusionBlock,
    # ResFeatureFusionBlock as FeatureFusionBlock,
    # ResBottleneck as Bottleneck,
    # MobileBlock as ConvBlock,
)

from basic.utils.shared_pool import SharedPool
from basic.utils.console.log import is_debug


def interpolate(y, ratio, mode, align_corners):
    if ratio == 1:
        return y
    y = F.interpolate(y, scale_factor=ratio, mode=mode, align_corners=align_corners)
    return y


def interpolate_as(y, x, mode='area', align_corners=None):
    if y.shape[-2:] == x.shape[-2:]:
        return y
    y = F.interpolate(y, size=x.shape[-2:], mode=mode, align_corners=align_corners)
    return y


def upsample(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate(g, ratio, mode, align_corners)


def downsample(g, ratio=1 / 2, mode='area', align_corners=None):
    return interpolate(g, ratio, mode, align_corners)


#region [Encoder]
class HiddenReinforcer(nn.Module):
    """
    Used in the value encoder, a single GRU
    """
    def __init__(self, y_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = nn.Conv2d(y_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, y, h):
        y = torch.cat([y, h], 1)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an
        # implementation error that I never bothered fixing
        values = self.transform(y)
        forget_gate = torch.sigmoid(values[:, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h

smtc_dim = 256
smtc_scale = 16

class ValueEncoder(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        self.backbone = backbone

        if backbone =='resnet18':
            from .resnet import build_resnet18
            network = build_resnet18(pretrained=True, extra_dim=3)
            # make it a list to avoid saving the weight
            self.features = [
                network.conv1,
                network.bn1,
                network.relu,  # 1/2, 64
                network.maxpool,
                network.layer1,  # 1/4, 64
                network.layer2,  # 1/8, 128
                network.layer3,  # 1/16, 256
            ]
        elif backbone =='mobilenet_v2':
            from .mobilenetv2 import build_mobilenet_v2
            network = build_mobilenet_v2(pretrained=True, extra_dim=3)
            # make it a list to avoid saving the weight
            self.features = list(network.features[:-1])
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for layer in self.features:
            layer.to(*args, **kwargs)

    def get_layer_params(self):
        if self.backbone == 'resnet18':
            return self.features[-3:]
        elif self.backbone == 'mobilenet_v2':
            return self.features[1:-1]
        else:
            raise ValueError(f'Unsupported backbone: {self.backbone}')

    def set_layer_params(self, params):
        if self.backbone =='resnet18':
            self.features[-3:] = params
        elif self.backbone =='mobilenet_v2':
            self.features[1:-1] = params
        else:
            raise ValueError(f'Unsupported backbone: {self.backbone}')

    def forward(self, lq, hq):
        y = torch.cat([hq, lq], dim=1)  # (B, C1, H, W) + (B, C2, H, W) -> (B, C1, H, W)
        for layer in self.features:
            y = layer(y)
        return y


class KeyEncoder(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()

        self.backbone = backbone

        global smtc_dim
        global smtc_scale
        if backbone == 'resnet18':
            from .resnet import build_resnet18
            network = build_resnet18(pretrained=True)
            self.features = nn.Sequential(
                network.conv1,
                network.bn1,
                network.relu,  # 1/2, 64
                network.maxpool,
                network.layer1,  # 1/4, 64
                network.layer2,  # 1/8, 128
                network.layer3,  # 1/16, 256
            )

            smtc_dim = 256
            smtc_scale = 16
        elif backbone == 'mobilenet_v2':
            from .mobilenetv2 import build_mobilenet_v2
            network = build_mobilenet_v2(pretrained=True)
            self.features = network.features[:-1]

            smtc_dim = 320
            smtc_scale = 32
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

    def get_layer_params(self):
        if self.backbone == 'resnet18':
            return self.features[-3:]
        elif self.backbone == 'mobilenet_v2':
            return self.features[1:-1]
        else:
            raise ValueError(f'Unsupported backbone: {self.backbone}')

    def forward(self, lq):
        x = self.features(lq)
        return x


class IlluminationEncoder(nn.Module):
    def __init__(self, illu_feat_dim=64, bias=True, version=None):
        super().__init__()

        self.version = version.lower() if version is not None else None

        illu_dim = 1
        if self.version in ['mean', 'gray', 'yuv', 'brightness', 'empty']:
            pass
        elif self.version == 'encode_with_gray':
            self.conv1 = nn.Conv2d(3 + illu_dim, illu_feat_dim, kernel_size=1, bias=bias)
        elif self.version is None:
            self.conv1 = nn.Conv2d(3, illu_feat_dim, kernel_size=1, bias=bias)
        else:
            raise ValueError(f'Unsupported version: {self.version}')

        self.dwconv = nn.Conv2d(illu_feat_dim, illu_feat_dim, kernel_size=5, padding=2, bias=bias, groups=illu_feat_dim)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(illu_feat_dim, illu_dim, kernel_size=1, bias=bias)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # if self.version in ['encode_with_gray', None]:
        #     # 初始化为 0
        #     self.conv1.bias.data.zero_()
        #     self.dwconv.bias.data.zero_()
        #     self.conv2[1].bias.data.zero_()
        pass

    def forward(self, hq, size_as=None, return_illu_feats=False):
        hq = interpolate_as(hq, size_as) if size_as is not None else hq

        if self.version in ['encode_with_gray', None]:  # 参考 Retinexformer 中的做法
            mean_hq = hq.mean(dim=1, keepdim=True)      # (B, 3, H, W) -> (B, 1, H, W)
            if self.version == 'encode_with_gray':
                hq = torch.cat([hq, mean_hq], dim=1)
            illu_feat = self.dwconv(self.conv1(hq))
            illu_gate = self.conv2(illu_feat)
            illu = torch.clamp(illu_gate, 0, 1)
            if self.training:
                SharedPool.get('scalars').append('illu_gate', illu_gate.mean().detach().cpu().item())
        elif self.version in ['gray', 'mean']:
            illu = hq.mean(dim=1, keepdim=True)      # (B, 3, H, W) -> (B, 1, H, W)
            illu_feat = illu
        elif self.version == 'yuv':
            illu = get_illumination_map(hq)           # (B, 3, H, W) -> (B, 1, H, W)
            illu_feat = illu
        elif self.version == 'brightness':
            r, g, b = hq[:, 0:1, :, :], hq[:, 1:2, :, :], hq[:, 2:3, :, :]
            illu = 0.2126 * r + 0.7152 * g + 0.0722 * b  # MABD 公式中的亮度值
            illu_feat = illu
        elif self.version == 'empty':
            illu = torch.zeros_like(hq[:, 0:1, :, :])
            illu_feat = illu
        else:
            raise ValueError(f'Unsupported version: {self.version}')

        if return_illu_feats:
            return illu, illu_feat
        else:
            return illu


class KeyProjector(nn.Module):
    def __init__(self, in_dim=smtc_dim, key_dim=32):
        super().__init__()

        self.out_proj = nn.Conv2d(in_dim, key_dim, kernel_size=1)
        # self.out_proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)
        # # shrinkage
        # self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # # selection
        # self.e_proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.out_proj.weight.data)
        nn.init.zeros_(self.out_proj.bias.data)

    def forward(self, x, need_s=False, need_e=False):
        # shrinkage = self.d_proj(x) ** 2 + 1 if need_s else None
        # selection = torch.sigmoid(self.e_proj(x)) if need_e else None
        #
        # return self.out_proj(x), shrinkage, selection
        return self.out_proj(x), None, None


class Projector(nn.Module):
    def __init__(self, in_dim=smtc_dim, out_dim=32, type='dwconv'):
        super().__init__()

        if type.lower() == 'dwconv':
            self.out_proj = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
                # 由于紧跟着的 Key Prj 和 Value Prj 是线性层，所以可以这样顺序的
            )
        elif type.lower() == 'conv':
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        elif type.lower() == 'bottleneck':
            self.out_proj = Bottleneck(in_dim, out_dim, dw_kernel_size=3)
        else:
            raise ValueError(f'Unsupported projector type: {type.lower()}')

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.out_proj, nn.Conv2d):
            nn.init.orthogonal_(self.out_proj.weight.data)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias.data)
        else:
            for layer in self.out_proj.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight.data)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias.data)

        # nn.init.orthogonal_(self.out_proj.weight.data)
        # nn.init.zeros_(self.out_proj.bias.data)

    def forward(self, x):
        return self.out_proj(x)


class ValueProjector(nn.Module):
    def __init__(self, in_dim=smtc_dim, value_dim=32, hidden_dim=32):
        super().__init__()

        # short term memory reinforcer
        self.fuser = FeatureFusionBlock(in_dim, in_dim, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, z, smtc_key, hidden_state=None, any_hidden_in=False):
        z = self.fuser(smtc_key, z)   # (B, x_dim, H, W) + (B, y_dim, H, W) -> (B, y_dim, H, W)

        if any_hidden_in and self.hidden_reinforce is not None:
            hidden_state = self.hidden_reinforce(z, hidden_state) # (B, hidden_dim, H, W)

        return z, hidden_state
#endregion


#region [Decoder]
class HiddenUpdater(nn.Module):
    """
    Used in the decoder, multi-scale feature + GRU
    """
    def __init__(self, y_dims=(256, 1), mid_dim=32, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(y_dim, mid_dim, kernel_size=1)
            for y_dim in y_dims
        ])
        self.transform = nn.Conv2d(mid_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, ys, h):
        """
        Args:
            ys (torch.Tensor): (B, y_dim, H, W)
            h (torch.Tensor): (B, h_dim, H, W)
        """
        ys = torch.stack([
            self.convs[i](interpolate_as(ys[i], ys[0]))
            for i in range(len(self.convs))
        ], dim=1)
        y = torch.sum(ys, dim=1)
        y = torch.cat([y, h], 1)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an
        # implementation error that I never bothered fixing
        values = self.transform(y)
        z, r, h = values.chunk(3, dim=1)
        forget_gate = torch.sigmoid(z)
        update_gate = torch.sigmoid(r)
        new_value = torch.tanh(h)
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h


class UpSampling(nn.Module):
    def __init__(self, in_dim, out_dim, scale=2):
        super().__init__()
        self.scale = scale

        self.conv = nn.Conv2d(in_dim, out_dim * (scale**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(
            self, skip_dim, illu_in_dim, illu_out_dim,
            scale_factor=2, kernel_size=7
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding = kernel_size // 2

        self.skip_conv = nn.Conv2d(skip_dim, illu_in_dim, kernel_size=3, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(illu_in_dim, illu_out_dim, kernel_size=1, bias=True),
            nn.Conv2d(illu_out_dim, illu_out_dim, kernel_size=kernel_size, padding=padding, bias=True, groups=illu_out_dim)
        )
        self.scale_factor = scale_factor

    def forward(self, skip_z, illu_feat):
        skip_z = self.skip_conv(skip_z)
        g = upsample(illu_feat, ratio=self.scale_factor)
        # print(4, skip_z.shape, g.shape)
        g = skip_z + g
        g = self.out_conv(g)
        return g


class IlluminationRefiner(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            in_dim=smtc_dim,
            readout_dim=1, illu_feat_dim=16, hidden_dim=32,
            bias=True, type='dwconv',
    ):
        super().__init__()
        self.readout_dim = readout_dim

        if type.lower() == 'dwconv':
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=bias),
                nn.Conv2d(in_dim, illu_feat_dim, kernel_size=1),
            )
        elif type.lower() == 'conv_block':
            self.out_conv = ConvBlock(in_dim, illu_feat_dim)
        elif type.lower() in ['conv', 'conv3x3']:
            self.out_conv = nn.Conv2d(in_dim, illu_feat_dim, kernel_size=3, padding=1)
        elif type.lower() == 'conv1x1':
            self.out_conv = nn.Conv2d(in_dim, illu_feat_dim, kernel_size=1)
        elif type.lower() == 'bottleneck':
            self.out_conv = Bottleneck(in_dim, illu_feat_dim, dw_kernel_size=3)
        else:
            raise ValueError(f'Unsupported projector type: {type.lower()}')

        # short term memory update
        self.fuser = FeatureFusionBlock(
            x_in_dim=in_dim, y_in_dim=readout_dim + hidden_dim,
            y_mid_dim=in_dim, y_out_dim=in_dim
        )   # x_dim + value_dim + hidden_dim -> y_dim
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater(
                y_dims=(in_dim, illu_feat_dim),
                mid_dim=hidden_dim * 4, hidden_dim=hidden_dim
            )
        else:
            self.hidden_update = None

    def forward(self, smtc_feat, memory_readout, hidden_state, hidden_out=True):
        if memory_readout.shape[1] == 1:
            memory_readout = memory_readout.repeat(1, self.readout_dim, 1, 1)

        if self.hidden_update is not None:
            _smtc_feat = self.fuser(smtc_feat, torch.cat([memory_readout, hidden_state], 1))
        else:
            _smtc_feat = self.fuser(smtc_feat, memory_readout)

        illu_feat = self.out_conv(_smtc_feat)
        # 如果输出是亮度图，则需要限制在 0-1 范围内
        # illu_feat = torch.sigmoid(illu_feat)

        if hidden_out and self.hidden_update is not None:
            hidden_state = self.hidden_update([_smtc_feat, illu_feat], hidden_state)
        else:
            hidden_state = None

        return illu_feat, hidden_state
#endregion


#region [Attention]
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNormBase(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNormBase, self).__init__()
        if not isinstance(normalized_shape, tuple):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1, "only support single dimension input"

    def forward(self, x):
        raise NotImplementedError


#region ==[LayerNorm]==
class ParameterFree_LayerNorm(LayerNormBase):
    def __init__(self, normalized_shape):
        super(ParameterFree_LayerNorm, self).__init__(normalized_shape)

        self.normalized_shape = normalized_shape

    def forward(self, x, eps=1e-6):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + eps)


class BiasFree_LayerNorm(LayerNormBase):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__(normalized_shape)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x, eps=1e-6):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + eps) * self.weight


class WithBias_LayerNorm(BiasFree_LayerNorm):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__(normalized_shape)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x, eps=1e-6):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + eps) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, norm_type):
        super(LayerNorm, self).__init__()
        if norm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif norm_type == 'ParameterFree':
            self.body = ParameterFree_LayerNorm(dim)
        elif norm_type == 'WithBias':
            self.body = WithBias_LayerNorm(dim)
        else:
            raise ValueError(f'Unsupported norm_type: {norm_type}')

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
#endregion


#region ==[Multi-DConv Head Transposed Self-Attention (MDTA)]==
"""
Modified from Restormer(Restormer/basicsr/models/archs/restormer_arch.py) and LLSKF
"""
class MDTA_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MDTA_Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        temperature = self.temperature * math.log(attn.shape[-1])
        attn = (attn * temperature).softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class MDTA_TransformerBlock(nn.Module):
    def __init__(
            self, in_dim, cue_dim,
            num_heads=2, ffn_expansion_factor=2.66,
            bias=False, norm_type='WithBias',
            zero_init_out=False,
    ):
        super().__init__()

        self.conv_cue = nn.Conv2d(cue_dim, in_dim, 1)

        self.norm_att = LayerNorm(in_dim, norm_type)
        self.attn = MDTA_Attention(in_dim, num_heads, bias)
        self.norm_ffn = LayerNorm(in_dim, norm_type)
        self.ffn = MDTA_FeedForward(in_dim, ffn_expansion_factor, bias)

        if zero_init_out:
            self.gate = nn.Parameter(torch.tensor([1e-6]), requires_grad=True)
        else:
            self.gate = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self, z, cue, *args, **kwargs):
        if z.shape[-2:] != cue.shape[-2:]:
            cue = F.interpolate(cue, size=z.shape[-2:], mode='bilinear', align_corners=False)

            logger = get_root_logger()
            logger.warning(f"The shape of z and cue is different, z: {z.shape}, cue: {cue.shape}")
        cue = self.conv_cue(cue)

        z_identity = z
        z = z_identity + self.attn(
            self.norm_att(z),
            self.norm_att(cue)
        ) * self.gate

        z_identity = z
        z = z_identity + self.ffn(self.norm_ffn(z)) * self.gate

        return z


class MDTA_TransformerBlock_hidden(nn.Module):
    def __init__(
            self, in_dim, cue_dim, hidden_dim=None,
            num_heads=2, ffn_expansion_factor=2.66,
            bias=False, norm_type='WithBias',
            zero_init_out=False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim

        self.conv_in = nn.Conv2d(in_dim, hidden_dim, 1)
        self.conv_cue = nn.Conv2d(cue_dim, hidden_dim, 1)

        self.norm_att = LayerNorm(hidden_dim, norm_type)
        self.attn = MDTA_Attention(hidden_dim, num_heads, bias)
        self.norm_ffn = LayerNorm(hidden_dim, norm_type)
        self.ffn = MDTA_FeedForward(hidden_dim, ffn_expansion_factor, bias)

        self.conv_out = nn.Conv2d(hidden_dim, in_dim, 1)

        if zero_init_out:
            self.gate = nn.Parameter(torch.tensor([1e-6]), requires_grad=True)
        else:
            self.gate = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self, z, cue):
        if z.shape[-2:] != cue.shape[-2:]:
            cue = F.interpolate(cue, size=z.shape[-2:], mode='bilinear', align_corners=False)

            logger = get_root_logger()
            logger.warning(f"The shape of z and cue is different, z: {z.shape}, cue: {cue.shape}")
        cue = self.conv_cue(cue)
        z = self.conv_in(z)

        z_identity = z
        z = z_identity + self.attn(
            self.norm_att(z),
            self.norm_att(cue)
        ) * self.gate

        z_identity = z
        z = z_identity + self.ffn(self.norm_ffn(z)) * self.gate

        z = self.conv_out(z)
        return z
#endregion


#region ==[Illumination Guided Multi-Head Self-Attention (IGMA)]==
class IGMA_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4., bias=False):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.activation = nn.GELU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.activation(x)
        x = self.project_out(x)
        return x


class IGMA_Attention(nn.Module):
    def __init__(self, dim, out_dim=None, bias=False):
        super().__init__()
        out_dim = out_dim or dim

        self.conv1 = nn.Conv2d(dim, dim, 1, bias=bias)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=bias)
        self.conv2 = nn.Conv2d(dim, out_dim, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        return x


class IGMA_TransformerBlock(nn.Module):
    def __init__(
            self, in_dim, cue_dim,
            cue_kernel_size=1, cue_activation="silu", cue_bias=False,
            num_heads=2, ffn_expansion_factor=4.,
            bias=False,
            zero_init_out=False,
    ):
        from basic.archs.util import Activation

        super().__init__()

        self.adaLN_modulation = nn.Sequential(
            Activation.get_module(cue_activation),
            nn.Conv2d(cue_dim, 4 * in_dim, kernel_size=cue_kernel_size, padding=cue_kernel_size // 2, bias=cue_bias)
        )

        norm_type = 'ParameterFree'
        self.pos_embed = nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=True, groups=in_dim)
        self.norm_att = LayerNorm(in_dim, norm_type)
        self.attn = IGMA_Attention(in_dim, in_dim, bias)
        self.norm_ffn = LayerNorm(in_dim, norm_type)
        self.ffn = IGMA_FeedForward(in_dim, ffn_expansion_factor, bias)


        if zero_init_out:
            self.gate = nn.Parameter(torch.tensor([1e-6]), requires_grad=True)
        else:
            self.gate = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, z, cue, *args, **kwargs):
        shift_attn, scale_attn, shift_ffn, scale_ffn = self.adaLN_modulation(cue).chunk(4, dim=1)

        z = z + self.pos_embed(z) * self.gate

        z_norm = self._modulate(self.norm_att(z), scale_attn, shift_attn)
        z_attn = self.attn(z_norm)
        z = z + z_attn * self.gate

        z_norm = self._modulate(self.norm_ffn(z), scale_ffn, shift_ffn)
        z_ffn = self.ffn(z_norm)
        z = z + z_ffn * self.gate

        if self.training:
            SharedPool.get('scalars').append('gate', self.gate.detach().cpu().item())
        return z
#endregion
#endregion


#region [Modulation]
def layer_norm(x, dim_at=1, eps=1e-6):
    mu = x.mean(dim_at, keepdim=True)
    sigma = x.var(dim_at, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(sigma + eps)


class SpatialModulation(nn.Module):
    def __init__(
            self,
            in_dim, cond_dim,
            cond_kernel_size=1, cond_activation="silu", cond_bias=False,
            zero_init=False,
    ):
        from basic.archs.util import Activation

        super().__init__()

        if cond_activation is None:
            self.adaLN_modulation = nn.Sequential(
                nn.Conv2d(cond_dim, 2 * in_dim, kernel_size=cond_kernel_size, padding=cond_kernel_size // 2, bias=cond_bias)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                Activation.get_module(cond_activation),
                nn.Conv2d(cond_dim, 2 * in_dim, kernel_size=cond_kernel_size, padding=cond_kernel_size // 2, bias=cond_bias)
            )

        # self.norm = layer_norm
        self.norm = nn.GroupNorm(num_groups=4, num_channels=in_dim, affine=False)

        self.reset_parameters(zero_init)

    def reset_parameters(self, zero_init=False):
        if zero_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * scale + shift

    def forward(self, z, cue, *args, **kwargs):
        z_identity = z

        z = self.norm(z)
        shift, scale = self.adaLN_modulation(cue).chunk(2, dim=1)
        z = self._modulate(z, shift, scale)
        return z_identity + z
#endregion


#region [JAFAR]
class JAFARSiding(nn.Module):
    def __init__(
            self,
            in_dim, cond_dim,
            zero_init=False,
    ):
        from basic.archs.gadget.jafar import JAFAR

        super().__init__()

        self.fusion_module = JAFAR(
            input_dim=cond_dim, v_dim=in_dim, qk_dim=in_dim
        )

        if zero_init:
            self.gate = nn.Parameter(torch.tensor([1e-6]), requires_grad=True)
        else:
            self.gate = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self, z, cue, *args, **kwargs):
        _z = self.fusion_module(cue, z)
        mask = (cue != 0).float()

        z = z + _z * mask
        return z

#endregion


#region ==[IGFE]==
from typing import Tuple

class IlluminationPrior(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)

        self.num_heads = num_heads
        self.alpha = 200

    def generate_prior_decay(self, prior_grid):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        B, _, H, W = prior_grid.shape
        grid_d = prior_grid.reshape(B, H * W, 1)

        _x = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        def symmetric_approx_cross_entropy_zero(_x):
            _x = torch.abs(_x)
            return - torch.log(1 - _x + 1e-6) * _x
        mask_p = (symmetric_approx_cross_entropy_zero(_x) * self.alpha).sum(dim=-1)    # (1, H, W)
        mask_p = mask_p.unsqueeze(1) * self.decay[None, :, None, None]  # head 维度对应，所以在 dim=1 处 unsqueeze
        return mask_p

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w], indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)

        mask = grid[:, None, :] - grid[None, :, :]
        _x = mask[..., 0]
        _y = mask[..., 1]
        mask = torch.sqrt(_x ** 2 + _y ** 2)    # (H, W)
        mask = (mask * self.decay[:, None, None]).unsqueeze(0)
        return mask

    def forward(self, *HW_tuple: Tuple[int], prior_map, return_all_mask=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        prior_map = F.interpolate(prior_map, size=HW_tuple, mode="bilinear", align_corners=False)

        mask = self.generate_pos_decay(*HW_tuple)
        mask_p = self.generate_prior_decay(prior_map)
        mask_prior = self.weight[0] * mask + self.weight[1] * mask_p

        if self.training:
            SharedPool.get('scalars').append('pos_mask_weight', self.weight[0].mean().detach().cpu().item())
            SharedPool.get('scalars').append('pho_mask_weight', self.weight[1].mean().detach().cpu().item())

        if not is_summary() and (self.weight[0] < 0 or self.weight[1] < 0):
            from basic.utils.console.log import get_root_logger
            logger = get_root_logger()
            logger.warning(f"Detected negative weight, got weight_position={self.weight[0].item()} and weight_prior={self.weight[1].item()} respectively.")

        if not is_summary() and (self.training and torch.any(torch.isnan(mask_prior))):
            from basic.utils.console.log import get_root_logger
            logger = get_root_logger()
            logger.warning(f"Detected NaN in prior mask, count {torch.isnan(mask_prior).sum()}, in mask {torch.isnan(mask).sum()} and mask_p {torch.isnan(mask_p).sum()} respectively.")

        if return_all_mask:
            return mask_prior, mask, mask_p
        return mask_prior

class RoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer("angle", angle)

    def forward(self, *HW_tuple):
        """
        HW_tuple: (H, W)
        返回 (sin, cos)，形状为 (H, W, dim_per_head)
        """
        H, W = HW_tuple
        index = torch.arange(H * W, device=self.angle.device)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])
        sin = sin.reshape(H, W, -1)
        cos = cos.reshape(H, W, -1)
        return sin, cos

"""
Modified from Restormer(Restormer/basicsr/models/archs/restormer_arch.py) and LLSKF
"""
class IGFE_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=8/3, bias=True):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor / 2) * 2

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features // 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class IGFE_SpatialAttention(nn.Module):
    def __init__(
            self, embed_dim, num_heads=4, bias=True, pool_factor=2,
            value_expansion_factor=1, out_dim=None,
    ):
        """
        x as query
        y as key & value
        """
        super().__init__()
        out_dim = out_dim or embed_dim

        self.factor = value_expansion_factor
        self.pool_factor = pool_factor
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_inited = False

        self.q_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=embed_dim)
        )
        self.kv_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1, bias=bias),
            nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=bias)
        )

        # 参考 LRFormer 中的设计
        self.pool = nn.AvgPool2d(kernel_size=pool_factor, stride=pool_factor, padding=0)

        # 参考 DFormer 中的设计
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=1, padding=2, groups=embed_dim, bias=bias)
        # self.out_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim * self.factor, embed_dim, kernel_size=1, bias=bias),
        #     nn.Conv2d(embed_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=bias)
        # )
        self.out_proj = nn.Conv2d(embed_dim * self.factor, out_dim, kernel_size=1, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj[0].weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.kv_proj[0].weight, gain=2**-2.5)

        if isinstance(self.out_proj, nn.Conv2d):
            nn.init.orthogonal_(self.out_proj.weight.data)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias.data)
        else:
            for layer in self.out_proj.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight.data)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias.data)

    def forward(self, x, y, rel_pos, return_out_map=False, rope_embedding=None):
        """
        Args:
            x: (B, C, H, W), query
            y: (B, C, H, W), key and value
            rel_pos (B, head, HW, HW):
        """
        b, c, h0, w0 = x.shape
        if self.training and not self.temperature_inited:
            self.temperature.data = (torch.ones_like(self.temperature) / math.log(h0 * w0)).detach()
            self.temperature_inited = True

            # from basic.utils.log import get_root_logger
            # logger = get_root_logger()
            # logger.info(f"temperature inited to: {self.temperature.mean().detach().cpu().item()}")

        mask = rel_pos  # mask range (-inf, 0), (B, head, HW, HW)

        lepe = self.lepe(x)

        if self.pool_factor > 1:
            x = self.pool(x)                    # (B, C, H, W) -> (B, C, H/2, W/2)
            y = self.pool(y)                    # (B, C, H, W) -> (B, C, H/2, W/2)

        h, w = x.shape[-2:]
        q = self.q_proj(x)
        k, v = self.kv_proj(y).chunk(2, dim=1)

        if is_debug() and not self.training:
            from basic.utils.console.logplot import get_root_plotter
            plotter = get_root_plotter(plot_sub_dir="phoattn")
            plotter.heatmap(y, fig_name="attn/y+")
            plotter.heatmap(q, fig_name="attn/q+")
            plotter.heatmap(k, fig_name="attn/k+")
            plotter.heatmap(v, fig_name="attn/v+")
            plotter.semantic_feature_map_joint(
                q, k, v,
                fig_name="attn/attn",
                fig_alias=["q+", "k+", "v+"]
            )

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        # 如果有旋转位置编码，则应用旋转位置编码
        if rope_embedding is not None:
            sin, cos = rope_embedding

            def angle_transform(x, sin, cos):
                x1 = x[:, :, :, :, ::2]
                x2 = x[:, :, :, :, 1::2]
                return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)

            q = rearrange(q, 'b head (h w) c -> b head h w c', h=h, w=w)
            k = rearrange(k, 'b head (h w) c -> b head h w c', h=h, w=w)

            q = angle_transform(q, sin, cos)
            k = angle_transform(k, sin, cos)

            q = rearrange(q, 'b head h w c -> b head (h w) c')
            k = rearrange(k, 'b head h w c -> b head (h w) c')

        attn = (q @ k.transpose(-2, -1))

        ## Normal VERSION
        # temperature = self.temperature # * math.log(attn.shape[-2]) / (attn.shape[-1] ** 0.5)
        # raw_attn = attn * temperature #
        # attn = raw_attn + mask
        # attn = attn.softmax(dim=-1)                 # (B, head, N, N)

        ## topk VERSION
        temperature = self.temperature # * math.log(attn.shape[-2]) / (attn.shape[-1] ** 0.5)
        raw_attn = attn * temperature #                             # (B, head, N, N)
        attn = raw_attn + mask                                      # (B, head, N, N)

        # top-k selective attention
        sqrt_N = min(max(int(attn.shape[-1] ** 0.5), 1), attn.shape[-1])
        topk_logits, topk_idx = torch.topk(attn.contiguous().float(), k=sqrt_N, dim=-1)
        topk_attn = torch.softmax(topk_logits, dim=-1).to(attn.dtype)  # (B, head, N, k)

        if is_debug() and not self.training:
            from basic.utils.console.log import get_stats
            for _h in range(self.num_heads):
                pass
                # print(f"attn {_h}", get_stats((q @ k.transpose(-2, -1))[:, _h]))
                # print(f"raw_attn {_h}", get_stats(raw_attn[:, _h]))
                # print(f"attn {_h}", get_stats(attn[:, _h]))
                # print(f"temperature {_h}", temperature[_h])
        if is_debug("train"):
            from basic.utils.console.log import get_stats
            if self.training:
                step, = SharedPool.get("train").try_get("current_step")
                if step % 10 == 0:
                    from basic.utils.console.logplot import get_root_plotter

                    plotter = get_root_plotter(plot_sub_dir="phoattn")

                    gt, = SharedPool.get("teacher_forcing").try_get("gt")
                    gt = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)

                    h1, w1 = h // 2, w // 2
                    _h = self.num_heads // 2
                    plotter.attention_map(
                        gt,
                        rearrange(
                            rearrange(raw_attn, 'b head (h w) n -> b head h w n', h=h, w=w)[:, _h, w1, h1].softmax(dim=-1),
                            "b (h w) -> b 1 h w", h=h, w=w
                        ),
                        fig_name=f"attn_{step}/raw_attn+"
                    )
                    plotter.attention_map(
                        gt,
                        rearrange(
                            rearrange(mask, 'b head (h w) n -> b head h w n', h=h, w=w)[:, _h, w1, h1].softmax(dim=-1),
                            "b (h w) -> b 1 h w", h=h, w=w
                        ),
                        fig_name=f"attn_{step}/mask+"
                    )
                    plotter.attention_map(
                        gt,
                        rearrange(
                            rearrange(attn, 'b head (h w) n -> b head h w n', h=h, w=w)[:, _h, w1, h1],
                            "b (h w) -> b 1 h w", h=h, w=w
                        ),
                        fig_name=f"attn_{step}/attn+"
                    )

                    from basic.utils.console.log import get_root_logger

                    test_logger = get_root_logger("metrics")
                    test_logger.info(f"raw_attn {step} {get_stats(raw_attn[:, _h])}")
                    test_logger.info(f"mask {step} {get_stats(mask[:, _h])}")
                    test_logger.info(f"attn {step} {get_stats(attn[:, _h])}")

        if self.training:
            SharedPool.get('scalars').append('spatial_attn_temperature', self.temperature.mean().detach().cpu().item())
            SharedPool.get('scalars').append('spatial_attn_temperature_scaled', temperature.mean().detach().cpu().item())

        ## topk VERSION
        b_idx = torch.arange(b, device=v.device)[:, None, None, None]               # (B,1,1,1)
        h_idx = torch.arange(self.num_heads, device=v.device)[None, :, None, None]  # (1,H,1,1)
        v_topk = v[b_idx, h_idx, topk_idx, :]                                       # (B,H,N,k,C)
        out = (topk_attn.unsqueeze(-1) * v_topk).sum(dim=-2)                        # (B,H,N,C)

        ## Normal VERSION
        # out = (attn @ v)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        if self.pool_factor > 1:
            out = F.interpolate(out, size=(h0, w0), mode='bilinear', align_corners=False)

        out = out + lepe
        out = self.out_proj(out)

        if return_out_map:
            return out, dict(
                raw_attn=raw_attn,
                attn=attn,
                mask=mask
            )
        return out

class IGFE_ChannelAttention(nn.Module):
    def __init__(
            self, embed_dim, num_heads=4, bias=True,
            value_expansion_factor=1, out_dim=None
    ):
        """
        y as query
        x as key & value
        """
        super().__init__()
        out_dim = out_dim or embed_dim

        self.factor = value_expansion_factor
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.log(embed_dim))

        self.q_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=embed_dim)
        )
        self.kv_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1, bias=bias),
            nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=bias)
        )

        # 参考 DFormer 中的设计
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=1, padding=2, groups=embed_dim, bias=bias)
        # self.out_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim * self.factor, embed_dim, kernel_size=1, bias=bias),
        #     nn.Conv2d(embed_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=bias)
        # )
        self.out_proj = nn.Conv2d(embed_dim * self.factor, out_dim, kernel_size=1, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj[0].weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.kv_proj[0].weight, gain=2**-2.5)

        if isinstance(self.out_proj, nn.Conv2d):
            nn.init.orthogonal_(self.out_proj.weight.data)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias.data)
        else:
            for layer in self.out_proj.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight.data)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias.data)

    def forward(self, x, y, return_out_map=False):
        """
        Args:
            x: (B, C, H, W)
            y: (B, C, H, W)
            rel_pos:
        """
        b, c, h0, w0 = x.shape

        lepe = self.lepe(x)

        h, w = x.shape[-2:]
        k, v = self.kv_proj(x).chunk(2, dim=1)
        q = self.q_proj(y)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1))
        attn = attn
        temperature = self.temperature * math.log(attn.shape[-1])
        attn = (attn / temperature).softmax(dim=-1)                 # (B, h, N, N)

        if self.training:
            SharedPool.get('scalars').append('channel_attn_temperature', self.temperature.mean().detach().cpu().item())
            SharedPool.get('scalars').append('channel_attn_temperature_scaled', temperature.mean().detach().cpu().item())

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out + lepe
        out = self.out_proj(out)

        if return_out_map:
            return out, dict(
                attn=attn,
            )
        return out

class IGFE_TransformerBlock(nn.Module):
    def __init__(
            self, in_dim, cue_dim, embed_dim=None,
            num_heads=4, ffn_expansion_factor=8/3,
            bias=True, norm_type='WithBias',
            layer_init_value=1e-6,
            prior_init_value=2,
            heads_range=4,

            pool_factor=2,
            any_siding=True,
    ):
        super().__init__()
        embed_dim = embed_dim or in_dim
        self.pool_factor = pool_factor

        # in/out projection
        self.conv_in = nn.Conv2d(in_dim, embed_dim, 1)
        self.conv_cue = nn.Conv2d(cue_dim, embed_dim, 1)
        self.conv_out = nn.Conv2d(embed_dim, in_dim, 1)

        # embedding
        self.prior = IlluminationPrior(embed_dim, num_heads, initial_value=prior_init_value, heads_range=heads_range)
        self.pos_embed_in = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=True, groups=embed_dim)
        self.pos_embed_cue = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=True, groups=embed_dim)
        self.rope = RoPE(embed_dim, num_heads)

        # transformer block
        self.norm_att_x = LayerNorm(embed_dim, norm_type)
        self.norm_att_y = LayerNorm(embed_dim, norm_type)
        self.attn = IGFE_SpatialAttention(embed_dim, num_heads, bias, pool_factor=pool_factor)
        self.norm_ffn = LayerNorm(embed_dim, norm_type)
        self.ffn = IGFE_FeedForward(embed_dim, ffn_expansion_factor, bias)

        # siding
        self.any_siding = any_siding
        if any_siding:
            self.gamma = nn.Parameter(layer_init_value * torch.ones(1, in_dim, 1, 1), requires_grad=True)

    def forward(self, x, y, prior, return_out_map=False, use_prior=True):
        """
        Args:
            x:      (B, C, H, W)
            y:      (B, C, H, W)
            prior:  (B, 1, H, W)
        """
        x_identity = x

        # [in projection]
        x = self.conv_in(x)
        y = self.conv_cue(y)
        if is_debug() and not self.training:
            from basic.utils.console.logplot import get_root_plotter
            plotter = get_root_plotter(plot_sub_dir="icmem")
            plotter.semantic_feature_map_joint(
                x, y,
                fig_name="feat_map/igfe/igfe",
                fig_alias=["x+", "y+"]
            )

        if is_debug("train"):
            from basic.utils.console.log import get_stats
            if self.training:
                step, = SharedPool.get("train").try_get("current_step")
                if step % 10 == 0:
                    from basic.utils.console.logplot import get_root_plotter
                    plotter = get_root_plotter(plot_sub_dir="phoattn")
                    plotter.image(
                        prior,
                        fig_name=f"attn_{step}/illu"
                    )
                    plotter.heatmap(
                        prior,
                        fig_name=f"attn_{step}/illu_heatmap",
                        show_colorbar=True
                    )

                    from basic.utils.console.log import get_root_logger
                    test_logger = get_root_logger("metrics")
                    test_logger.info(f"illu_val_readout {step} {get_stats(prior)}")

        # [transformer]
        prior_size = (x.shape[-2] // self.pool_factor, x.shape[-1] // self.pool_factor)
        if use_prior:
            illu = self.prior(*prior_size, prior_map=prior.detach())
        else:
            hw = prior_size[0] * prior_size[1]
            illu = torch.zeros(1, 1, hw, hw, device=x.device)

        x = x + self.pos_embed_in(x)
        y = y + self.pos_embed_cue(y)
        rope_embedding = self.rope(prior_size[0], prior_size[1])

        x_attn = self.attn(
            self.norm_att_x(x),
            self.norm_att_y(y),
            rel_pos=illu,
            return_out_map=return_out_map,
            rope_embedding=rope_embedding
        )
        if return_out_map:
            x_attn, out_map = x_attn
        x = x + x_attn

        x = x + self.ffn(
            self.norm_ffn(x)
        )

        # [out projection]
        x = self.conv_out(x)

        # [siding]
        if self.any_siding:
            x = x_identity + x * self.gamma
            if self.training:
                SharedPool.get('scalars').append('gamma', self.gamma.mean().detach().cpu().item())

        if return_out_map:
            return x, out_map
        return x

class IGFE_TransformerXBlock(nn.Module):
    def __init__(
            self, in_dim, cue_dim, embed_dim=None,
            num_heads=4, ffn_expansion_factor=8/3,
            bias=True, norm_type='WithBias',
            layer_init_value=1e-6,
            prior_init_value=2,
            heads_range=4,

            pool_factor=2,
            any_siding=True,
            attention_type='s',
    ):
        super().__init__()
        embed_dim = embed_dim or in_dim
        self.pool_factor = pool_factor

        # in/out projection
        self.conv_in = nn.Conv2d(in_dim, embed_dim, 1)
        self.conv_cue = nn.Conv2d(cue_dim, embed_dim, 1)
        self.conv_out = nn.Conv2d(embed_dim, in_dim, 1)

        # embedding
        self.prior = IlluminationPrior(embed_dim, num_heads, initial_value=prior_init_value, heads_range=heads_range)
        self.pos_embed_in = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=True, groups=embed_dim)
        self.pos_embed_cue = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=True, groups=embed_dim)

        # transformer block
        self.attn_modules = nn.ModuleList(
            self.build_attention(attention_type, embed_dim, num_heads, bias, pool_factor, norm_type)
        )
        self.norm_ffn = LayerNorm(embed_dim, norm_type)
        self.ffn = IGFE_FeedForward(embed_dim, ffn_expansion_factor, bias)

        # siding
        self.any_siding = any_siding
        if any_siding:
            self.gamma = nn.Parameter(layer_init_value * torch.ones(1, in_dim, 1, 1), requires_grad=True)

    def build_attention(self, attention_type, embed_dim, num_heads, bias, pool_factor, norm_type):
        attn_modules = []
        for t in attention_type:
            if t == 's':
                norm_att_x = LayerNorm(embed_dim, norm_type)
                norm_att_y = LayerNorm(embed_dim, norm_type)
                attn = IGFE_SpatialAttention(embed_dim, num_heads, bias, pool_factor=pool_factor)

                attn_modules.extend([
                    norm_att_x,
                    norm_att_y,
                    attn,
                ])
            elif t == 'c':
                norm_att_x = LayerNorm(embed_dim, norm_type)
                norm_att_y = LayerNorm(embed_dim, norm_type)
                attn = IGFE_ChannelAttention(embed_dim, num_heads, bias)

                attn_modules.extend([
                    norm_att_x,
                    norm_att_y,
                    attn,
                ])
            else:
                raise ValueError(f"Unknown attention type: {t}")

        return attn_modules

    def forward(self, x, y, prior, return_out_map=False):
        """
        Args:
            x:      (B, C, H, W)
            y:      (B, C, H, W)
            prior:  (B, 1, H, W)
        """
        x_identity = x

        # [in projection]
        x = self.conv_in(x)
        y = self.conv_cue(y)
        if is_debug() and not self.training:
            from basic.utils.console.logplot import get_root_plotter
            plotter = get_root_plotter(plot_sub_dir="icmem")
            plotter.semantic_feature_map_joint(
                x, y, y - x,
                fig_name="feat_map/igfe/igfe",
                fig_alias=["x+", "y+", "y-x+"]
            )

            from basic.utils.console.log import get_stats
            print(get_stats(x))
            print(get_stats(y))

        # [transformer]
        prior_size = (x.shape[-2] // self.pool_factor, x.shape[-1] // self.pool_factor)
        illu = self.prior(prior_size, prior_map=prior.detach())

        x = x + self.pos_embed_in(x)
        y = y + self.pos_embed_cue(y)

        if return_out_map:
            out_maps = []
        for i in range(len(self.attn_modules) // 3):
            norm_att_x, norm_att_y, attn = self.attn_modules[3*i:3*i+3]
            x_attn = attn(
                norm_att_x(x),
                norm_att_y(y), # - norm_att_x(x),
                rel_pos=illu,
                return_out_map=return_out_map
            )
            if return_out_map:
                x_attn, out_map = x_attn
                out_maps.append(out_map)
            x = x + x_attn

        x = x + self.ffn(
            self.norm_ffn(x)
        )

        # [out projection]
        x = self.conv_out(x)

        # [siding]
        if self.any_siding:
            x = x_identity + x * self.gamma
            if self.training:
                SharedPool.get('scalars').append('gamma', self.gamma.mean().detach().cpu().item())

        if return_out_map:
            return x, out_maps
        return x

class IGFE_Transformer(nn.Module):
    def __init__(
            self, in_dim, cue_dim, embed_dim=None,
            num_heads=4, ffn_expansion_factor=8/3,
            bias=True, norm_type='WithBias',
            layer_init_value=1e-6,
            prior_init_value=2,
            prior_use_layer=None,
            heads_range=4,

            pool_factor=2,
            any_siding=True,
            num_layers=2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            IGFE_TransformerBlock(
                in_dim, cue_dim, embed_dim,
                num_heads, ffn_expansion_factor,
                bias, norm_type,
                layer_init_value,
                prior_init_value,
                heads_range,
                pool_factor,
                any_siding=False,
            )
            for _ in range(num_layers)
        ])

        self.prior_use_layer = prior_use_layer

        # siding
        self.any_siding = any_siding
        if any_siding:
            self.gamma = nn.Parameter(layer_init_value * torch.ones(1, in_dim, 1, 1), requires_grad=True)

    def forward(self, x, y, prior, return_out_map=False):
        """
        Args:
            x:      (B, C, H, W)
            y:      (B, C, H, W)
            prior:  (B, 1, H, W)
        """
        x_identity = x

        out_maps = []
        for i, layer in enumerate(self.layers):
            x = layer(
                x, y, prior,
                return_out_map=return_out_map,
                use_prior=self.prior_use_layer is None or i in self.prior_use_layer
            )
            if return_out_map:
                x, out_map = x
                out_maps.append(out_map)

        if self.any_siding:
            x = x_identity + x * self.gamma
            if self.training:
                SharedPool.get('scalars').append('gamma', self.gamma.mean().detach().cpu().item())

        if return_out_map:
            return x, out_maps
        return x

class IGFE_TransformerX(nn.Module):
    def __init__(
            self, in_dim, cue_dim, embed_dim=None,
            num_heads=4, ffn_expansion_factor=8/3,
            bias=True, norm_type='WithBias',
            layer_init_value=1e-6,
            prior_init_value=2,
            heads_range=4,

            pool_factor=2,
            any_siding=True,
            num_layers=2,
            attention_type='s',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            IGFE_TransformerXBlock(
                in_dim, cue_dim, embed_dim,
                num_heads, ffn_expansion_factor,
                bias, norm_type,
                layer_init_value,
                prior_init_value,
                heads_range,
                pool_factor,
                attention_type=attention_type,
                any_siding=False,
            )
            for _ in range(num_layers)
        ])

        # siding
        self.any_siding = any_siding
        if any_siding:
            self.gamma = nn.Parameter(layer_init_value * torch.ones(1, in_dim, 1, 1), requires_grad=True)

    def forward(self, x, y, prior, return_out_map=False):
        """
        Args:
            x:      (B, C, H, W)
            y:      (B, C, H, W)
            prior:  (B, 1, H, W)
        """
        x_identity = x

        out_maps = []
        for layer in self.layers:
            x = layer(x, y, prior, return_out_map=return_out_map)
            if return_out_map:
                x, out_map = x
                out_maps.append(out_map)

        if self.any_siding:
            x = x_identity + x * self.gamma
            if self.training:
                SharedPool.get('scalars').append('gamma', self.gamma.mean().detach().cpu().item())

        if return_out_map:
            return x, out_maps
        return x
#endregion


def smooth_clamp01(x, k=10):
    e_kx = torch.exp(k * x)
    e_kx_minus_one = e_kx / math.exp(k)
    return torch.log((1 + e_kx) / (1 + e_kx / e_kx_minus_one)) / k


# if __name__ == '__main__':
#     from basic.metrics.summary import get_params
#     net = resnet50(pretrained=True)
#     print(f"resnet50: {get_params(net) / 1e6:.3f} M")
#     net = resnet18(pretrained=True, extra_dim=3)
#     print(f"resnet18: {get_params(net) / 1e6:.3f} M")
#
#     net = ValueEncoder()
#     print(f"ValueEncoder: {get_params(net) / 1e6:.3f} M")
#     x = torch.randn(1, 3, 256, 256)
#     z = torch.randn(1, 128, 16, 16)
#     y = net(x, x, z)
#
#     net = KeyEncoder()
#     print(f"KeyEncoder: {get_params(net) / 1e6:.3f} M")
#     x = torch.randn(1, 3, 256, 256)
#     y = net(x)