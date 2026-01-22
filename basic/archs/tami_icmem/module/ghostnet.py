import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class GhostConvBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostConvBlock, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.cheap_operation = nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# kernel_size=1, dw_size=3: 128, 128 -> #params: 9,024
# kernel_size=1, dw_size=5: 128, 128 -> #params: 10,048
# kernel_size=3, dw_size=3: 128, 128 -> #params: 74,560
class GhostModule(nn.Module):
    def __init__(
            self, inp, oup,
            kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,
            with_norm=False
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        if with_norm:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        else:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# dw_kernel_size=3: 128, 128, 128     -> #params: 45,856
# dw_kernel_size=3: 128+128, 128, 128 -> #params: 71,840
# dw_kernel_size=5: 128, 128, 128     -> #params: 49,952
# dw_kernel_size=5: 128+128, 128, 128 -> #params: 75,528
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, out_chs, mid_chs=None, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., with_norm=False):
        super(GhostBottleneck, self).__init__()
        mid_chs = mid_chs or out_chs
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            if with_norm:
                self.bn_dw = nn.BatchNorm2d(mid_chs)
            else:
                self.bn_dw = nn.Identity()

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            if with_norm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                              padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                    nn.BatchNorm2d(in_chs),
                    nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_chs),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                              padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                    nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNetFeatureFusionBlock(nn.Module):
    def __init__(
            self,
            x_in_dim, y_in_dim, y_mid_dim, y_out_dim,
            dw_kernel_size=5, bias=True
    ):
        super().__init__()

        self.block = GhostBottleneck(
            x_in_dim + y_in_dim, y_mid_dim, y_out_dim,
            dw_kernel_size=dw_kernel_size, stride=1, se_ratio=0.25
        )

    def forward(self, x, y):
        y = torch.cat([x, y], dim=1)
        y = self.block(y)
        return y


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    block = GhostNetFeatureFusionBlock(128, 128, 128, 128)
    print(count_parameters(block))

    block = GhostBottleneck(128, 128, 128, dw_kernel_size=5, stride=2, se_ratio=0.25)
    print(count_parameters(block))

    block = GhostModule(128, 128, kernel_size=1, dw_size=3)
    print(count_parameters(block))