"""
resnet.py - A modified ResNet structure
We append extra channels to the first conv by some network surgery
"""

from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo


#region ==[Modules]==
from .cbam import CBAM

# 128, 128 -> #params: 17,792
class DSConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_dim, out_dim, kernel_size, stride, padding, groups=out_dim),
        )

    def forward(self, x):
        return self.net(x)


# 128, 128 -> #params: 34,304
class MobileBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.branch = None
        else:
            self.branch = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim),
            nn.ReLU6(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, y):
        out_y = self.net(y)

        if self.branch is not None:
            y = self.branch(y)

        return out_y + y


# 128, 128 -> #params: 295,168
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.branch = None
        else:
            self.branch = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        )

    def forward(self, y):
        out_y = self.net(y)

        if self.branch is not None:
            y = self.branch(y)

        return out_y + y


# LightWeightResBlock: 128, 128, 128, 128 -> #params 120,171
# ResBlock:            128, 128, 128, 128 -> #params 772,971
class MobileFeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, y_in_dim, y_mid_dim, y_out_dim, bias=True):
        super().__init__()

        self.block1 = MobileBlock(x_in_dim + y_in_dim, y_mid_dim)
        self.attention = CBAM(y_mid_dim)
        self.block2 = MobileBlock(y_mid_dim, y_out_dim)
        # self.block1 = nn.Conv2d(x_in_dim + y_in_dim, y_mid_dim, kernel_size=1, bias=bias)
        # self.attention = CBAM(y_mid_dim)
        # self.block2 = nn.Conv2d(y_mid_dim, y_out_dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        y = torch.cat([x, y], dim=1)
        y = self.block1(y)
        r = self.attention(y)
        y = self.block2(y + r)
        return y


class ResFeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, y_in_dim, y_mid_dim, y_out_dim, bias=True):
        super().__init__()

        self.block1 = ResBlock(x_in_dim + y_in_dim, y_mid_dim)
        self.attention = CBAM(y_mid_dim)
        self.block2 = ResBlock(y_mid_dim, y_out_dim)
        # self.block1 = nn.Conv2d(x_in_dim + y_in_dim, y_mid_dim, kernel_size=1, bias=bias)
        # self.attention = CBAM(y_mid_dim)
        # self.block2 = nn.Conv2d(y_mid_dim, y_out_dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        y = torch.cat([x, y], dim=1)
        y = self.block1(y)
        r = self.attention(y)
        y = self.block2(y + r)
        return y
#endregion


#region ==[Pretrained Models]==
def load_weights_add_extra_dim(target, source_state, extra_dim=1):
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    # print(v1.shape, tar_v.shape)
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c,extra_dim,w,h), device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        dilation=dilation, bias=False,
        padding=dilation, padding_mode="reflect"
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_dim=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3+extra_dim, 64,
            kernel_size=7, stride=2, bias=False,
            padding=3, padding_mode="reflect")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

def build_resnet18(pretrained=True, extra_dim=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_dim)
    if pretrained:
        load_weights_add_extra_dim(model, model_zoo.load_url(model_urls['resnet18']), extra_dim)
    return model

def build_resnet50(pretrained=True, extra_dim=0):
    model = ResNet(ResBottleneck, [3, 4, 6, 3], extra_dim)
    if pretrained:
        load_weights_add_extra_dim(model, model_zoo.load_url(model_urls['resnet50']), extra_dim)
    return model
#endregion


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    block = MobileBlock(128, 128)
    print(count_parameters(block))

    block = DSConv(128, 128)
    print(count_parameters(block))