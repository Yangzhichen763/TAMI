from collections import OrderedDict

import torch
import torch.nn as nn

from torch.utils import model_zoo
from torchvision.models.mobilenetv2 import MobileNetV2, mobilenet_v2
from torchvision.ops import Conv2dNormActivation

model_urls = {
    'mobilenet_v2': "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
}


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


class CustomMobileNetV2(MobileNetV2):
    def __init__(self, extra_dim=0):
        super().__init__()

        norm_layer = nn.BatchNorm2d
        input_channel = 32
        self.features[0] = Conv2dNormActivation(3 + extra_dim, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)


def build_mobilenet_v2(pretrained=True, extra_dim=0, **kwargs):
    model = CustomMobileNetV2(extra_dim=extra_dim)
    if pretrained:
        load_weights_add_extra_dim(model, model_zoo.load_url(model_urls['mobilenet_v2']), extra_dim=extra_dim)
    return model
