
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from basic.metrics.summary import is_summary

from basic.utils.console.log import get_root_logger
logger = get_root_logger()


from basic.archs.tami_icmem.me_arch import ICMemoryEnhancedNet
from .snr_aware_arch import SNRAwareNet

from basic.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ICMemSNRAwareNet(ICMemoryEnhancedNet, SNRAwareNet):
    def __init__(
            self,
            nf=64, front_RBs=5, back_RBs=10,
            predeblur=False, HR_in=False, w_TSA=True,

            memory_decoder=None,
            memory_trainable_only=False,
            any_multi_layer=False,
            **kwargs
    ):
        SNRAwareNet.__init__(
            self,
            nf, front_RBs, back_RBs,
            predeblur, HR_in, w_TSA
        )

        decode = self.decode_multilayer if any_multi_layer else self.decode
        ICMemoryEnhancedNet.__init__(
            self,
            encoder=self.encode, decoder=decode,
            i_memory_decoder=memory_decoder,
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
            **kwargs
        )

    def encode(self, x, mask=None):
        x_center = x
        if mask is None:
            mask = torch.ones_like(x_center[:, 0:1, :, :])

            if not is_summary():
                logger.warning('Mask is not provided, using all ones as mask.')

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))  # (B, 64, H, W)
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))  # (B, 64, H, W) -> (B, 64, H/2, W/2)
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))  # (B, 64, H/2, W/2) -> (B, 64, H/4, W/4)

        fea = self.feature_extraction(L1_fea_3)             # (B, 64, H/4, W/4)
        return x_center, mask, (L1_fea_1, L1_fea_2, L1_fea_3), fea

    def decode(self, x_center, mask, skip_feas, fea):
        L1_fea_1, L1_fea_2, L1_fea_3 = skip_feas

        fea = yield fea

        fea_light = self.recon_trunk_light(fea)             # (B, 64, H/4, W/4)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0

        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise))) # (B, 64, H/4, W/4) -> (B, 64, H/2, W/2)
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise))) # (B, 64, H/2, W/2) -> (B, 64, H, W)
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))                      # (B, 64, H, W)
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center

        yield out_noise
