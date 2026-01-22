import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import util as arch_util
from .transformer.Models import Encoder_patch66

from basic.utils.console.log import get_root_logger
logger = get_root_logger()


from basic.utils.registry import ARCH_REGISTRY

'''
Adapted from SNRAware(https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance/blob/main/models/archs/low_light_transformer.py)
'''

@ARCH_REGISTRY.register()
class SNRAwareNet(nn.Module):
    def __init__(
            self,
            nf=64, front_RBs=5, back_RBs=10,
            predeblur=False, HR_in=False, w_TSA=True
    ):
        super(SNRAwareNet, self).__init__()
        self.nf = nf
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

    def forward(self, x, mask=None):
        x_center = x
        if mask is None:
            mask = torch.ones_like(x_center[:, 0:1, :, :])
            logger.warning('Mask is not provided, using all ones as mask.')

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)

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
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center

        return out_noise

    # def forward(self, x, mask=None):
    #     from basic.utils.log import is_debug
    #     from basic.utils.logplot import get_root_plotter
    #     plotter = get_root_plotter(plot_sub_dir="snraware")
    #
    #     x_center = x
    #     if mask is None:
    #         mask = torch.ones_like(x_center[:, 0:1, :, :])
    #         logger.warning('Mask is not provided, using all ones as mask.')
    #
    #     L1_fea_1 = self.lrelu(self.conv_first_1(x_center))  # (B, 64, H, W)
    #     L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))  # (B, 64, H, W) -> (B, 64, H/2, W/2)
    #     L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))  # (B, 64, H/2, W/2) -> (B, 64, H/4, W/4)
    #
    #     fea = self.feature_extraction(L1_fea_3)             # (B, 64, H/4, W/4)
    #     if is_debug() and not self.training:
    #         plotter.semantic_feature_map(fea, fig_name="fea/smtc+")
    #         plotter.heatmap(fea, fig_name="fea/smtc_hm+")
    #     fea_light = self.recon_trunk_light(fea)             # (B, 64, H/4, W/4)
    #     if is_debug() and not self.training:
    #         plotter.semantic_feature_map(fea_light, fig_name="fea_light/smtc+")
    #         plotter.heatmap(fea_light, fig_name="fea_light/smtc_hm+")
    #
    #     h_feature = fea.shape[2]
    #     w_feature = fea.shape[3]
    #     mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')
    #
    #     xs = np.linspace(-1, 1, fea.size(3) // 4)
    #     ys = np.linspace(-1, 1, fea.size(2) // 4)
    #     xs = np.meshgrid(xs, ys)
    #     xs = np.stack(xs, 2)
    #     xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
    #     xs = xs.view(fea.size(0), -1, 2)
    #
    #     height = fea.shape[2]
    #     width = fea.shape[3]
    #     fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
    #     fea_unfold = fea_unfold.permute(0, 2, 1)
    #
    #     mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
    #     mask_unfold = mask_unfold.permute(0, 2, 1)
    #     mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
    #     mask_unfold[mask_unfold <= 0.5] = 0.0
    #
    #     fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
    #     fea_unfold = fea_unfold.permute(0, 2, 1)
    #     fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)
    #
    #     channel = fea.shape[1]
    #     mask = mask.repeat(1, channel, 1, 1)
    #     fea = fea_unfold * (1 - mask) + fea_light * mask
    #
    #     out_noise = self.recon_trunk(fea)
    #     out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
    #     if is_debug() and not self.training:
    #         plotter.semantic_feature_map(fea_light, fig_name="out_noise/smtc+")
    #         plotter.heatmap(fea_light, fig_name="out_noise/smtc_hm+")
    #     out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise))) # (B, 64, H/4, W/4) -> (B, 64, H/2, W/2)
    #     out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
    #     out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise))) # (B, 64, H/2, W/2) -> (B, 64, H, W)
    #     out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
    #     out_noise = self.lrelu(self.HRconv(out_noise))                      # (B, 64, H, W)
    #     out_noise = self.conv_last(out_noise)
    #     out_noise = out_noise + x_center
    #
    #     return out_noise
