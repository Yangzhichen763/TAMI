import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .hvi_cid_util import HV_LCA, I_LCA, RGB2HVI, NormDownsample, NormUpsample
from basic.archs.util import clone_module
from basic.archs.memory.memory_enhanced_arch import MemoryEnhancedNet

from basic.utils.registry import ARCH_REGISTRY

'''
Modified from HVI-CIDNet(https://github.com/Fediory/HVI-CIDNet/blob/master/net/CIDNet.py)
'''


@ARCH_REGISTRY.register()
class CIDNet(nn.Module):
    def __init__(self,
                 channels=(36, 36, 72, 144),
                 heads=(1, 2, 4, 8),
                 norm=False,
                 transform=None,
                 ):
        super(CIDNet, self).__init__()

        ch1, ch2, ch3, ch4 = channels
        head1, head2, head3, head4 = heads

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        if transform is None:
            self.trans = RGB2HVI()
        else:
            self.trans = transform

    def _in_transform(self, x):
        dtypes = x.dtype

        hvi = self.trans.HVIT(x)                    # (B, 3, H, W)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes) # (B, 1, H, W)
        return i, hvi

    def _encode(self, i, hvi):
        # [encoding]
        i_enc0 = self.IE_block0(i)                  # (B, 1, H, W) -> (B, ch1, H, W)
        hv_0 = self.HVE_block0(hvi)                 # (B, 3, H, W) -> (B, ch1, H, W)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc1 = self.IE_block1(i_enc0)             # (B, ch1, H, W) -> (B, ch2, H/2, W/2)
        hv_1 = self.HVE_block1(hv_0)                # (B, ch1, H, W) -> (B, ch2, H/2, W/2)
        i_enc2 = self.I_LCA1(i_enc1, hv_1)          # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)           # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        i_jump1 = i_enc2
        hv_jump1 = hv_2

        i_enc2 = self.IE_block2(i_enc2)             # (B, ch2, H/2, W/2) -> (B, ch3, H/4, W/4)
        hv_2 = self.HVE_block2(hv_2)                # (B, ch2, H/2, W/2) -> (B, ch3, H/4, W/4)
        i_enc3 = self.I_LCA2(i_enc2, hv_2)          # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)           # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        i_jump2 = i_enc3
        hv_jump2 = hv_3

        i_enc3 = self.IE_block3(i_enc2)             # (B, ch3, H/4, W/4) -> (B, ch4, H/8, W/8)
        hv_3 = self.HVE_block3(hv_2)                # (B, ch3, H/4, W/4) -> (B, ch4, H/8, W/8)
        i_enc3 = self.I_LCA3(i_enc3, hv_3)          # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)
        hv_3 = self.HV_LCA3(hv_3, i_enc3)           # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)

        return (i_enc0, i_enc1, i_enc2, i_enc3), (hv_0, hv_1, hv_2, hv_3), (i_jump0, i_jump1, i_jump2), (hv_jump0, hv_jump1, hv_jump2)

    def _decode(self, i_feats, hv_feats, i_jumps, hv_jumps):
        i_enc0, i_enc1, i_enc2, i_enc3 = i_feats
        hv_0, hv_1, hv_2, hv_3 = hv_feats
        i_jump0, i_jump1, i_jump2 = i_jumps
        hv_jump0, hv_jump1, hv_jump2 = hv_jumps

        # [decoding]
        # if not self.training:
        #     from basic.utils.logplot import get_root_plotter
        #     plotter = get_root_plotter(plot_sub_dir="cidnet/3d")
        #     plotter.heatmap(hv_3, fig_name="hv3d+")
        #     plotter.heatmap(i_enc3, fig_name="i3d+")
        hv_3 = self.HV_LCA4(hv_3, i_enc3)           # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)
        i_dec3 = self.I_LCA4(i_enc3, hv_3)          # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)
        hv_3 = self.HVD_block3(hv_3, hv_jump2)      # (B, ch4+ch3, H/8, W/8) -> (B, ch3, H/4, W/4)
        i_dec3 = self.ID_block3(i_dec3, i_jump2)    # (B, ch4+ch3, H/8, W/8) -> (B, ch3, H/4, W/4)

        # if not self.training:
        #     from basic.utils.logplot import get_root_plotter
        #     plotter = get_root_plotter(plot_sub_dir="cidnet/3")
        #     plotter.heatmap(hv_3, fig_name="hv3+")
        #     plotter.heatmap(i_enc3, fig_name="i3+")
        hv_2 = self.HV_LCA5(hv_3, i_dec3)           # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)          # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        hv_2 = self.HVD_block2(hv_2, hv_jump1)      # (B, ch3+ch2, H/4, W/4) -> (B, ch2, H/2, W/2)
        i_dec2 = self.ID_block2(i_dec2, i_jump1)    # (B, ch3+ch2, H/4, W/4) -> (B, ch2, H/2, W/2)

        # if not self.training:
        #     from basic.utils.logplot import get_root_plotter
        #     plotter = get_root_plotter(plot_sub_dir="cidnet/2")
        #     plotter.heatmap(hv_3, fig_name="hv2+")
        #     plotter.heatmap(i_enc3, fig_name="i2+")
        hv_1 = self.HV_LCA6(hv_2, i_dec2)           # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        i_dec1 = self.I_LCA6(i_dec2, hv_2)          # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)      # (B, ch2+ch1, H/2, W/2) -> (B, ch1, H, W)
        i_dec1 = self.ID_block1(i_dec1, i_jump0)    # (B, ch2+ch1, H/2, W/2) -> (B, ch1, H, W)

        # if not self.training:
        #     from basic.utils.logplot import get_root_plotter
        #     plotter = get_root_plotter(plot_sub_dir="cidnet/1")
        #     plotter.heatmap(hv_3, fig_name="hv1+")
        #     plotter.heatmap(i_enc3, fig_name="i1+")
        hv_0 = self.HVD_block0(hv_1)                # (B, ch1, H, W) -> (B, 2, H, W)
        i_dec0 = self.ID_block0(i_dec1)             # (B, ch1, H, W) -> (B, 1, H, W)
        return i_dec0, hv_0

    def _out_transform(self, i_dec0, hv_0, hvi):
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi  # (B, 3, H, W)
        output_rgb = self.trans.PHVIT(output_hvi)   # (B, 3, H, W)
        return output_rgb

    def forward(self, x):
        """
        Args:
            x: input image (RGB) with shape (B, 3, H, W)
        """
        i, hvi = self._in_transform(x)

        # [encoding]
        i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

        # [decoding]
        i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)

        # [output]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)

        return output_rgb

    def hvi_transform(self, x):
        hvi = self.trans.HVIT(x)
        return hvi


@ARCH_REGISTRY.register()
class VideoCIDNet(MemoryEnhancedNet, CIDNet):
    def __init__(
            self,
            channels=(36, 36, 72, 144),
            heads=(1, 2, 4, 8),
            norm=False,
            transform=None,
            memory_decoder=None,
            memory_trainable_only=False,
    ):
        CIDNet.__init__(
            self,
            channels, heads, norm, transform
        )
        MemoryEnhancedNet.__init__(
            self,
            i_memory_decoder=clone_module(memory_decoder),
            hv_memory_decoder=clone_module(memory_decoder),
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
        )

    def forward(self, x):
        return self.forward_memory(x)

    def forward_image(self, x):
        """
        Args:
            x: input image (RGB) with shape (B, 3, H, W)
        """
        i, hvi = self._in_transform(x)

        # [encoding]
        i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

        # [decoding]
        i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)

        # [output]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)

        return output_rgb

    def forward_memory(self, x, *args, **kwargs):
        with torch.no_grad():
            # [input]
            i, hvi = self._in_transform(x)

            # [encoding]
            i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

        # [memory]
        memory_i = self.memory_blocks_map['i_memory_decoder'](i_feats[-1])
        memory_hv = self.memory_blocks_map['hv_memory_decoder'](hv_feats[-1])
        i_feats = i_feats[:-1] + (memory_i,)
        hv_feats = hv_feats[:-1] + (memory_hv,)

        # [decoding]
        i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)

        # [output]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)

        return output_rgb


# bi-directional
@ARCH_REGISTRY.register()
class BiVideoCIDNet(MemoryEnhancedNet, CIDNet):
    def __init__(
            self,
            channels=(36, 36, 72, 144),
            heads=(1, 2, 4, 8),
            norm=False,
            transform=None,
            memory_decoder=None,
            memory_trainable_only=False,
            use_predict_value=True,
    ):
        CIDNet.__init__(
            self,
            channels, heads, norm, transform
        )
        MemoryEnhancedNet.__init__(
            self,
            i_memory_decoder=clone_module(memory_decoder),
            hv_memory_decoder=clone_module(memory_decoder),
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
        )
        self.use_predict_value = use_predict_value

    def forward(self, x):
        return self.forward_memory(x)

    def forward_image(self, x):
        """
        Args:
            x: input image (RGB) with shape (B, 3, H, W)
        """
        i, hvi = self._in_transform(x)

        # [encoding]
        i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

        # [decoding]
        i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)

        # [output]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)

        return output_rgb

    def forward_memory(self, x, *args, **kwargs):
        with torch.no_grad():
            # [input]
            i, hvi = self._in_transform(x)

            # [encoding]
            i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

        # [memory]
        memory_i = self.memory_blocks_map['i_memory_decoder'](i_feats[-1])
        memory_hv = self.memory_blocks_map['hv_memory_decoder'](hv_feats[-1])
        i_feats = i_feats[:-1] + (memory_i,)
        hv_feats = hv_feats[:-1] + (memory_hv,)

        # [decoding]
        i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)

        # [output]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)

        # [value encode]
        if self.use_predict_value:
            with torch.no_grad():
                _i, _hvi = self._in_transform(output_rgb)
                _i_feats, _hv_feats, _i_jumps, _hv_jumps = self._encode(_i, _hvi)

            self.memory_blocks_map['i_memory_decoder'].write_memory(memory_i, _i_feats[-1])
            self.memory_blocks_map['hv_memory_decoder'].write_memory(memory_hv, _hv_feats[-1])
        else:
            self.memory_blocks_map['i_memory_decoder'].write_memory(memory_i)
            self.memory_blocks_map['hv_memory_decoder'].write_memory(memory_hv)

        return output_rgb


# teacher forcing
@ARCH_REGISTRY.register()
class TFCIDNet(MemoryEnhancedNet, CIDNet):
    def __init__(
            self,
            channels=(36, 36, 72, 144),
            heads=(1, 2, 4, 8),
            norm=False,
            transform=None,
            memory_decoder=None,
            memory_trainable_only=False,
    ):
        CIDNet.__init__(
            self,
            channels, heads, norm, transform
        )
        MemoryEnhancedNet.__init__(
            self,
            i_memory_decoder=clone_module(memory_decoder),
            hv_memory_decoder=clone_module(memory_decoder),
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
        )

    def forward(self, x):
        return self.forward_tf(x)

    def forward_tf(self, x):
        with torch.no_grad():
            # [input]
            i, hvi = self._in_transform(x)

            # [encoding]
            i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

        # [memory]
        memory_i = self.memory_blocks_map['i_memory_decoder'].teacher_forcing_module(i_feats[-1])
        memory_hv = self.memory_blocks_map['hv_memory_decoder'].teacher_forcing_module(hv_feats[-1])
        i_feats = i_feats[:-1] + (memory_i,)
        hv_feats = hv_feats[:-1] + (memory_hv,)

        # [decoding]
        i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)

        # [output]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)

        return output_rgb


# teacher forcing test
@ARCH_REGISTRY.register()
class test_TFCIDNet(MemoryEnhancedNet, CIDNet):
    def __init__(
            self,
            channels=(36, 36, 72, 144),
            heads=(1, 2, 4, 8),
            norm=False,
            transform=None,
            memory_decoder=None,
            memory_trainable_only=False,
    ):
        CIDNet.__init__(
            self,
            channels, heads, norm, transform
        )
        MemoryEnhancedNet.__init__(
            self,
            i_memory_decoder=clone_module(memory_decoder),
            hv_memory_decoder=clone_module(memory_decoder),
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
        )

    def forward(self, x):
        return self.forward_tf(x)

    def forward_tf(self, x):
        # [encoding]
        with torch.no_grad():
            i, hvi = self._in_transform(x)
            i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

            # [decoding]
            i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)
            output_rgb = self._out_transform(i_dec0, hv_0, hvi)

            # [value encoding]
            i, hvi = self._in_transform(output_rgb)
            i_feats, hv_feats, i_jumps, hv_jumps = self._encode(i, hvi)

        # [teacher forcing]
        memory_i = self.memory_blocks_map['i_memory_decoder'].teacher_forcing_module(i_feats[-1])
        memory_hv = self.memory_blocks_map['hv_memory_decoder'].teacher_forcing_module(hv_feats[-1])
        i_feats = i_feats[:-1] + (memory_i,)
        hv_feats = hv_feats[:-1] + (memory_hv,)

        # [decoding]
        i_dec0, hv_0 = self._decode(i_feats, hv_feats, i_jumps, hv_jumps)
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)


        return output_rgb


if __name__ == '__main__':
    model = CIDNet().to('cuda')
    x = torch.randn(2, 3, 256, 224).to('cuda')
    y = model(x)
    print(y.shape)