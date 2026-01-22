import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.archs.tami_icmem.me_arch import ICMemoryEnhancedNet
from .hvi_cid_arch import CIDNet

from basic.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ICMemCIDNet(ICMemoryEnhancedNet, CIDNet):
    def __init__(
            self,
            channels=(36, 36, 72, 144),
            heads=(1, 2, 4, 8),
            norm=False,
            transform=None,
            memory_decoder=None,
            memory_trainable_only=False,
            any_multi_layer=False,
            **kwargs
    ):
        CIDNet.__init__(
            self,
            channels, heads, norm, transform
        )

        decode = self.decode_multilayer if any_multi_layer else self.decode
        ICMemoryEnhancedNet.__init__(
            self,
            encoder=self.encode, decoder=decode, multilayer_encoder=self.encode_multilayer,
            i_memory_decoder=memory_decoder,
            memory_trainable_only=memory_trainable_only, without_module_init=True,
            one_off_pass=False,
            **kwargs
        )

    def encode_multilayer(self, x):
        # [in transform]
        i, hvi = self._in_transform(x)

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

        return hv_jump0, hv_jump1, hv_jump2, hv_3

    def decode_multilayer(self, hvi, jumps, feats, hv_3):
        (i_enc0, hv_0), (i_enc1, hv_1), (i_enc2, hv_2), (i_enc3, hv_3) = feats
        (i_jump0, i_jump1, i_jump2), (hv_jump0, hv_jump1, hv_jump2) = jumps

        # [decoding]
        hv_3 = yield hv_3
        hv_3 = self.HV_LCA4(hv_3, i_enc3)           # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)
        i_dec3 = self.I_LCA4(i_enc3, hv_3)          # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)
        hv_3 = self.HVD_block3(hv_3, hv_jump2)      # (B, ch4+ch3, H/8, W/8) -> (B, ch3, H/4, W/4)
        i_dec3 = self.ID_block3(i_dec3, i_jump2)    # (B, ch4+ch3, H/8, W/8) -> (B, ch3, H/4, W/4)

        hv_3 = yield hv_3
        hv_2 = self.HV_LCA5(hv_3, i_dec3)           # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)          # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        hv_2 = self.HVD_block2(hv_2, hv_jump1)      # (B, ch3+ch2, H/4, W/4) -> (B, ch2, H/2, W/2)
        i_dec2 = self.ID_block2(i_dec2, i_jump1)    # (B, ch3+ch2, H/4, W/4) -> (B, ch2, H/2, W/2)

        hv_2 = yield hv_2
        hv_1 = self.HV_LCA6(hv_2, i_dec2)           # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        i_dec1 = self.I_LCA6(i_dec2, hv_2)          # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)      # (B, ch2+ch1, H/2, W/2) -> (B, ch1, H, W)
        i_dec1 = self.ID_block1(i_dec1, i_jump0)    # (B, ch2+ch1, H/2, W/2) -> (B, ch1, H, W)

        hv_1 = yield hv_1
        hv_0 = self.HVD_block0(hv_1)                # (B, ch1, H, W) -> (B, 2, H, W)
        i_dec0 = self.ID_block0(i_dec1)             # (B, ch1, H, W) -> (B, 1, H, W)

        # [out transform]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)
        yield output_rgb

    def encode(self, x):
        # [in transform]
        i, hvi = self._in_transform(x)

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

        return hvi, ((i_jump0, i_jump1, i_jump2), (hv_jump0, hv_jump1, hv_jump2)), ((i_enc0, hv_0), (i_enc1, hv_1), (i_enc2, hv_2), (i_enc3, hv_3)), hv_3

    def decode(self, hvi, jumps, feats, hv_3):
        (i_enc0, hv_0), (i_enc1, hv_1), (i_enc2, hv_2), (i_enc3, _) = feats
        (i_jump0, i_jump1, i_jump2), (hv_jump0, hv_jump1, hv_jump2) = jumps

        # [decoding]
        hv_3 = yield hv_3
        hv_3 = self.HV_LCA4(hv_3, i_enc3)           # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)
        i_dec3 = self.I_LCA4(i_enc3, hv_3)          # (B, ch4+ch4, H/8, W/8) -> (B, ch4, H/8, W/8)
        hv_3 = self.HVD_block3(hv_3, hv_jump2)      # (B, ch4+ch3, H/8, W/8) -> (B, ch3, H/4, W/4)
        i_dec3 = self.ID_block3(i_dec3, i_jump2)    # (B, ch4+ch3, H/8, W/8) -> (B, ch3, H/4, W/4)

        hv_2 = self.HV_LCA5(hv_3, i_dec3)           # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)          # (B, ch3+ch3, H/4, W/4) -> (B, ch3, H/4, W/4)
        hv_2 = self.HVD_block2(hv_2, hv_jump1)      # (B, ch3+ch2, H/4, W/4) -> (B, ch2, H/2, W/2)
        i_dec2 = self.ID_block2(i_dec2, i_jump1)    # (B, ch3+ch2, H/4, W/4) -> (B, ch2, H/2, W/2)

        hv_1 = self.HV_LCA6(hv_2, i_dec2)           # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        i_dec1 = self.I_LCA6(i_dec2, hv_2)          # (B, ch2+ch2, H/2, W/2) -> (B, ch2, H/2, W/2)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)      # (B, ch2+ch1, H/2, W/2) -> (B, ch1, H, W)
        i_dec1 = self.ID_block1(i_dec1, i_jump0)    # (B, ch2+ch1, H/2, W/2) -> (B, ch1, H, W)

        hv_0 = self.HVD_block0(hv_1)                # (B, ch1, H, W) -> (B, 2, H, W)
        i_dec0 = self.ID_block0(i_dec1)             # (B, ch1, H, W) -> (B, 1, H, W)

        # [out transform]
        output_rgb = self._out_transform(i_dec0, hv_0, hvi)
        yield output_rgb

    def forward(self, x):
        return ICMemoryEnhancedNet.forward(self, x)

