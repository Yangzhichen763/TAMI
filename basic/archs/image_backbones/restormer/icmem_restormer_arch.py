## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange

from basic.archs.tami_icmem.me_arch import ICMemoryEnhancedNet
from .restormer_arch import Restormer

from basic.utils.registry import ARCH_REGISTRY


##########################################################################
##---------- Restormer -----------------------
@ARCH_REGISTRY.register()
class ICMemRestormer(ICMemoryEnhancedNet, Restormer):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        memory_decoder=None,
        memory_trainable_only=False,
        any_multi_layer=False,
        **kwargs
    ):
        Restormer.__init__(
            self,
            inp_channels=inp_channels, out_channels=out_channels, dim=dim,
            num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
            heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
            LayerNorm_type=LayerNorm_type, dual_pixel_task=dual_pixel_task,
        )

        decode = self.decode_multilayer if any_multi_layer else self.decode
        ICMemoryEnhancedNet.__init__(
            self,
            encoder=self.encode, decoder=decode,
            memory_decoder=memory_decoder,
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
            **kwargs
        )

    def encode(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        return inp_img, inp_enc_level1, (out_enc_level1, out_enc_level2, out_enc_level3), inp_enc_level4

    def decode(self, inp_img, inp_enc_level1, enc_outs, inp_enc_level4):
        out_enc_level1, out_enc_level2, out_enc_level3 = enc_outs

        latent = self.latent(inp_enc_level4)

        latent = yield latent
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        yield out_dec_level1

