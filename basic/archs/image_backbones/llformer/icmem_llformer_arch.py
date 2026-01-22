## Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method
## Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, Tong Lu
## https://arxiv.org/pdf/2212.11548.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import numpy as np

from basic.archs.util import clone_module, no_grad_if
from .llformer_util import *
from .llformer_arch import *
from basic.archs.tami_icmem.me_arch import ICMemoryEnhancedNet

from basic.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ICMemLLFormer(ICMemoryEnhancedNet, LLFormer):
    def __init__(
            self,
            inp_channels=3,
            out_channels=3,
            dim = 16,
            num_blocks = [1,2,4,8],
            num_refinement_blocks = 2,
            heads = [1,2,4,8],
            ffn_expansion_factor = 2.66,
            bias = False,
            LayerNorm_type = 'WithBias',
            attention=True,
            skip = False,
            memory_decoder=None,
            memory_trainable_only=False,
            any_multi_layer=False,
            **kwargs
    ):
        LLFormer.__init__(
            self,
            inp_channels=inp_channels, out_channels=out_channels, dim=dim,
            num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
            heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
            LayerNorm_type=LayerNorm_type, attention=attention, skip=skip
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

    def _encode(self, x):
        # [In]
        inp_enc_encoder1 = self.patch_embed(x)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2)

        inp_fusion_123 = torch.cat(
            [out_enc_encoder1.unsqueeze(1), out_enc_encoder2.unsqueeze(1), out_enc_encoder3.unsqueeze(1)], dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        # [Down Sample]
        inp_enc_level1_0 = self.down_1(out_fusion_123)

        out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)
        inp_enc_level2_0 = self.down_2(out_enc_level1_0)

        out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)
        inp_enc_level3_0 = self.down_3(out_enc_level2_0)

        out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0)
        inp_enc_level4_0 = self.down_4(out_enc_level3_0)

        out_encodings = out_enc_level1_0, out_enc_level2_0, out_enc_level3_0
        return inp_enc_level4_0, out_fusion_123, out_encodings

    def _bottleneck(self, x_mid_in):
        x_mid_out = self.decoder_level4(x_mid_in)
        return x_mid_out

    def _decode(self, x, out_fusion_123, out_encodings):
        out_enc_level1_0, out_enc_level2_0, out_enc_level3_0, out_enc_level4_0 = out_encodings

        # [Up Sample]
        out_enc_level4_0 = self.up4_3(out_enc_level4_0)
        inp_enc_level3_1 = self.coefficient_4_3[0, :][None, :, None, None] * out_enc_level3_0 + self.coefficient_4_3[1, :][None, :, None, None] * out_enc_level4_0
        inp_enc_level3_1 = self.skip_4_3(inp_enc_level3_1)  ### conv 1x1
        out_enc_level3_1 = self.decoder_level3_1(inp_enc_level3_1)

        out_enc_level3_1 = self.up3_2(out_enc_level3_1)
        inp_enc_level2_1 = self.coefficient_3_2[0, :][None, :, None, None] * out_enc_level2_0 + self.coefficient_3_2[1, :][None, :, None, None] * out_enc_level3_1
        inp_enc_level2_1 = self.skip_3_2(inp_enc_level2_1)  ### conv 1x1
        out_enc_level2_1 = self.decoder_level2_1(inp_enc_level2_1)

        out_enc_level2_1 = self.up2_1(out_enc_level2_1)
        inp_enc_level1_1 = self.coefficient_2_1[0, :][None, :, None, None] * out_enc_level1_0 + self.coefficient_2_1[1, :][None, :, None, None] *  out_enc_level2_1
        inp_enc_level1_1 = self.skip_1_0(inp_enc_level1_1)  ### conv 1x1
        out_enc_level1_1 = self.decoder_level1_1(inp_enc_level1_1)

        out_enc_level1_1 = self.up2_0(out_enc_level1_1)

        # [Mid Skip]
        out_fusion_123 = self.latent(out_fusion_123)
        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123  + self.coefficient_1_0[1, :][None, :, None, None] *  out_enc_level1_1

        # [Out]
        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)
        inp_fusion = torch.cat([out_1.unsqueeze(1),out_2.unsqueeze(1),out_3.unsqueeze(1)],dim=1)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)
        out = self.conv_fuss_2(out_fusion_123)

        if self.skip:
            out = self.output(out) + x
        else:
            out = self.output(out)

        return out

    def encode(self, x):
        # [encoding]
        x_mid_in, out_fusion_123, out_encodings = self._encode(x)

        return x, out_fusion_123, out_encodings, x_mid_in

    def decode(self, x, out_fusion_123, out_encodings, x_mid_in):
        # [bottleneck]
        x_mid_out = self._bottleneck(x_mid_in)

        x_mid_out = yield x_mid_out

        # [decoding]
        x_out = self._decode(x, out_fusion_123, out_encodings + (x_mid_out,))
        yield x_out

    def decode_multilayer(self, x, out_fusion_123, out_encodings, x_mid_in):
        out_enc_level1_0, out_enc_level2_0, out_enc_level3_0 = out_encodings

        # [bottleneck]
        x_mid_out = self._bottleneck(x_mid_in)

        # [Up Sample]
        out_enc_level4_0 = yield x_mid_out
        out_enc_level4_0 = self.up4_3(out_enc_level4_0)
        inp_enc_level3_1 = self.coefficient_4_3[0, :][None, :, None, None] * out_enc_level3_0 + self.coefficient_4_3[1, :][None, :, None, None] * out_enc_level4_0
        inp_enc_level3_1 = self.skip_4_3(inp_enc_level3_1)  ### conv 1x1
        out_enc_level3_1 = self.decoder_level3_1(inp_enc_level3_1)

        out_enc_level3_1 = yield out_enc_level3_1
        out_enc_level3_1 = self.up3_2(out_enc_level3_1)
        inp_enc_level2_1 = self.coefficient_3_2[0, :][None, :, None, None] * out_enc_level2_0 + self.coefficient_3_2[1, :][None, :, None, None] * out_enc_level3_1
        inp_enc_level2_1 = self.skip_3_2(inp_enc_level2_1)  ### conv 1x1
        out_enc_level2_1 = self.decoder_level2_1(inp_enc_level2_1)

        out_enc_level2_1 = yield out_enc_level2_1
        out_enc_level2_1 = self.up2_1(out_enc_level2_1)
        inp_enc_level1_1 = self.coefficient_2_1[0, :][None, :, None, None] * out_enc_level1_0 + self.coefficient_2_1[1, :][None, :, None, None] *  out_enc_level2_1
        inp_enc_level1_1 = self.skip_1_0(inp_enc_level1_1)  ### conv 1x1
        out_enc_level1_1 = self.decoder_level1_1(inp_enc_level1_1)

        out_enc_level1_1 = yield out_enc_level1_1
        out_enc_level1_1 = self.up2_0(out_enc_level1_1)

        # [Mid Skip]
        out_fusion_123 = self.latent(out_fusion_123)
        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123  + self.coefficient_1_0[1, :][None, :, None, None] *  out_enc_level1_1

        # [Out]
        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)
        inp_fusion = torch.cat([out_1.unsqueeze(1),out_2.unsqueeze(1),out_3.unsqueeze(1)],dim=1)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)
        out = self.conv_fuss_2(out_fusion_123)

        if self.skip:
            out = self.output(out) + x
        else:
            out = self.output(out)

        yield out


