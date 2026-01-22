from .urwkv_arch import URWKV
import torch

from basic.archs.tami_icmem.me_arch import ICMemoryEnhancedNet

from basic.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ICMemURWKV(ICMemoryEnhancedNet, URWKV):
    def __init__(
            self, dim,
            memory_decoder=None,
            memory_trainable_only=False,
            any_multi_layer=False,
            **kwargs
    ):
        URWKV.__init__(self, dim)

        decode = self.decode_multilayer if any_multi_layer else self.decode
        ICMemoryEnhancedNet.__init__(
            self,
            encoder=self.encode, decoder=decode,
            memory_decoder=memory_decoder,
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
            **kwargs
        )

    def encode(self, x):
        inter_feat = []
        encode_list, inter_feat = self.encoder(x, inter_feat)
        return x, encode_list, inter_feat, encode_list[-1]

    def decode(self, x, encode_list, inter_feat, _):
        outer_shortcut = x

        z = encode_list[-1]

        z = yield z
        encode_list[-1] = z
        z = self.decoder(z, encode_list, inter_feat)
        x = torch.add(z, outer_shortcut)

        yield x
