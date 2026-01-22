
import torch.nn as nn
from einops import rearrange

from .module.modules import (
    IlluminationEncoder, IlluminationRefiner,
    KeyProjector, ValueProjector,
    MDTA_TransformerBlock, IGMA_TransformerBlock, JAFARSiding,
    IGFE_TransformerBlock, IGFE_Transformer,
    IGFE_TransformerXBlock, IGFE_TransformerX,
    SpatialModulation, Projector
)
from .bank.memory_util import *

from basic.archs.helper.clone import LazyCloneModule


class ICMem(nn.Module):
    def __init__(
            self,
            dims=(3, 36, 36, 72, 144),
            key_dim=16, value_dim=64, hidden_dim=16,
            illu_feat_dim=64,

            projector_type='dwconv',
            ir_type='conv',
            tuning_module_type='modulation',
            illu_encoder_version=None,
            any_multilayer_feature_fusion=True,

            igfe_options=None,
    ):
        super().__init__()

        if igfe_options is None:
            igfe_options = {}

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.illu_encoder = IlluminationEncoder(illu_feat_dim=illu_feat_dim, version=illu_encoder_version)   # 用来 guide illu_decoder 产生的 illumination
        self._illu_encoder = {'frozen': LazyCloneModule(self.illu_encoder)}

        self.lq_projector = Projector(in_dim=dims[-1], out_dim=dims[-1], type=projector_type)
        self.hq_projector = Projector(in_dim=dims[-1], out_dim=dims[-1], type=projector_type)
        self.key_projector = KeyProjector(in_dim=dims[-1], key_dim=key_dim)
        self.value_projector = ValueProjector(in_dim=dims[-1], value_dim=value_dim, hidden_dim=hidden_dim)

        self.any_multilayer_feature_fusion = any_multilayer_feature_fusion
        if any_multilayer_feature_fusion:
            self.multilayer_feature_fuser = None

        illu_dim = 1
        if tuning_module_type.lower() == 'modulation':
            module_class = SpatialModulation
            base_igfe_options = dict(cond_activation='silu', cond_dim=illu_feat_dim, zero_init=True)
        elif tuning_module_type.lower() == 'mdta':
            module_class = MDTA_TransformerBlock
            base_igfe_options = dict(cond_dim=illu_feat_dim, num_heads=4, bias=False, zero_init_out=True)
        elif tuning_module_type.lower() == 'igma':
            module_class = IGMA_TransformerBlock
            base_igfe_options = dict(cond_dim=illu_feat_dim, num_heads=4, bias=False, zero_init_out=True)
        elif tuning_module_type.lower() == 'igfe':
            module_class = IGFE_TransformerBlock
            base_igfe_options = dict(cue_dim=illu_feat_dim)
        elif tuning_module_type.lower() == 'igfe_transformer':
            module_class = IGFE_Transformer
            base_igfe_options = dict(cue_dim=illu_feat_dim)
        elif tuning_module_type.lower() == 'igfe_x':
            module_class = IGFE_TransformerXBlock
            base_igfe_options = dict(cue_dim=illu_feat_dim)
        elif tuning_module_type.lower() == 'igfe_transformer_x':
            module_class = IGFE_TransformerX
            base_igfe_options = dict(cue_dim=illu_feat_dim)
        else:
            raise ValueError(f"Unknown igfe_type: {tuning_module_type}")
        tuning_modules = []
        start_i = max(-len(dims), -len(igfe_options))
        for i, dim in enumerate(reversed(dims[start_i:])):
            _igfe_options = dict(
                in_dim=dim, embed_dim=dim,
            )
            _igfe_options.update(base_igfe_options)
            _igfe_options.update(igfe_options[i])
            tuning_modules.append(
                module_class(**_igfe_options)
            )
        self.tuning_modules = nn.ModuleList(tuning_modules)

        self.scaling_module = self._scaling

        self.illu_decoder = IlluminationRefiner(
            in_dim=dims[-1], illu_feat_dim=illu_feat_dim, readout_dim=illu_dim,
            hidden_dim=hidden_dim, type=ir_type
        )

    @staticmethod
    def _scaling(feat, size_as):
        if feat.shape[-2:] != size_as.shape[-2:]:
            feat = F.interpolate(feat, size=size_as.shape[-2:], mode='bilinear', align_corners=False)
        return feat

    def set_encoder(self, key_encoder, value_encoder, is_multilayer=False):
        self.is_multilayer = is_multilayer
        self.encoders = dict(key=key_encoder, value=value_encoder)  # 由于是使用 shared weight，故无需注册到模块中

    def encode_key(self, *lq, need_s=False, need_e=False):
        feat = self.encoders['key'](*lq)        # (B, 3, H, W) -> (B, C, H, W), C=dims
        if isinstance(feat, tuple):
            feat = feat[-1]

        feat = self.lq_projector(feat)
        key, shrinkage, selection = self.key_projector(feat, need_s, need_e)
        return key, shrinkage, selection, feat

    def project_key(self, feat):
        feat = self.lq_projector(feat)
        return feat

    def encode_value(self, *hq, smtc_key, hidden_state=None, any_hidden_in=False):
        feat = self.encoders['value'](*hq)  # (B, 3+3, H, W) -> (B, C, H, W), C=dims
        if isinstance(feat, tuple):
            feat = feat[-1]

        feat = self.hq_projector(feat)
        value, hidden_state = self.value_projector(
            feat, smtc_key, hidden_state,
            any_hidden_in=any_hidden_in
        )
        return value, hidden_state, feat

    def encode_illus(self, hq, size_as=None, frozen=False):
        if not frozen:
            illu_encoder = self.illu_encoder
        else:
            illu_encoder = self._illu_encoder['frozen']
            illu_encoder.update(self.illu_encoder)
        illus = illu_encoder(hq, size_as)
        return illus

    def downscale(self, x, size_as):
        if x.shape[-2:] != size_as.shape[-2:]:
            x = F.interpolate(x, size=size_as.shape[-2:], mode='area', align_corners=None)
        return x

    def decode_illu(self, feat_key, memory_readout, hidden_state=None, hidden_out=False):
        illu_dec, hidden_state = self.illu_decoder.forward(
            feat_key, memory_readout, hidden_state,
            hidden_out=hidden_out
        )
        return illu_dec, hidden_state

    def retrieve(self, query, key, value):
        r_value = self.retrieve_module(query, key, value)
        return r_value

    # Used in training only.
    # This step is replaced by MemoryManager in test time
    def read_memory(
            self,
            query_key, memory_key, memory_value,
            memory_shrinkage=None, query_selection=None,
            softmax_func=None
    ):
        """
        query_key       : (B, C_K, H, W)
        query_selection : (B, C_K, H, W)
        memory_key      : (B, C_K, N)
        memory_shrinkage: (B, 1  , N)
        memory_value    : (B, C_V, N)
        """
        h, w = query_key.shape[-2:]
        affinity = get_affinity(memory_key, query_key, memory_shrinkage, query_selection, softmax_func=softmax_func)
        memory = readout(affinity, memory_value)
        memory = rearrange(memory, 'b c (h w) -> b c h w', h=h, w=w)

        return memory

    def forward(self, mode, *args, **kwargs):
        raise NotImplementedError