## Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
## Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
## https://arxiv.org/pdf/2303.06705.pdf

from basic.archs.util import clone_module
from .util import *
from .retinexformer_arch import (
    Illumination_Estimator, Denoiser, RetinexFormer_Single_Stage,
    RetinexFormer
)
from basic.archs.memory.memory_enhanced_arch import MemoryEnhancedNet
from basic.archs.util_modules import Sequential
from basic.archs.tami_icmem.me_arch import ICMemoryEnhancedNet
from basic.utils.console.log import is_debug

from basic.utils.registry import ARCH_REGISTRY

class ICMemDenoiser(ICMemoryEnhancedNet, Denoiser):
    def __init__(
            self,
            in_dim=3, out_dim=3, dim=31,
            level=2, num_blocks=[2, 4, 4],
            memory_decoder=None,
            memory_trainable_only=False,
    ):
        Denoiser.__init__(
            self,
            in_dim=in_dim, out_dim=out_dim, dim=dim,
            level=level, num_blocks=num_blocks,
        )
        ICMemoryEnhancedNet.__init__(
            self,
            encoder=self.encode, decoder=self.decode,
            memory_decoder=memory_decoder,
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
        )

        self.memory_decoder = memory_decoder

    def encode(self, x, illu_fea):
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        return x, fea_encoder, illu_fea_list, illu_fea, fea

    def decode_multilayer(self, x, fea_encoder, illu_fea_list, illu_fea, fea):
        # feature map visualization
        # if not self.training:
        #     from basic.utils.logplot import get_root_plotter
        #     plotter = get_root_plotter(plot_sub_dir="retinexformer/igmem/illufea_decode")
        #     plotter.heatmap(illu_fea, fig_name=f"illufea+")
        fea = yield fea

        # Bottleneck
        fea = self._bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            # feature map visualization
            # if not self.training:
            #     from basic.utils.logplot import get_root_plotter
            #     plotter = get_root_plotter(plot_sub_dir="retinexformer/igmem/illufea_decode")
            #     plotter.heatmap(illu_fea, fig_name=f"illufea_{i}+")
            illu_fea = illu_fea_list[self.level - 1 - i]
            illu_fea = yield illu_fea
            fea = LeWinBlcok(fea, illu_fea)
        fea_out = fea

        # Mapping
        fea_out = self.mapping(fea_out) + x
        yield fea_out           # 3 ->

    def decode(self, x, fea_encoder, illu_fea_list, illu_fea, fea):
        # feature map visualization
        if is_debug() and not self.training:
            from basic.utils.console.logplot import get_root_plotter
            plotter = get_root_plotter(plot_sub_dir="retinexformer/igmem")
            plotter.heatmap(illu_fea, fig_name=f"illufea/illufea+")
            plotter.semantic_feature_map(illu_fea, fig_name=f"illufea/illufea_smtc+")
        fea_idendity = fea
        fea = yield fea
        if is_debug() and not self.training:
            from basic.utils.console.logplot import get_root_plotter
            plotter = get_root_plotter(plot_sub_dir="retinexformer/igmem")
            plotter.heatmap(fea_idendity, fig_name=f"fea_decode/illufea_before+")
            plotter.heatmap(fea, fig_name=f"fea_decode/illufea_after+")
            plotter.semantic_feature_map_joint(
                fea_idendity, fea,
                fig_name=f"fea_decode/illufea_smtc",
                fig_alias=["before+", "after+"]
            )

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            # feature map visualization
            # if not self.training:
            #     from basic.utils.logplot import get_root_plotter
            #     plotter = get_root_plotter(plot_sub_dir="retinexformer/igmem/illufea_decode")
            #     plotter.heatmap(illu_fea, fig_name=f"illufea_{i}+")
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)
        fea_out = fea

        # Mapping
        fea_out = self.mapping(fea_out) + x
        yield fea_out           # 3 ->

    def forward(self, x, illu_fea):
        return ICMemoryEnhancedNet.forward(self, x, illu_fea)

class ICMemRetinexFormer_Single_Stage(RetinexFormer_Single_Stage):
    def __init__(
            self,
            in_channels=3, out_channels=3, n_feat=31,
            level=2, num_blocks=[1, 1, 1],
            **memory_kwargs
    ):
        assert n_feat % (in_channels + 1) == 0, f"n_feat should be divisible by in_channels, but got {n_feat} and {in_channels}"

        super().__init__(
            in_channels=in_channels, out_channels=out_channels, n_feat=n_feat,
            level=level, num_blocks=num_blocks,
        )
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = ICMemDenoiser(
            in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
            num_blocks=num_blocks, **memory_kwargs
        )

    def forward(self, x):
        # img:        (b,3,h,w)
        # illu_fea:   (b,c,h,w)
        # illu_map:   (b,3,h,w)
        illu_fea, illu_map = self.estimator(x)
        input_x = x * illu_map + x
        output_x = self.denoiser(input_x, illu_fea)

        return output_x


@ARCH_REGISTRY.register()
class ICMemRetinexFormer(MemoryEnhancedNet, RetinexFormer):
    def __init__(self,
            in_channels=3,
            out_channels=3,
            n_feat=31,
            stage=3,
            num_blocks=[1, 1, 1],
            memory_decoder=None,
            memory_trainable_only=False,
            **kwargs
    ):
        memory_decoders = [
            clone_module(memory_decoder)
            for i in range(stage)
        ]
        RetinexFormer.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            n_feat=n_feat,
            stage=stage,
            num_blocks=num_blocks,
        )

        self.stage = stage

        modules_body = [
            ICMemRetinexFormer_Single_Stage(
                in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2,
                num_blocks=num_blocks,
                memory_decoder=memory_decoders[i],
                memory_trainable_only=memory_trainable_only
            )
            for i in range(stage)
        ]

        self.body = Sequential(*modules_body)

        MemoryEnhancedNet.__init__(
            self,
            **{
                f"memory_decoder_{i}": memory_decoders[i]
                for i in range(stage)
            },
            memory_trainable_only=memory_trainable_only,
            without_module_init=True,
            **kwargs
        )

    def forward(self, x):
        # (B, C, H, W)
        out = self.body(x)
        return out
