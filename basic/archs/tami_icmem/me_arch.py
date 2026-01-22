import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .network import ICMem
from .bank.memory_manager import MemoryManager
from .bank.memory_util import generate_gaussian_kernels, weighted_softmax

from basic.archs.memory.util import is_mem_engaged
from basic.archs.util import no_grad_if
from basic.metrics.summary import is_summary

from basic.archs.memory.enhanced_archs.memory_enhanced_encoder_decoder_arch import MemoryEnhancedEncoderDecoderNet
from basic.utils.shared_pool import SharedPool
from basic.archs.tami_icmem.util import get_illumination_map
from basic.losses.basic_loss import MSELoss

from basic.utils.console.log import is_debug

from basic.utils.registry import MODULE_REGISTRY


class ICMemoryEnhancedNet(MemoryEnhancedEncoderDecoderNet):
    def __init__(
            self, encoder, decoder, multilayer_encoder=None,
            memory_trainable_only=False, without_module_init=False, one_off_pass=False,
            **memory_blocks_dict
    ):
        # 适合用于多继承的情况，如果使用多继承，就可以设置 without_module_init=True，防止重复初始化
        super().__init__(
            encoder=encoder, decoder=decoder,
            memory_trainable_only=memory_trainable_only, without_module_init=without_module_init,
            one_off_pass=one_off_pass,
            **memory_blocks_dict
        )

        for memory_decoder in self.memory_blocks:
            memory_decoder.set_encoder_decoder(encoder, decoder, multilayer_encoder)

    def forward(self, *x, **kwargs):
        if len(kwargs.items()) > 0:
            x = x + tuple(kwargs.values())
        if is_mem_engaged():
            return self.forward_memory(*x)
        else:
            return self.forward_image(*x)

    def forward_image(self, *x):
        for block in self.memory_blocks:
            y = block.simple_forward(*x)
            return y

    def forward_memory(self, *x):
        for block in self.memory_blocks:
            if self.training:
                gt, = SharedPool.get("teacher_forcing").try_get("gt")
                illu_tgt = get_illumination_map(gt)
                SharedPool.get("teacher_forcing").clear_and_append("illu_tgt", illu_tgt)

                return block.train_forward(*x)
            else:
                return block.infer_forward(*x)


@MODULE_REGISTRY.register()
class ICMemoryDecoder(nn.Module):
    def __init__(
            self,
            # memory module args
            dims=(3, 36, 36, 72, 144),
            key_dim=32, value_dim=128, hidden_dim=32,
            illu_feat_dim=64,

            projector_type='dwconv',
            ir_type='conv',
            tuning_module_type='modulation',
            illu_encoder_version=None,

            # igfe options
            igfe_options=None,
            # memory bank args
            top_k=5,
            min_mid_term_frames=1, max_mid_term_frames=3,
            enable_long_term=False, max_long_term_elements=10000, num_long_term_prototypes=64,
            enable_long_term_count_usage=False,
            # optional settings
            memory_scale: float=None, memory_size=None,
            one_off_pass=False,
            teacher_forcing=False,
            # other settings
            ema = 0.99925,
            deep_update_prob=0.2,
            random_sample_during_training=True,
    ):
        super().__init__()

        self.memory_module = ICMem(
            dims=dims,
            key_dim=key_dim, value_dim=value_dim, hidden_dim=hidden_dim, illu_feat_dim=illu_feat_dim,

            projector_type=projector_type,
            ir_type=ir_type,
            tuning_module_type=tuning_module_type,
            illu_encoder_version=illu_encoder_version,

            igfe_options=igfe_options,
        )
        self.smtc_memory_bank = MemoryManager(
            hidden_dim=hidden_dim, top_k=top_k,
            max_mid_term_frames=max_mid_term_frames, min_mid_term_frames=min_mid_term_frames,
            enable_long_term=enable_long_term, max_long_term_elements=max_long_term_elements,
            num_long_term_prototypes=num_long_term_prototypes,
            enable_long_term_count_usage=enable_long_term_count_usage,
        )
        self.illu_memory_bank = MemoryManager(
            hidden_dim=hidden_dim, top_k=top_k,
            max_mid_term_frames=max_mid_term_frames, min_mid_term_frames=min_mid_term_frames,
            enable_long_term=enable_long_term, max_long_term_elements=max_long_term_elements,
            num_long_term_prototypes=num_long_term_prototypes,
            enable_long_term_count_usage=enable_long_term_count_usage,
        )

        self.gaussian_weights = None

        self.deep_update_prob = deep_update_prob
        self.memory_scale = memory_scale
        self.memory_size = memory_size
        self.one_off_pass = one_off_pass
        self.teacher_forcing = teacher_forcing
        self.random_sample_during_training = random_sample_during_training

        self.ema = ema


        # self._sim_loss_func = MSELoss(loss_weight=0.25)
        # self._illu_temporal_loss_func = MSELoss(loss_weight=0.25)
        # self._retinex_loss_func = MSELoss(loss_weight=0.25)
        self._sim_loss_func = MSELoss()
        self._illu_temporal_loss_func = MSELoss()
        self._retinex_loss_func = MSELoss()

    def set_encoder_decoder(self, encoder, decoder, multilayer_encoder=None):
        # encoder's last return value should be (B, dims[-1], H, W)
        # decoder's yield return value should be (B, dims[0], H, W), (B, dims[1], H, W), ..., (B, dims[-1], H, W)
        self._encoder = encoder
        self._decoder_iter = decoder

        if multilayer_encoder is None:
            self.memory_module.set_encoder(encoder, encoder, is_multilayer=False)
        else:
            self.memory_module.set_encoder(multilayer_encoder, multilayer_encoder, is_multilayer=True)
        return self

    def simple_forward(self, *x):
        # [encoding]
        with no_grad_if(self.one_off_pass):
            zs = self._encoder(*x)               # ==[*]== encoding

            if not isinstance(zs, tuple):
                zs = (zs,)
            z = zs[-1]

        # [decoding]
        dec = self._decoder_iter(*zs)                 # ==[*]== decoding
        _ = next(dec)
        for _ in self.memory_module.tuning_modules:
            try:
                z = dec.send(z)
            except StopIteration:
                break
        return z

    def _add_loss(self, loss):
        SharedPool.get('losses').append(
            "loss",
            loss
        )

    # noinspection SpellCheckingInspection
    def train_forward(self, *x, sequence_end=False):
        """
        abbreviations:
        smtcs -> semantics
        illus -> illuminations
        key -> key
        val -> value
        qry -> query
        """
        mem = self.memory_module
        smtc_bank = self.smtc_memory_bank
        illu_bank = self.illu_memory_bank
        lq = x[0]
        cue = x[1:]
        b = lq.shape[0]

        # ====================
        #       encoding
        # ====================
        with no_grad_if(self.one_off_pass):
            zs = self._encoder(*x)                  # ==[*]== encoding: image-model encoder forward

            if not isinstance(zs, tuple):
                zs = (zs,)

        # ====================
        #   retrieve memory
        # ====================
        # [key encoding]
        qry_key, shrinkage, selection, smtc_key = mem.encode_key(lq, *cue, need_s=not sequence_end, need_e=True)
        h, w = qry_key.shape[-2:]

        # [memory initialization]
        if smtc_bank.get_hidden() is None:
            smtc_bank.create_hidden_state(b, size_as=smtc_key)

        # [memory retrieval]
        if not smtc_bank.is_empty():
            ref_keys, ref_vals, ref_shrinkages, indices = smtc_bank.get_ref(batch_size=b, random=self.random_sample_during_training)
            _, ref_illu_vals, _, _indices = illu_bank.get_ref(batch_size=b, indices=indices, random=self.random_sample_during_training)

            smtc_key_readout = mem.read_memory(qry_key, ref_keys, ref_vals, ref_shrinkages, selection)
            illu_val_readout = mem.read_memory(smtc_key_readout, ref_vals, ref_illu_vals)
        else:
            illu_val_readout = torch.zeros(b, 1, h, w, device=qry_key.device)

        # ====================
        #  tuning & decoding
        # ====================
        dec = self._decoder_iter(*zs)               # get decoder iterator
        _z = z = next(dec)
        illu_val, hidden = mem.decode_illu(
            smtc_key, illu_val_readout, smtc_bank.get_hidden(),
            hidden_out=not sequence_end
        )
        delta_zs = []
        for i, tuning_module in enumerate(mem.tuning_modules):
            try:
                _illu = mem.scaling_module(illu_val, size_as=z)
                _illu_readout = mem.scaling_module(illu_val_readout, size_as=z)

                _z = tuning_module(z, _illu, _illu_readout)  # ==[*]== IGFE: tuning
                delta_zs.append(_z - z)
                z = dec.send(_z)                    # ==[*]== decoding: image-model decoder forward
            except StopIteration:
                break
        hq = z

        # ====================
        #  construct memory
        # ====================
        if not sequence_end:
            smtc_bank.set_hidden(hidden)

        # [memory update] save key/value feature and illumination as memory
        if not sequence_end:    # No need to encode the last frame
            if self.teacher_forcing:
                gt, = SharedPool.get("teacher_forcing").try_get("gt")

                # # guadual teacher forcing for longer training
                # progress, = SharedPool.get('memory').try_get('progress')
                # cosine_annealing_progress = (1 + math.cos(progress * math.pi)) / 2
                # if random.random() < cosine_annealing_progress:
                #     hq = gt
                hq = gt
            result_key = mem.project_key(_z)
            mem_val, hidden, smtc_val = mem.encode_value(
                hq, *cue, smtc_key=result_key,
                hidden_state=smtc_bank.get_hidden(), any_hidden_in=not sequence_end
            )  # (B, C, H, W)
            mem_illu = mem.encode_illus(hq=hq, size_as=smtc_key, frozen=True)

            # ----------------------------------------------------
            #           Symmetry Constract Loss
            # ----------------------------------------------------
            sim = self.get_similarity(smtc_key, smtc_val, any_4d_to_3d=True)
            self._add_loss(
                self._sim_loss_func(sim, sim.transpose(-1, -2))     # symmetry sililarity                                                                  # 对称相似度
                + self._sim_loss_func(
                    torch.diagonal(sim, dim1=-1, dim2=-2),
                    torch.diagonal(torch.ones_like(sim), dim1=-1, dim2=-2)
                )                                                   # identity similarity
            )

            if not hasattr(self, "smtc_keys"):
                self.smtc_keys = []
            if not hasattr(self, "smtc_vals"):
                self.smtc_vals = []
            i = torch.randint(0, len(self.smtc_keys), (1,))
            if len(self.smtc_keys) > 0:
                _smtc_key = self.smtc_keys[i]
                sim = self.get_similarity(_smtc_key, smtc_val, any_4d_to_3d=True)
                self._add_loss(
                    self._sim_loss_func(sim, sim.transpose(-1, -2)) # 对称相似度
                )
            if len(self.smtc_vals) > 0:
                _smtc_val = self.smtc_vals[i]
                sim = self.get_similarity(_smtc_val, smtc_key, any_4d_to_3d=True)
                self._add_loss(
                    self._sim_loss_func(sim, sim.transpose(-1, -2)) # 对称相似度
                )

            if hasattr(self, "smtc_keys"):
                self.smtc_keys.append(smtc_key)
            if hasattr(self, "smtc_vals"):
                self.smtc_vals.append(smtc_val)

            # ----------------------------------------------------
            #  Inter- and Intra-frame Illumination Alignment Loss
            # ----------------------------------------------------
            if not self.illu_memory_bank.is_empty():
                if not hasattr(self, "illu_val_readouts"):
                    self.illu_val_readouts = []
                    self.mem_illus = []
                elif len(self.illu_val_readouts) > 0:
                    illu_val_readout_last = self.illu_val_readouts[-1]
                    mem_illu_last = self.mem_illus[-1]
                    self._add_loss(
                        self._illu_temporal_loss_func(mem_illu - mem_illu_last, illu_val_readout - illu_val_readout_last)
                    )
                self._add_loss(
                    self._illu_temporal_loss_func(illu_val_readout, mem_illu)
                )
                self.illu_val_readouts.append(illu_val_readout)
                self.mem_illus.append(mem_illu)

            # ----------------------------------------------------
            #                     Retinex Loss
            # ----------------------------------------------------
            lq_illu = mem.encode_illus(hq=lq.detach(), size_as=smtc_key)
            hq_illu = mem.encode_illus(hq=hq.detach(), size_as=smtc_key)
            lq_rgb = mem.downscale(lq.detach(), size_as=smtc_key)
            hq_rgb = mem.downscale(hq.detach(), size_as=smtc_key)
            self._add_loss(
                self._retinex_loss_func(
                    lq_rgb * hq_illu - hq_rgb * lq_illu,
                    torch.zeros_like(lq_rgb)
                )
                # + self._retinex_loss_func(lq.detach().mean() + hq.detach().mean(), lq_illu.mean() + hq_illu.mean())
                # + self._retinex_loss_func(lq.detach().mean() - lq_illu.mean(), hq.detach().mean() - hq_illu.mean())
                + self._retinex_loss_func(lq.detach().mean() + hq_illu.mean(), lq_illu.mean() + hq.detach().mean())
            )

            smtc_bank.add_memory_without_compress(qry_key, mem_val, shrinkage, selection=selection if smtc_bank.enable_long_term else None)
            illu_bank.add_memory_without_compress(mem_val, mem_illu)
            smtc_bank.set_hidden(hidden)

        # ==[logging]==
        # compute delta_z and delta_y
        delta_z = torch.sum(torch.stack([
            torch.mean(delta_z, dim=(1, 2, 3), keepdim=True) for delta_z in delta_zs
            # (B, dim, H, W) -> (B, 1, 1, 1)
        ], dim=0), dim=0)  # n * (B, 1, 1, 1) -> (B, 1, 1, 1)
        SharedPool.get('scalars').append(
            "delta_z",
            torch.max(delta_z).detach().cpu().item()
        )
        SharedPool.get('scalars').append(
            "delta_y",
            torch.mean(-10 * torch.log10((gt - hq) ** 2 + 1e-12)).detach().cpu().item()
        )

        if is_debug("train") and self.training:
            from basic.utils.console.logplot import get_root_plotter
            step, = SharedPool.get("train").try_get("current_step")
            if step % 10 == 0:
                plotter = get_root_plotter(plot_sub_dir="phoattn")
                plotter.heatmap(
                    lq_illu,
                    fig_name=f"attn_{step}/lq_illu_heatmap",
                    show_colorbar=True
                )
                plotter.heatmap(
                    hq_illu,
                    fig_name=f"attn_{step}/hq_illu_heatmap",
                    show_colorbar=True
                )

        return z

    def infer_forward(self, *x, sequence_end=False):
        """
        abbreviations:
        smtcs -> semantics
        illus -> illuminations
        key -> key
        val -> value
        qry -> query
        """
        mem = self.memory_module
        smtc_bank = self.smtc_memory_bank
        illu_bank = self.illu_memory_bank
        lq = x[0]
        cue = x[1:]
        b = lq.shape[0]

        # ====================
        #  construct memory
        # ====================
        # encode memory if memory exists
        if hasattr(self, 'last_dict') and self.last_dict is not None:
            encode_memory_action = self.last_dict.get('action', None)
            if encode_memory_action is not None:
                encode_memory_action()

        # ====================
        #       encoding
        # ====================
        with no_grad_if(self.one_off_pass):
            zs = self._encoder(*x)               # ==[*]== encoding

            if not isinstance(zs, tuple):
                zs = (zs,)

        # ====================
        #   retrieve memory
        # ====================
        # [key encoding]
        qry_key, shrinkage, selection, smtc_key = mem.encode_key(lq, *cue, need_s=not sequence_end, need_e=True)
        h, w = qry_key.shape[-2:]

        # interpolate for faster inference speed
        qry_key = self.interpolate(qry_key, h, w)
        shrinkage = self.interpolate(shrinkage, h, w)
        selection = self.interpolate(selection, h, w)
        _h, _w = qry_key.shape[-2:]

        # [memory initialization]
        if smtc_bank.get_hidden() is None:
            smtc_bank.create_hidden_state(b, size_as=smtc_key)

        # [memory retrieval]
        if not smtc_bank.is_empty():
            smtc_key_readout = smtc_bank.match_memory(qry_key, selection)[0]    # (B, C_k, H, W) -> (B, C_v, H, W)
            illu_val_readout = illu_bank.match_memory(smtc_key_readout)[0]      # (B, C_v, H, W) -> (B, 1, H, W)
            illu_val_readout = self.interpolate_back(illu_val_readout, h, w)
        else:
            smtc_key_readout = torch.zeros(b, mem.value_dim, h, w, device=qry_key.device)
            illu_val_readout = torch.zeros(b, 1, h, w, device=qry_key.device)

        # ====================
        #  tuning & decoding
        # ====================
        dec = self._decoder_iter(*zs)               # get decoder iterator
        z = next(dec)
        z_lq = z
        illu_val, hidden = mem.decode_illu(
            smtc_key, illu_val_readout, smtc_bank.get_hidden(),
            hidden_out=not sequence_end
        )
        if not is_summary() and torch.any(torch.isnan(illu_val)):
            from basic.utils.console.log import get_root_logger
            logger = get_root_logger()
            logger.warning(f"nan count in illu_val_readout: {torch.sum(torch.isnan(illu_val))}")

        if not (is_debug() and not self.training):
            for i, tuning_module in enumerate(mem.tuning_modules):
                try:
                    _illu = mem.scaling_module(illu_val, size_as=z)
                    _illu_readout = mem.scaling_module(illu_val_readout, size_as=z)

                    z = tuning_module(z, _illu, _illu_readout)  # ==[*]== IGFE: tuning
                    z = dec.send(z)                    # ==[*]== decoding: image-model decoder forward
                except StopIteration:
                    break
        else:
            for i, tuning_module in enumerate(mem.tuning_modules):
                try:
                    _illu = mem.scaling_module(illu_val, size_as=z)
                    _illu_val_readout = mem.scaling_module(illu_val_readout, size_as=z)

                    z, out_maps = tuning_module(z, _illu, _illu_val_readout, return_out_map=True)  # ==[*]== IGFE: tuning

                    # [visualization]
                    from basic.utils.console.logplot import get_root_plotter
                    plotter = get_root_plotter(plot_sub_dir="icmem")
                    gt, = SharedPool.get("teacher_forcing").try_get("gt")
                    i_frame, = SharedPool.get('test').try_get("i")

                    z, out_maps = z
                    if not isinstance(out_maps, list):
                        out_maps = [out_maps]

                    for out_map in out_maps:
                        raw_attn = torch.softmax(out_map['raw_attn'], dim=-1)
                        attn = out_map['attn']
                        mask = torch.softmax(out_map['mask'], dim=-1)

                        _, heads, n = attn.shape[:3]
                        for _heads in range(heads):
                            if not smtc_bank.is_empty():
                                any_jump = plotter.attention_map_selection(
                                    gt,
                                    attn[:, [_heads]], raw_attn[:, [_heads]], mask[:, [_heads]],
                                    windows_names=[f"attn+", f"raw_attn+", f"mask+"],
                                    save_names=[f"attn_{_heads}+", f"raw_attn_{_heads}+", f"mask_{_heads}+"],
                                    fig_name=f"pred_illu_attn/attn_{i_frame}/"
                                )
                            else:
                                any_jump = plotter.attention_map_selection(
                                    gt,
                                    attn[:, [_heads]], raw_attn[:, [_heads]],
                                    windows_names=[f"attn+", f"raw_attn+"],
                                    save_names=[f"attn_{_heads}+", f"raw_attn_{_heads}+"],
                                    fig_name=f"pred_illu_attn/attn_{i_frame}/"
                                )
                            if any_jump:
                                break

                    z = dec.send(z)                    # ==[*]== decoding: image-model decoder forward

                except StopIteration:
                    break
        hq = z

        if not sequence_end:
            smtc_bank.set_hidden(hidden)

        if is_debug() and not self.training:
            from basic.utils.console.logplot import get_root_plotter
            plotter = get_root_plotter(plot_sub_dir="icmem")

            plotter.image(illu_val_readout, fig_name="pred_illu/pred+")
            plotter.semantic_feature_map(illu_val, fig_name="pred_illu/pred_smtc+")
            # if not smtc_bank.is_empty():
            #     for i in range(10 + 1):
            #         confidence = 0.1 * i
            #         def mask_func(x, *args, **kwargs):
            #             masked_softmax(x, *args, threshold=confidence, **kwargs)
            #         hc_illu_val_readout = illu_val_readout = illu_bank.match_memory(smtc_key_readout, softmax_func=mask_func)[0]
            #         plotter.image(hc_illu_val_readout, fig_name=f"pred_illu/pred_hc_{i}+")

            # 只输出 gt 图像，不参与模型推理
            gt, = SharedPool.get("teacher_forcing").try_get("gt")
            illu_gt = mem.encode_illus(hq=gt, size_as=illu_val_readout)
            plotter.image(illu_gt, fig_name="pred_illu/gt+")

        # ==[memory update]== save as memory
        def action():
            mem_val, hidden, smtc_val = mem.encode_value(
                hq, *cue, smtc_key=smtc_key,
                hidden_state=smtc_bank.get_hidden(), any_hidden_in=not sequence_end
            )  # (B, C, H, W)
            mem_illu = mem.encode_illus(hq=hq, size_as=smtc_key)

            # similarity visualization
            if is_debug() and not self.training:
                from basic.utils.console.logplot import get_root_plotter
                plotter = get_root_plotter(plot_sub_dir="icmem")
                # plotter.similarity_map(
                #     smtc_key, smtc_val, fig_name="image/kv_sim_cosine+",
                #     similarity_type="cosine", map_type="per_image"
                # )
                # plotter.similarity_map(
                #     smtc_key, smtc_val, fig_name="image_post/kv_sim_cosine+",
                #     similarity_type="cosine", map_type="per_image",
                #     similarity_post_process_func=lambda x: self.masked_softmax(x, norm=True)
                # )

                plotter.similarity_map(
                    smtc_key, smtc_val, fig_name="fig/kv_sim_cosine+",
                    similarity_type="cosine", map_type="figure"
                )

                plotter.semantic_feature_map_joint(
                    smtc_key, smtc_val,
                    fig_name="feat_map/retrieved/smtc",
                    fig_alias=["key+", "val+"]
                )
                if not self.smtc_memory_bank.is_empty():
                    plotter.semantic_feature_map(qry_key, fig_name="mem_map/qry_key+")
                    plotter.semantic_feature_map(mem_val, fig_name="mem_map/mem_val+")

            mem_val = self.interpolate(mem_val, h, w)
            mem_illu = self.interpolate(mem_illu, h, w)
            smtc_bank.add_memory(qry_key, mem_val, shrinkage,
                                 selection=selection if smtc_bank.enable_long_term else None)
            illu_bank.add_memory(mem_val, mem_illu)
            smtc_bank.set_hidden(hidden)

        if not hasattr(self, "last_dict"):
            self.last_dict = None
        self.last_dict = {
            "action": action
        }

        # [logging]==
        if is_debug() and not self.training:
            from basic.utils.console.logplot import get_root_plotter
            plotter = get_root_plotter(plot_sub_dir="icmem")

            with no_grad_if(self.one_off_pass):
                x = (hq,)
                zs = self._encoder(*x)

                if not isinstance(zs, tuple):
                    zs = (zs,)

                dec = self._decoder_iter(*zs)
                z_hq = next(dec)

            plotter.semantic_feature_map_joint(
                z_lq, z_hq, z_hq - z_lq,
                fig_name="feat_map/z/z_lq",
                fig_alias=["lq+", "hq+", "hq-lq+"]
            )


            # similarity
            if not hasattr(self, "values"):
                self.values = []

            self.values.append(smtc_val)
            # from basic.utils.logplot import get_root_plotter
            # plotter = get_root_plotter(plot_sub_dir="icmem")
            # t_values = torch.stack(self.values, dim=1)
            # t_keys = smtc_key.unsqueeze(dim=1)

            # plotter.similarity_map(
            #     t_keys, t_values, fig_name="t_fig/kv_sim_cosine+",
            #     similarity_type="cosine", map_type="figure"
            # )
            # for t in range(t_values.shape[1]):
            #     plotter.similarity_map(
            #         t_keys[:, 0], t_values[:, t], fig_name=f"t_image/{t_values.shape[1]}/{t}_kv_sim_cosine+",
            #         similarity_type="cosine", map_type="per_image"
            #     )
            #     plotter.similarity_map(
            #         t_keys[:, 0], t_values[:, t], fig_name=f"t_image_post/{t_values.shape[1]}/{t}_kv_sim_cosine+",
            #         similarity_type="cosine", map_type="per_image",
            #         similarity_post_process_func=lambda x: self.masked_softmax(x, norm=True)
            #     )

        return z

    def forward(self, *x):
        if self.training:
            return self.train_forward(*x)
        else:
            return self.infer_forward(*x)

    def reset_memory(self):
        self.smtc_memory_bank.reset()
        self.illu_memory_bank.reset()
        if hasattr(self, "smtc_keys"):
            self.smtc_keys = []
        if hasattr(self, "smtc_vals"):
            self.smtc_vals = []
        if hasattr(self, "illu_val_readouts"):
            self.illu_val_readouts = []
        if hasattr(self, "mem_illus"):
            self.mem_illus = []
        if hasattr(self, "last_dict"):
            self.last_dict = None

    def interpolate(self, x, h, w):
        if x is None:
            return None
        if self.memory_size is not None:
            memory_size = self.memory_size
            if isinstance(memory_size, int):
                memory_size = (memory_size, memory_size)
        elif self.memory_scale is not None:
            memory_size = (round(h * self.memory_scale), round(w * self.memory_scale))
        else:
            return x
        return F.interpolate(x, size=memory_size, mode='bilinear', align_corners=False)

    def interpolate_back(self, x, h, w):
        if self.memory_size is None and self.memory_scale is None:
            return x
        if x is None:
            return None
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

    def get_similarity(self, x, y, any_4d_to_3d=True, normalized=True):
        if any_4d_to_3d:
            x = rearrange(x, 'b c h w -> b (h w) c')
            y = rearrange(y, 'b c h w -> b (h w) c')
        if normalized:
            x_norm = F.normalize(x, dim=-1)      # (B, N, d)
            y_norm = F.normalize(y, dim=-1)     # (B, n, d)
        else:
            x_norm = x
            y_norm = y
        similarity = torch.matmul(y_norm, x_norm.transpose(-1, -2))  # (B, n, d) @ (B, N, d)^T -> (B, n, N)
        return similarity

    def init_gaussian_weights(self, ref):
        if self.gaussian_weights is None:
            self.gaussian_weights = generate_gaussian_kernels(*ref.shape[-2:], device=ref.device)

    def weighted_softmax(self, similarity, return_usage=False, **softmax_kwargs):
        return weighted_softmax(similarity, self.gaussian_weights, return_usage=return_usage, **softmax_kwargs)

    @staticmethod
    def masked_softmax(
            similarity, threshold=0.8, mode='softmax', norm=False,
    ):
        # similarity: (B, N, HW/P)
        x_mask = (similarity > threshold).float()

        if mode == 'softmax':
            x_masked = similarity.masked_fill(x_mask == 0, -1e9)
            F.softmax(x_masked)
        elif mode =='minmax':
            x_max = similarity.max()
            x_min = threshold
            similarity = similarity * x_mask + (1 - x_mask) * threshold
            similarity = (similarity - x_min) / (x_max - x_min) * x_max
        else:
            raise NotImplementedError

        if norm:
            x_max = similarity.max()
            x_min = similarity.min()
            if x_max - x_min > 1e-6:
                similarity = (similarity - x_min) / (x_max - x_min)

        return similarity

