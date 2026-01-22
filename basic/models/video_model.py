from collections import OrderedDict
import inspect
import functools

import torch
import torch.nn as nn

from basic.models.base_model import BaseModel
from basic.archs.memory.memory_enhanced_arch import MemoryEnhancedNet
from basic.metrics import timer
from basic.archs import define_network
from basic.losses import get_loss_func, get_all_loss_func
from basic.utils.convert import padding
from basic.utils.dist import master_only
from basic.archs.memory.util import mem_engaged, no_mem_engaged, mem_engaged_if

from basic.utils.console.log import log_context
from basic.metrics.summary import print_frozen_params
from basic.utils.shared_pool import SharedPool

from basic.utils.registry import MODEL_REGISTRY


def normalize_optimizers(func):
    @functools.wraps(func)
    def wrapper(self, optimizers=None, *args, **kwargs):
        if optimizers is None:
            optimizers = self.get_optimizers_by_names(optimizers)
        if not isinstance(optimizers, list) and not isinstance(optimizers, tuple):
            optimizers = [optimizers]
        if isinstance(optimizers[0], str):
            optimizers = self.get_optimizers_by_names(optimizers)
        return func(self, optimizers, *args, **kwargs)

    return wrapper


@MODEL_REGISTRY.register()
class OnlineVideoModel(BaseModel):
    def __init__(self, opt, verbose=True):
        super(OnlineVideoModel, self).__init__(opt, verbose=verbose)

        ### [Model]
        with log_context(lambda: self.logger,
                         start_msg='Building model...',
                         end_msg='Model is built.') as _logger:
            # [build model] 构建模型
            self.net = define_network(opt['network'])
            self.net = self.model_to_device(self.net)

            # [load resume state] 加载模型断点状态信息
            self.load_resume_state(opt)

            # [load pre-trained weights] 加载预训练模型
            self.load_pretrain_weights(self.net, opt)


        ### [Output & Loggers]
        self.logger_init(opt)


        ### [Train Settings]
        if self.is_train:
            self.net.train()

            # [Loss] 损失函数
            self.loss_funcs = []        # 重建损失 reconstruction loss
            for loss_conf in opt['train']['loss']:
                loss_func = get_loss_func(loss_conf).to(self.device)
                self.loss_funcs.append(loss_func)
            self.latent_loss_funcs = {} # 潜空间损失 latent loss
            if 'latent_loss' in opt['train']:
                self.latent_loss_funcs = get_all_loss_func(opt['train']['latent_loss'])
            self.reg_loss_funcs = []    # 正则化损失 regularization loss
            if 'reg_loss' in opt['train']:
                for loss_conf in opt['train']['reg_loss']:
                    loss_func = get_loss_func(loss_conf).to(self.device)
                    self.reg_loss_funcs.append(loss_func)

            # [Optimizer & Schedulers] 优化器 & 学习率调节器
            self.setup_optimizers(self.net)
            self.setup_schedulers()

            self.grad_clip = self._is_grad_clip()
            self.grad_clip_params = self._init_grad_clip_params()
        else:
            self.net.eval()

        ### [Others]
        if self.is_train:
            if hasattr(self, 'amp_scaler'):
                self._backward_func = self._backward_amp
                self._step_optimizer_func = self._step_optimizer_amp
            else:
                self._backward_func = self._backward
                self._step_optimizer_func = self._step_optimizer

            if 'video_backward_handler' in opt['train']:
                self.bpv_handler = getattr(self, f"_bpv_{opt['train']['video_backward_handler']}")
                _logger.info(f"Video backward handler is set to {opt['train']['video_backward_handler']}")
            else:
                self.bpv_handler = self._bpv_frame_wise
                _logger.info(f"Video backward handler is not specified, use frame-wise backward.")

            if 'backward_handler' in opt['train']:
                self.bp_handler = getattr(self, f"_bp_{opt['train']['backward_handler']}")
                _logger.info(f"Backward handler is set to {opt['train']['backward_handler']}")
            else:
                self.bp_handler = self._bp_normal
                _logger.info(f"Backward handler is not specified, use frame-wise backward.")

            self.loss_acc = []

        self.lq = None
        self.gt = None
        self.pred = None

        self.lqs = []
        self.gts = []
        self.preds = []

        ### [Val Settings]
        self.val_metrics_settings(opt)

    #region ==[Train]==
    #region ==[Default Train & Test]==
    # 相当于训练的一个 iteration
    def optimize_parameters(self, lq=None, gt=None, memory_end=False, net_kwargs=None, loss_kwargs=None):
        if net_kwargs is None:
            net_kwargs = {}
        if loss_kwargs is None:
            loss_kwargs = {}
        gt = self.gt if gt is None else gt
        lq = self.lq if lq is None else lq

        # [inference] 模型推理
        with mem_engaged():
            pred = self.net(lq, **net_kwargs)

        # [loss] 计算损失和反向传播
        loss =  torch.scalar_tensor(0.0, device=self.device)
        loss_img = self._sum_losses(pred, gt, **loss_kwargs)
        loss += loss_img

        # regularization loss
        for reg_loss_func in self.reg_loss_funcs:
            reg_loss = reg_loss_func(self.net)
            loss += reg_loss

        # other loss
        while (_loss := SharedPool.get('losses').try_pop('loss')) is not None:
            _loss, = _loss
            if torch.tensor(_loss.shape).prod().item() > 1:
                _loss = _loss.mean()
            loss += _loss

        # [params update] 更新参数
        if self._bp_sequence_wise_backward.__name__ == self.bp_handler.__name__:
            if not hasattr(self, 'losses'):
                self.losses = []
            self.losses.append(loss)

            if memory_end:
                loss = torch.stack(self.losses).mean()

                self.losses = []
        self.loss_backward_and_optimize_parameters(loss, memory_end=memory_end)

        self.pred = pred
        self.log_dict = dict(
            loss=loss.item(),
            lr=self.get_current_learning_rate()
        )

        # 输出用于可视化的 scalar 信息
        for key, items in SharedPool.get('scalars').pop_all():
            for i in range(len(items)):
                self.log_dict.update({f"{key}_{i}": items[i]})

    # 相当于测试的一个 iteration
    def test(self, lq=None, padding_mul=8, net_kwargs=None):
        if net_kwargs is None:
            net_kwargs = {}
        lq = self.lq if lq is None else lq

        with mem_engaged_if(isinstance(self.net, MemoryEnhancedNet)), \
                torch.inference_mode(), \
                padding(lq, mul=padding_mul) as (padded_lq, unpadding), timer() as t:
            pred = self.net(padded_lq, **net_kwargs)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = unpadding(pred)

        self.pred = torch.clamp(pred, 0, 1)
        self.log_dict = dict(
            latency=t.elapsed()
        )
    #endregion

    def feed_data(self, data, need_gt=True):
        self.lq = data['lq']['image'].to(self.device)
        if need_gt:
            if 'gt' not in data:
                raise ValueError('No GT found in data for feeding data! Please check your dataset configuration.')
            self.gt = data['gt']['image'].to(self.device)

            SharedPool.get("teacher_forcing").clear_and_append("gt", self.gt)

    #region --backward--
    def _backward(self, loss, optimizer=None, **loss_kwargs):
        loss.backward(**loss_kwargs)

    def _backward_amp(self, loss, optimizer=None, **loss_kwargs):
        self.amp_scaler.scale(loss).backward(**loss_kwargs)
        self.amp_scaler.step(optimizer)
    #endregion

    #region --gradient clip--
    def _is_grad_clip(self):
        if isinstance(self.train_opt['grad_clip'], dict):
            grad_clip = self.train_opt['grad_clip']['value']
        elif isinstance(self.train_opt['grad_clip'], bool):
            grad_clip = self.train_opt['grad_clip']
        else:
            grad_clip = False

        return grad_clip

    def _init_grad_clip_params(self):
        if 'grad_clip' not in self.train_opt:
            return None
        # max_norm 指允许的最大梯度范数，保持梯度方向，缩放大小
        params = dict(max_norm=0.01, norm_type=2)
        if isinstance(self.train_opt['grad_clip'], dict):
            params.update(self.train_opt['grad_clip'].get('params', {}))
        return params

    def _try_clip_grad_norm(self, allow_name="all"):
        if not self.grad_clip:
            return
        if allow_name == "all" or not isinstance(self.net, MemoryEnhancedNet):
            nn.utils.clip_grad_norm_(self.net.parameters(), **self.grad_clip_params)
            return
        params = self.parts[allow_name]['params']
        nn.utils.clip_grad_norm_(params, **self.grad_clip_params)
    #endregion

    #region --step optimizer--
    @normalize_optimizers
    def _step_optimizer(self, optimizers):
        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad()

    @normalize_optimizers
    def _zero_grad(self, optimizers):
        for optimizer in optimizers:
            optimizer.zero_grad()

    @normalize_optimizers
    def _step_optimizer_amp(self, optimizers):
        for optimizer in optimizers:
            self.amp_scaler.step(optimizer)
            self.amp_scaler.update()
        for optimizer in optimizers:
            optimizer.zero_grad()
    #endregion

    #region --backpropagation handler--
    def _bp_normal(self, loss, part: str, memory_end, **loss_kwargs):
        self._backward_func(loss, **loss_kwargs)
        self._try_clip_grad_norm(part)
        self._step_optimizer_func(part)

    def _bp_frame_wise(self, loss, part: str, memory_end, **loss_kwargs):
        # 反向传播的时候，每帧更新一次
        self._backward_func(loss, **loss_kwargs)
        self._try_clip_grad_norm(part)
        self._step_optimizer_func(part)
        if 'memory' in part and memory_end:
            self.net.reset_memory()

    def _bp_sequence_wise_delayed(self, loss, part: str, memory_end, **loss_kwargs):
        # 反向传播的时候，每个序列更新一次
        self._backward_func(loss, **loss_kwargs)
        if memory_end:
            self._try_clip_grad_norm(part)
            self._step_optimizer_func(part)
            if 'memory' in part:
                self.net.reset_memory()
        else:  # 梯度没有累积
            self._zero_grad()

    def _bp_sequence_wise_gradient_accumulation(self, loss, part: str, memory_end, **loss_kwargs):
        # 反向传播的时候，每个序列更新一次，梯度累积
        self._backward_func(loss, **loss_kwargs)
        if memory_end:
            self._try_clip_grad_norm(part)
            self._step_optimizer_func(part)
            if 'memory' in part:
                self.net.reset_memory()

    def _bp_sequence_wise_backward(self, loss, part: str, memory_end, **loss_kwargs):
        # 每个序列反向传播一次，此时这个 loss 是所有序列的 loss 的平均值
        if memory_end:
            self._backward_func(loss, **loss_kwargs)
            self._try_clip_grad_norm(part)
            self._step_optimizer_func(part)
            if 'memory' in part:
                self.net.reset_memory()
    #endregion

    #region --image model backpropagation--
    def _bpi_normal(self, loss, memory_end, **loss_kwargs):
        self._backward_func(loss, **loss_kwargs)
        self._try_clip_grad_norm('image_model')
        self._step_optimizer_func('image_model')
    #endregion

    #region --memory module backpropagation--
    def _bpv_frame_wise(self, loss, memory_end, **loss_kwargs):
        # 反向传播的时候，每帧更新一次
        self._backward_func(loss, **loss_kwargs)
        self._try_clip_grad_norm('memory_decoder')
        self._step_optimizer_func('memory_decoder')
        if memory_end:
            self.net.reset_memory()

    def _bpv_sequence_wise_delayed(self, loss, memory_end, **loss_kwargs):
        # 反向传播的时候，每个序列更新一次
        self._backward_func(loss, **loss_kwargs)
        if memory_end:
            self._try_clip_grad_norm('memory_decoder')
            self._step_optimizer_func('memory_decoder')
            self.net.reset_memory()
        else:  # 梯度没有累积
            self._zero_grad('memory_decoder')

    def _bpv_sequence_wise_gradient_accumulation(self, loss, memory_end, **loss_kwargs):
        # 反向传播的时候，每个序列更新一次，梯度累积
        self._backward_func(loss, **loss_kwargs)
        if memory_end:
            self._try_clip_grad_norm('memory_decoder')
            self._step_optimizer_func('memory_decoder')
            self.net.reset_memory()

    def _bpv_sequence_wise_backward(self, loss, memory_end, **loss_kwargs):
        # 每个序列反向传播一次，此时这个 loss 是所有序列的 loss 的平均值
        if memory_end:
            self._backward_func(loss, **loss_kwargs)
            self._try_clip_grad_norm('memory_decoder')
            self._step_optimizer_func('memory_decoder')
            self.net.reset_memory()
    #endregion

    def loss_backward_and_optimize_parameters(self, loss, memory_end, **loss_kwargs):
        SharedPool.get('train').clear_and_append('backward_done', memory_end)

        if hasattr(self.net, 'reset_memory'):
            for part in self.parts.keys():
                self.bp_handler(loss, part=part, memory_end=memory_end, **loss_kwargs)
        else:
            for part in self.parts.keys():
                self._bp_normal(loss, part=part, memory_end=memory_end, **loss_kwargs)

    def loss_backward_and_optimize_parameters_image_model(self, loss, memory_end, **loss_kwargs):
        SharedPool.get('train').clear_and_append('backward_done', memory_end)

        self._bpi_normal(loss, memory_end=memory_end, **loss_kwargs)

    def loss_backward_and_optimize_parameters_memory_module(self, loss, memory_end, **loss_kwargs):
        SharedPool.get('train').clear_and_append('backward_done', memory_end)

        self.bpv_handler(loss, memory_end=memory_end, **loss_kwargs)
    #endregion

    #region ==[Store & Measures]==
    def reset_memory(self):
        if hasattr(self.net, 'reset_memory'):
            self.net.reset_memory()

    def append_history(self, lq=None, gt=None, pred=None):
        """
        Args:
            lq (Tensor): LQ image tensor with shape (C, H, W), from get_current_visuals().
            gt (Tensor): GT image tensor with shape (C, H, W), from get_current_visuals().
            pred (Tensor): predicted image tensor with shape (C, H, W), from get_current_visuals().
        """
        self.lqs.append(lq)
        self.gts.append(gt)
        self.preds.append(pred)

    def clear_history(self):
        self.lqs = []
        self.gts = []
        self.preds = []

    def calculate_metrics(self, preds=None, gts=None):
        """
        Args:
            preds (list of Tensor): predicted image tensor with shape (N, C, H, W) or tensor list with length N,  shape (C, H, W).
            gts (list of Tensor): GT image tensor with shape (N, C, H, W) or tensor list with length N,  shape (C, H, W).
        """
        preds = self.preds if preds is None else preds
        gts = self.gts if gts is None else gts

        from basic.utils.console.log import get_root_logger
        metrics_logger = get_root_logger("metrics")
        metrics_logger.info(f"metrics data length: {len(preds)}")

        log_dict = {}
        for metrics_label, metrics_func in self.metric_funcs.items():
            # get the number of parameters of the metrics function
            num_params = len(inspect.signature(metrics_func).parameters)

            # calculate the metrics
            if preds is None:
                raise ValueError('Please run test() or optimize_parameters() first to get pred image.')
            if num_params == 2:
                if gts is None:
                    raise ValueError('Please feed_data() with need_gt=True to get gt image.')
                metrics = metrics_func(preds, gts)
            elif num_params == 1:
                metrics = metrics_func(preds)
            else:
                raise ValueError(
                    f"Metrics function {metrics_func.__name__} should have 1 or 2 parameters, but got {num_params}!")
            log_dict[metrics_label] = metrics.item()

        # self.log_dict.update(log_dict)    # 不更新 self.log_dict，以防出现重复使用
        return log_dict

    #endregion

    def get_current_visuals(self, need_gt=True):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach()[0].float().cpu()
        out_dict['pred'] = self.pred.detach()[0].float().cpu()
        if need_gt:
            if self.gt is None:
                raise ValueError(
                    'No GT found in data for feeding data! Please check your dataset configuration or feed_data() with need_gt=True first.')
            out_dict['gt'] = self.gt.detach()[0].float().cpu()
        return out_dict.copy()

    def get_current_best_metric_log(self, phase='val'):
        """
        Get the best metric result of the specified phase.

        Returns:
            dict: the best metric result of the specified phase. e.g.
            {
                'PSNR': {
                    'value': 27.5,
                    'iter': 1000,
                },
                'SSIM': {
                    'value': 0.85,
                    'iter': 1500,
                },
            }
        """
        self._try_initialize_best_metric_results(phase)
        if phase not in self.best_metric_results.keys():
            self.logger.error(f"Phase {phase} not found in best_metric_results. "
                              f" The best metric model will not be saved. "
                              f" Please check your phase name.")
            return

        return self.best_metric_results[phase]

    # override
    def setup_optimizers(self, net):
        # optimize the unfrozen parameters
        optim_params = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        # print frozen parameters
        if len(optim_params) != len(list(net.parameters())):
            print_frozen_params(net, pre_message_str='Frozen parameters will not be optimized:\n')

        if isinstance(self.opt['train']['optimizer'], list):
            for optim_option in self.opt['train']['optimizer']:
                optim_name = optim_option.pop('name')
                if optim_name == 'image_model' or 'image' in optim_name:
                    if isinstance(net, MemoryEnhancedNet):
                        _optim_params = net.get_non_memory_parameters()  # e.g. {'encoder_decoder': Adam(...), 'connection': Adam(...)}
                    else:
                        _optim_params = optim_params
                elif optim_name == 'memory_decoder' or 'memory' in optim_name:
                    _optim_params = net.get_memory_parameters()
                elif optim_name == 'all' or optim_name == 'joint':
                    _optim_params = optim_params
                else:
                    raise ValueError(f"Optimizer name {optim_name} not found in the network!")
                self.optimizers[optim_name] = self.get_optimizer(_optim_params, optim_option) # e.g. {'encoder_decoder': Adam(...), 'connection': Adam(...)}

                if optim_name not in self.parts:
                    self.parts[optim_name] = {}
                self.parts[optim_name]['params'] = _optim_params
        else:
            optim_name = 0
            self.optimizers = {optim_name: self.get_optimizer(optim_params)} # e.g. [Adam(...)]
            if optim_name not in self.parts:
                self.parts[optim_name] = {}
            self.parts[optim_name]['params'] = optim_params


    def setup_schedulers(self):
        """
        Set up schedulers for each optimizer from the options.
        """
        schedulers = {}
        for name, optimizer in self.optimizers.items():
            if name == 0:
                name = None
            scheduler = self.get_scheduler(optimizer, name=name)
            schedulers[name] = scheduler
        self.schedulers = schedulers


    @master_only
    def save(self, epoch, current_iter, name=None, **other_state_dict):
        """
        Save the current [training state & checkpoints].

        Args:
            epoch (int): current epoch
            current_iter (int): current iteration
            name (str): name of the saved state
            other_state_dict (dict): other state to be saved, such as optimizer, scheduler, etc.
        """
        name = name or f'{current_iter}'
        pretrain_model_path = self.save_network(self.net, name, self.net_label)
        training_state_path = self.save_training_state(epoch, current_iter, state_name=name, **other_state_dict)
        return pretrain_model_path, training_state_path

    @master_only
    def try_save_best(self, epoch, current_iter, on_save_best=None, phase='val', **other_state_dict):
        """
        Save the best [training state & checkpoints] according to the validation metrics.
        If the current validation metric is better than the previous best, save the current state as the best state.
        Else, do nothing.

        Args:
            epoch (int): current epoch
            current_iter (int): current iteration
            on_save_best (function): a function to be called when the best state is saved.
            other_state_dict (dict): other state to be saved, such as optimizer, scheduler, etc.
        """
        self._try_initialize_best_metric_results(phase)
        if phase not in self.best_metric_results.keys():
            self.logger.error(f"Phase {phase} not found in best_metric_results. "
                              f" The best metric model will not be saved. "
                              f" Please check your phase name.")
            return

        for metric_label, _ in self.metric_funcs.items():
            value = self.log_dict[metric_label]
            is_best_metrics = self._is_best_metric_result(phase, metric_label, value)
            self._update_best_metric_result(phase, metric_label, value, current_iter)
            if is_best_metrics:
                on_save_best(metric_label, value)
                self.save_network(self.net, f'best_{metric_label}', self.net_label)
                self.save_training_state(epoch, current_iter, state_name=f'best_{metric_label}',
                                         **{metric_label: value}, **other_state_dict)

    def __getattr__(self, item):
        # 如果 model 中不存在该属性，则尝试从 net 中获取
        if not hasattr(self, "net"):
            raise AttributeError(f"No network found in {self.__class__.__name__}!")

        if hasattr(self.net, item):
            return getattr(self.net, item)
        else:
            raise AttributeError(f"No attribute named '{item}' found in {self.__class__.__name__}!")


def tensor_5D_to_4D(tensor):
    """
    Convert a 5D tensor to a 4D tensor by removing the first dimension.

    Args:
        tensor (torch.Tensor): input tensor with shape (N, T, C, H, W)

    Returns:
        torch.Tensor: output tensor with shape (N * T, C, H, W)
    """
    B, T, C, H, W = tensor.shape
    return tensor.reshape(B * T, C, H, W), T

def tensor_4D_to_5D(tensor, T):
    """
    Convert a 4D tensor to a 5D tensor by adding a new dimension.

    Args:
        tensor (torch.Tensor): input tensor with shape (N * T, C, H, W)
        T (int): number of time steps

    Returns:
        torch.Tensor: output tensor with shape (N, T, C, H, W)
    """
    B, C, H, W = tensor.shape
    return tensor.reshape(B // T, T, C, H, W)


@MODEL_REGISTRY.register()
class OfflineVideoModel(BaseModel):
    def __init__(self, opt, verbose=True):
        super(OfflineVideoModel, self).__init__(opt, verbose=verbose)

        ### [Model]
        with log_context(lambda: self.logger,
                         start_msg='Building model...',
                         end_msg='Model is built.') as _logger:
            # [build model] 构建模型
            self.net = define_network(opt['network'])
            self.net = self.model_to_device(self.net)

            # [load resume state] 加载模型断点状态信息
            self.load_resume_state(opt)

            # [load pre-trained weights] 加载预训练模型
            self.load_pretrain_weights(self.net, opt)


        ### [Output & Loggers]
        self.logger_init(opt)


        ### [Train Settings]
        if self.is_train:
            self.net.train()

            # [Loss] 损失函数
            self.loss_funcs = []        # 重建损失 reconstruction loss
            for loss_conf in opt['train']['loss']:
                loss_func = get_loss_func(loss_conf).to(self.device)
                self.loss_funcs.append(loss_func)
            self.latent_loss_funcs = {} # 潜空间损失 latent loss
            if 'latent_loss' in opt['train']:
                self.latent_loss_funcs = get_all_loss_func(opt['train']['latent_loss'])
            self.reg_loss_funcs = []    # 正则化损失 regularization loss
            if 'reg_loss' in opt['train']:
                for loss_conf in opt['train']['reg_loss']:
                    loss_func = get_loss_func(loss_conf).to(self.device)
                    self.reg_loss_funcs.append(loss_func)

            # [Optimizer & Schedulers] 优化器 & 学习率调节器
            self.setup_optimizers(self.net)
            self.setup_schedulers()

            self.grad_clip = self._is_grad_clip()
            self.grad_clip_params = self._init_grad_clip_params()
        else:
            self.net.eval()

        ### [Others]
        if self.is_train:
            if hasattr(self, 'amp_scaler'):
                self._backward_func = self._backward_amp
                self._step_optimizer_func = self._step_optimizer_amp
            else:
                self._backward_func = self._backward
                self._step_optimizer_func = self._step_optimizer

            self.loss_acc = []

        self.lqs = None
        self.gts = None
        self.preds = None

        ### [Val Settings]
        self.val_metrics_settings(opt)

    #region ==[Train]==
    #region ==[Default Train & Test]==
    # 相当于训练的一个 iteration
    def optimize_parameters(self, net_kwargs=None, loss_kwargs=None):
        if net_kwargs is None:
            net_kwargs = {}
        if loss_kwargs is None:
            loss_kwargs = {}
        gts = self.gts
        lqs = self.lqs

        _lqs, T = tensor_5D_to_4D(lqs)
        _gts, T = tensor_5D_to_4D(gts)

        if isinstance(self.net, MemoryEnhancedNet):
            ### with memory module, and joint update image model and memory module
            if hasattr(self.net, 'joint_training') and self.net.joint_training:
                # memory enhanced output loss calculation
                with mem_engaged():
                    _lqs, T = tensor_5D_to_4D(lqs)
                    _preds = self.net(_lqs, **net_kwargs)
                    preds = tensor_4D_to_5D(_preds, T)
                loss_img = self._sum_losses(_preds, _gts, **loss_kwargs)

                # regularization loss
                for reg_loss_func in self.reg_loss_funcs:
                    reg_loss = reg_loss_func(self.net)
                    loss_img = loss_img + reg_loss

                # other loss
                while (loss := SharedPool.get('losses').try_pop('loss')) is not None:
                    loss, = loss
                    if torch.tensor(loss.shape).prod().item() > 1:
                        loss = loss.mean()
                    loss_img = loss_img + loss

                # loss backward & update
                self.loss_backward_and_optimize_parameters(loss)

                preds = preds
                loss = loss_img
            ### with memory module, and separately update image model and memory module
            else:
                # vanilla image model output loss backward & update
                if not self.net.memory_trainable_only:  # for fine-tuning
                    with no_mem_engaged():
                        _lqs, T = tensor_5D_to_4D(lqs)
                        _preds = self.net(_lqs, **net_kwargs)
                        preds = tensor_4D_to_5D(_preds, T)
                    loss_img = self._sum_losses(_preds, _gts, **loss_kwargs)
                    self.loss_backward_and_optimize_parameters_image_model(loss_img)

                # memory enhanced output loss calculation
                with mem_engaged():
                    _lqs, T = tensor_5D_to_4D(lqs)
                    _preds = self.net(_lqs, **net_kwargs)
                    preds = tensor_4D_to_5D(_preds, T)
                loss_img = self._sum_losses(_preds, _gts, **loss_kwargs)

                # regularization loss
                for reg_loss_func in self.reg_loss_funcs:
                    reg_loss = reg_loss_func(self.net)
                    loss_img = loss_img + reg_loss

                # other loss
                while (loss := SharedPool.get('losses').try_pop('loss')) is not None:
                    loss, = loss
                    if torch.tensor(loss.shape).prod().item() > 1:
                        loss = loss.mean()
                    loss_img = loss_img + loss

                # loss backward & update
                self.loss_backward_and_optimize_parameters_memory_module(loss)

                if self.net.memory_trainable_only:
                    self._zero_grad('image_model')

                preds = preds
                loss = loss_img
        else:
            ### without memory module
            # [inference] 模型推理
            with no_mem_engaged():
                _lqs, T = tensor_5D_to_4D(lqs)
                _preds = self.net(_lqs, **net_kwargs)
                preds = tensor_4D_to_5D(_preds, T)

            # [backward & update] 反向传播 & 更新参数
            loss = self._sum_losses(_preds, _gts)
            self.loss_backward_and_optimize_parameters(loss)

        self.preds = preds
        self.log_dict = dict(
            loss=loss.item(),
            lr=self.get_current_learning_rate()
        )

        # 输出用于可视化的 scalar 信息
        for key, items in SharedPool.get('scalars').pop_all():
            for i in range(len(items)):
                self.log_dict.update({f"{key}_{i}": items[i]})

    # 相当于测试的一个 iteration
    def test(self, lqs=None, padding_mul=8, net_kwargs=None):
        if net_kwargs is None:
            net_kwargs = {}
        lqs = self.lqs if lqs is None else lqs

        with mem_engaged_if(isinstance(self.net, MemoryEnhancedNet)), \
                torch.inference_mode(), \
                padding(lqs, mul=padding_mul) as (padded_lqs, unpadding), timer() as t:
            _lqs, T = tensor_5D_to_4D(padded_lqs)
            _preds = self.net(_lqs, **net_kwargs)
            preds = tensor_4D_to_5D(_preds, T)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            preds = unpadding(preds)

        self.preds = torch.clamp(preds, 0, 1)
        self.log_dict = dict(
            latency=t.elapsed()
        )
    #endregion

    def feed_data(self, datas, need_gts=True):
        # datas should contains:
        #   lqs: (B, T, C, H, W)
        #   gts: (B, T, C, H, W)
        assert 'lqs' in datas, 'No LQ found in data for feeding data!'
        self.lqs = datas['lqs']['images'].to(self.device)
        if need_gts:
            if 'gts' not in datas:
                raise ValueError('No GT found in data for feeding data! Please check your dataset configuration.')
            self.gts = datas['gts']['images'].to(self.device)

            SharedPool.get("teacher_forcing").clear_and_append("gts", self.gts)

    #region --backward--
    def _backward(self, loss, optimizer=None, **loss_kwargs):
        loss.backward(**loss_kwargs)

    def _backward_amp(self, loss, optimizer=None, **loss_kwargs):
        self.amp_scaler.scale(loss).backward(**loss_kwargs)
        self.amp_scaler.step(optimizer)
    #endregion

    #region --gradient clip--
    def _is_grad_clip(self):
        if isinstance(self.train_opt['grad_clip'], dict):
            grad_clip = self.train_opt['grad_clip']['value']
        elif isinstance(self.train_opt['grad_clip'], bool):
            grad_clip = self.train_opt['grad_clip']
        else:
            grad_clip = False

        return grad_clip

    def _init_grad_clip_params(self):
        if 'grad_clip' not in self.train_opt:
            return None
        # max_norm 指允许的最大梯度范数，保持梯度方向，缩放大小
        params = dict(max_norm=0.01, norm_type=2)
        if isinstance(self.train_opt['grad_clip'], dict):
            params.update(self.train_opt['grad_clip'].get('params', {}))
        return params

    def _try_clip_grad_norm(self, allow_names=('image_model', 'memory_decoder')):
        if not self.grad_clip:
            return
        if allow_names == "all" or not isinstance(self.net, MemoryEnhancedNet):
            nn.utils.clip_grad_norm_(self.net.parameters(), **self.grad_clip_params)
            return
        if 'image_model' in allow_names:
            nn.utils.clip_grad_norm_(self.net.get_memory_parameters(), **self.grad_clip_params)
        if any('memory' in name for name in allow_names):
            nn.utils.clip_grad_norm_(self.net.get_non_memory_parameters(), **self.grad_clip_params)

    def _try_clip_all_grad_norm(self, joint_norm=True):
        if joint_norm:
            # joint_norm == True：梯度剪裁的时候，将两个部分都考虑进去
            self._try_clip_grad_norm("all")
        else:
            # joint_norm == False：梯度剪裁的时候，两个部分单独考虑
            self._try_clip_grad_norm(['image_model', 'memory_decoder'])
    #endregion

    #region --step optimizer--
    @normalize_optimizers
    def _step_optimizer(self, optimizers):
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

    @normalize_optimizers
    def _zero_grad(self, optimizers):
        for optimizer in optimizers:
            optimizer.zero_grad()

    @normalize_optimizers
    def _step_optimizer_amp(self, optimizers):
        for optimizer in optimizers:
            self.amp_scaler.step(optimizer)
            self.amp_scaler.update()
            optimizer.zero_grad()
    #endregion

    #region --backpropagation handler--
    def loss_backward_and_optimize_parameters(self, loss, **loss_kwargs):
        self._backward_func(loss, **loss_kwargs)
        self._try_clip_grad_norm()
        self._step_optimizer_func()
        self.reset_memory()

    def loss_backward_and_optimize_parameters_image_model(self, loss, **loss_kwargs):
        self._backward_func(loss, **loss_kwargs)
        self._try_clip_grad_norm('image_model')
        self._step_optimizer_func('image_model')

    def loss_backward_and_optimize_parameters_memory_module(self, loss, **loss_kwargs):
        self._backward_func(loss, **loss_kwargs)
        self._try_clip_grad_norm('memory_decoder')
        self._step_optimizer_func('memory_decoder')
        self.reset_memory()
    #endregion

    #region ==[Store & Measures]==
    def reset_memory(self):
        if hasattr(self.net, 'reset_memory'):
            self.net.reset_memory()

    def calculate_metrics(self, preds=None, gts=None):
        """
        Args:
            preds (list of Tensor): predicted image tensor with shape (N, C, H, W) or tensor list with length N,  shape (C, H, W).
            gts (list of Tensor): GT image tensor with shape (N, C, H, W) or tensor list with length N,  shape (C, H, W).
        """
        preds = self.preds if preds is None else preds
        gts = self.gts if gts is None else gts

        from basic.utils.console.log import get_root_logger
        metrics_logger = get_root_logger("metrics")
        metrics_logger.info(f"metrics data length: {len(preds)}")

        log_dict = {}
        for metrics_label, metrics_func in self.metric_funcs.items():
            # get the number of parameters of the metrics function
            num_params = len(inspect.signature(metrics_func).parameters)

            # calculate the metrics
            if preds is None:
                raise ValueError('Please run test() or optimize_parameters() first to get preds image.')
            if num_params == 2:
                if gts is None:
                    raise ValueError('Please feed_data() with need_gts=True to get gts image.')
                metrics = metrics_func(preds, gts)
            elif num_params == 1:
                metrics = metrics_func(preds)
            else:
                raise ValueError(
                    f"Metrics function {metrics_func.__name__} should have 1 or 2 parameters, but got {num_params}!")
            log_dict[metrics_label] = metrics.item()

        # self.log_dict.update(log_dict)    # 不更新 self.log_dict，以防出现重复使用
        return log_dict

    #endregion

    def get_current_visuals(self, need_gts=True):
        out_dict = OrderedDict()
        out_dict['lqs'] = self.lqs.detach()[0].float().cpu()
        out_dict['preds'] = self.preds.detach()[0].float().cpu()
        if need_gts:
            if self.gts is None:
                raise ValueError(
                    'No GT found in data for feeding data! Please check your dataset configuration or feed_data() with need_gts=True first.')
            out_dict['gts'] = self.gts.detach()[0].float().cpu()
        return out_dict.copy()

    def get_current_best_metric_log(self, phase='val'):
        """
        Get the best metric result of the specified phase.

        Returns:
            dict: the best metric result of the specified phase. e.g.
            {
                'PSNR': {
                    'value': 27.5,
                    'iter': 1000,
                },
                'SSIM': {
                    'value': 0.85,
                    'iter': 1500,
                },
            }
        """
        self._try_initialize_best_metric_results(phase)
        if phase not in self.best_metric_results.keys():
            self.logger.error(f"Phase {phase} not found in best_metric_results. "
                              f" The best metric model will not be saved. "
                              f" Please check your phase name.")
            return

        return self.best_metric_results[phase]

    # override
    def setup_optimizers(self, net):
        # optimize the unfrozen parameters
        optim_params = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        # print frozen parameters
        if len(optim_params) != len(list(net.parameters())):
            print_frozen_params(net, pre_message_str='Frozen parameters will not be optimized:\n')

        if isinstance(self.opt['train']['optimizer'], list):
            for optim_option in self.opt['train']['optimizer']:
                optim_name = optim_option.pop('name')
                if optim_name == 'image_model' or 'image' in optim_name:
                    if not isinstance(net, MemoryEnhancedNet):
                        self.optimizers[optim_name] = self.get_optimizer(optim_params, optim_option) # e.g. {'encoder_decoder': Adam(...), 'connection': Adam(...)}
                        break
                    _optim_params = net.get_non_memory_parameters()
                elif optim_name == 'memory_decoder' or 'memory' in optim_name:
                    _optim_params = net.get_memory_parameters()
                else:
                    raise ValueError(f"Optimizer name {optim_name} not found in the network!")
                self.optimizers[optim_name] = self.get_optimizer(_optim_params, optim_option) # e.g. {'encoder_decoder': Adam(...), 'connection': Adam(...)}
        else:
            self.optimizers = {0: self.get_optimizer(optim_params)} # e.g. [Adam(...)]


    def setup_schedulers(self):
        """
        Set up schedulers for each optimizer from the options.
        """
        schedulers = {}
        for name, optimizer in self.optimizers.items():
            if name == 0:
                name = None
            scheduler = self.get_scheduler(optimizer, name=name)
            schedulers[name] = scheduler
        self.schedulers = schedulers


    @master_only
    def save(self, epoch, current_iter, name=None, **other_state_dict):
        """
        Save the current [training state & checkpoints].

        Args:
            epoch (int): current epoch
            current_iter (int): current iteration
            name (str): name of the saved state
            other_state_dict (dict): other state to be saved, such as optimizer, scheduler, etc.
        """
        name = name or f'{current_iter}'
        pretrain_model_path = self.save_network(self.net, name, self.net_label)
        training_state_path = self.save_training_state(epoch, current_iter, state_name=name, **other_state_dict)
        return pretrain_model_path, training_state_path

    @master_only
    def try_save_best(self, epoch, current_iter, on_save_best=None, phase='val', **other_state_dict):
        """
        Save the best [training state & checkpoints] according to the validation metrics.
        If the current validation metric is better than the previous best, save the current state as the best state.
        Else, do nothing.

        Args:
            epoch (int): current epoch
            current_iter (int): current iteration
            on_save_best (function): a function to be called when the best state is saved.
            other_state_dict (dict): other state to be saved, such as optimizer, scheduler, etc.
        """
        self._try_initialize_best_metric_results(phase)
        if phase not in self.best_metric_results.keys():
            self.logger.error(f"Phase {phase} not found in best_metric_results. "
                              f" The best metric model will not be saved. "
                              f" Please check your phase name.")
            return

        for metric_label, _ in self.metric_funcs.items():
            value = self.log_dict[metric_label]
            is_best_metrics = self._is_best_metric_result(phase, metric_label, value)
            self._update_best_metric_result(phase, metric_label, value, current_iter)
            if is_best_metrics:
                on_save_best(metric_label, value)
                self.save_network(self.net, f'best_{metric_label}', self.net_label)
                self.save_training_state(epoch, current_iter, state_name=f'best_{metric_label}',
                                         **{metric_label: value}, **other_state_dict)

    def __getattr__(self, item):
        # 如果 model 中不存在该属性，则尝试从 net 中获取
        if not hasattr(self, "net"):
            raise AttributeError(f"No network found in {self.__class__.__name__}!")

        if hasattr(self.net, item):
            return getattr(self.net, item)
        else:
            raise AttributeError(f"No attribute named '{item}' found in {self.__class__.__name__}!")

