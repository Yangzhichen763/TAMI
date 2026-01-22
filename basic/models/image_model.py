import torch
import torch.nn as nn
from collections import OrderedDict
import inspect

from basic.models.base_model import BaseModel
from basic.metrics import timer
from basic.archs import define_network
from basic.losses import get_loss_func
from basic.utils.dist import master_only
from basic.utils.convert import padding

from basic.utils.console.log import log_context


class ImageModel(BaseModel):
    def __init__(self, opt, verbose=True):
        super().__init__(opt, verbose=verbose)

        ### [Model]
        with log_context(lambda: self.logger,
                         start_msg='Building model...',
                         end_msg='Model built.') as _logger:
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
            self.loss_funcs = []
            for loss_conf in opt['train']['loss']:
                loss_func = get_loss_func(loss_conf).to(self.device)
                self.loss_funcs.append(loss_func)

            # [Optimizer & Schedulers] 优化器 & 学习率调节器
            super().setup_optimizers(self.net)
            super().setup_schedulers()
            self.optimizer = self.optimizers[0]

            self.grad_clip = self._is_grad_clip()
            self.grad_clip_params = self._init_grad_clip_params()
        else:
            self.net.eval()

        self.lq = None
        self.gt = None
        self.pred = None


        ### [Val Settings]
        self.val_metrics_settings(opt)

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

    def _try_clip_grad_norm(self):
        if not self.grad_clip:
            return

        nn.utils.clip_grad_norm_(self.net.parameters(), **self.grad_clip_params)
    #endregion

    def feed_data(self, data, need_gt=True):
        self.lq = data['lq']['image'].to(self.device)
        if need_gt:
            if 'gt' not in data:
                raise ValueError('No GT found in data for feeding data!')
            self.gt = data['gt']['image'].to(self.device)

    # 相当于训练的一个 iteration
    def optimize_parameters(self, **kwargs):
        self.optimizer.zero_grad()
        self.pred = self.net(self.lq)

        loss = self.loss_func(self.pred, self.gt)

        self._try_clip_grad_norm()
        loss.backward()
        self.optimizer.step()

        self.log_dict = dict(loss=loss.item())

    # 相当于测试的一个 iteration
    def test(self, mul=8, **kwargs):
        self.net.eval()
        with torch.inference_mode(), padding(self.lq, mul=mul) as padded_image, timer() as t:
            pred = self.net(padded_image)
            self.pred = torch.clamp(pred, 0, 1)

        self.log_dict = dict(
            latency=t.elapsed()
        )

    def calculate_metrics(self, pred=None, gt=None):
        """
        Args:
            pred (list of Tensor): predicted image tensor with shape (C, H, W).
            gt (list of Tensor): GT image tensor with shape (C, H, W).
        """
        pred = self.pred if pred is None else pred
        gt = self.gt if gt is None else gt

        log_dict = {}
        for metrics_label, metrics_func in self.metric_funcs.items():
            num_params = len(inspect.signature(metrics_func).parameters)
            if pred is None:
                raise ValueError('Please run test() or optimize_parameters() first to get pred image.')


            if num_params == 2:
                if gt is None:
                    raise ValueError('Please feed_data() with need_gt=True to get gt image.')
                metrics = metrics_func(pred, gt)

            elif num_params == 1:
                metrics = metrics_func(pred)

            else:
                raise ValueError(f"Metrics function {metrics_func.__name__} should have 1 or 2 parameters")

            log_dict[metrics_label] = metrics.item()

        self.log_dict.update(log_dict)
        return log_dict

    def get_current_visuals(self, need_gt=True):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach()[0].float().cpu()
        out_dict['pred'] = self.pred.detach()[0].float().cpu()
        if need_gt:
            if self.gt is None:
                raise ValueError('No GT found in data for feeding data! Please check your dataset configuration or feed_data() with need_gt=True first.')
            out_dict['gt'] = self.gt.detach()[0].float().cpu()
        return out_dict

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
