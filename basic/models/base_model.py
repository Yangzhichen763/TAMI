import os
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .util import get_scheduler, get_schedulers, get_optimizer, get_optimizers, get_named_scheduler
from basic.utils.general import retry
from basic.utils.misc import get_random_state
from basic.utils.dist import get_dist_info, master_only

from basic.utils.console.log import get_root_logger, get_log_if
from basic.metrics.summary import print_frozen_params


'''
Modified from BasicSR(https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/models/base_model.py)
'''


class BaseModel:
    """
    Base model.
    """

    def __init__(self, opt, verbose=True):
        self.opt = opt
        self.is_train = opt['is_train']
        self.device = opt['device']['train'] if self.is_train else opt['device']['val']
        self.schedulers = {}
        self.optimizers = {}
        self.random_state = None

        self.parts = {}

        ### [Output]
        self.log_dict = OrderedDict()
        self.best_metric_results = OrderedDict()

        self.net_label = 'G'
        if self.is_train:
            self.train_opt = opt['train']

        ### [Distributed]
        self.rank, _ = get_dist_info()
        self.is_master = self.rank <= 0
        self.verbose = verbose
        self.logger = get_log_if(get_root_logger, self.is_master and verbose)

        if self.verbose and not self.is_train:
            self.logger.warning('![Evaluation mode]. If set verbose=True, the testing results directory will be created. '
                                'If in the testing phase, please ignore this warning.')

    def feed_data(self, data):
        pass

    def optimize_parameters(self, **kwargs):
        pass

    #region [Output & Log]
    def get_current_visuals(self):
        pass

    def get_current_log(self):
        """
        Get current log information.

        Returns:
            dict: current log information. e.g.
            training phase: {
                'loss': 0.123,
            }
            validation phase: {
                'latency': 0.100,
                'PSNR': 27.5,
                'SSIM': 0.85,
            }
        """
        return self.log_dict.copy()

    def update_current_log(self, log_dict):
        """
        Set current log information.

        Args:
            log_dict (dict): log information to be set.
        """
        self.log_dict.update(log_dict)

    def get_current_best_metric_log(self):
        """
        Returns:
            dict: the best metric result. e.g.
            {
                'val': {
                    'PSNR': {
                        'value': 27.5,
                        'iter': 1000,
                    },
                    'SSIM': {
                        'value': 0.85,
                        'iter': 1500,
                    },
                },
            }
        """
        return self.best_metric_results.copy()

    def logger_init(self, option):
        from basic.utils.console.log import logger_init_from_config

        if self.is_master and self.verbose:
            # 这里设置 log_info = True 是为了防止在日志文件发生移动后，配置文件的信息丢失，导致复现困难
            logger_init_from_config(option, rename=True, mkdir=self.resume_state is None, log_info=True)
    #endregion

    def save_checkpoint(self, **kwargs):
        """
        Save networks and training state.
        """
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False, rgb2bgr=True, use_image=True):
        """
        Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
            use_image (bool): Whether to use saved images to compute metrics (PSNR, SSIM), if not, then use data directly from network' output. Default: True
        """
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return self.nondist_validation(dataloader, current_iter, tb_logger,
                                           save_img, rgb2bgr, use_image)


    #region [Optimizer] 通过配置文件对模型参数构建一个 optimizer
    def get_optimizer(self, params, option=None) -> torch.optim.Optimizer:
        """
        Get optimizer based on the specified type.
        """
        option = option or self.opt['train']['optimizer']
        return get_optimizer(params, option)

    def get_optimizers(self, params, option=None):
        """
        Get optimizers for each network from the options.
        """
        option = option or self.opt['train']['optimizer']
        return get_optimizers(params, option)

    def setup_optimizers(self, net):
        """
        Set up optimizers for each network from the options.
        """
        # optimize the unfrozen parameters
        optim_params = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        # print frozen parameters
        if len(optim_params) != len(list(net.parameters())):
            print_frozen_params(net, pre_message_str='Frozen parameters will not be optimized:\n')

        if isinstance(self.opt['train']['optimizer'], list):
            self.optimizers = self.get_optimizers(optim_params)  # e.g. {'encoder_decoder': Adam(...), 'connection': Adam(...)}
        else:
            self.optimizers = {0: self.get_optimizer(optim_params)} # e.g. [Adam(...)]

    #region --[Training]--
    def get_optimizers_by_names(self, allow_names=None):
        # 如果没有指定名字，传入的是 None，则返回所有 optimizer
        if allow_names is None:
            return list(self.optimizers.values())

        if allow_names is not None and isinstance(allow_names, str):
            allow_names = [allow_names]

        optimizers = []
        for name, optimizer in self.optimizers.items():
            if name in allow_names:
                optimizers.append(optimizer)

        if optimizers == []:
            logger = get_root_logger()
            if allow_names == []:
                logger.warning('No optimizer provided.')
            else:
                logger.warning(f'No optimizer found for names: {allow_names}')
        return optimizers
    #endregion

    #endregion

    #region [Scheduler] 通过配置文件对每一个 optimizer 创建 scheduler
    def get_scheduler(self, optimizer, name=None, option=None):
        """
        Get scheduler based on the specified type.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler.
        """
        option = option or self.opt['train']['scheduler']
        if name is None:
            return get_scheduler(optimizer, option)
        else:
            return get_named_scheduler(optimizer, option, name)

    def get_schedulers(self, params, option=None):
        """
        Get schedulers for each optimizer from the options.

        Returns:
            dict: The schedulers.
        """
        option = option or self.opt['train']['scheduler']
        return get_schedulers(params, option)

    def setup_schedulers(self):
        """
        Set up schedulers for each optimizer from the options.
        """
        schedulers = {}
        for name, optimizer in self.optimizers.items():
            scheduler = self.get_scheduler(optimizer)
            schedulers[name] = scheduler
        self.schedulers = schedulers
    #endregion

    #region [Model] 模型相关
    # [Device & Distributed] 将模型加载到指定设备上，并包装模型为 DataParallel 或 DistributedDataParallel
    def model_to_device(self, net):
        """
        Model to device. It also warps models with DistributedDataParallel or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        # [Distributed Data Parallel] 分布式数据并行
        if 'dist' in self.opt and self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        # [Data Parallel] 简单的数据并行
        elif 'gpu_ids' in self.opt:
            if self.is_train and len(self.opt['gpu_ids']['train']) > 1:
                net = DataParallel(net)
            elif not self.is_train and len(self.opt['gpu_ids']['val']) > 1:
                net = DataParallel(net)
        return net

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay)

    # 如果使用并行或者其他方式包装模型，使用这个函数可以获取到去除包装之后的模型
    @staticmethod
    def get_bare_model(net):
        """
        Get bare model, especially under wrapping with DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """
        Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = BaseModel.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        self.logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        self.logger.info(net_str)
    #endregion

    #region [Learning Rate] 学习率相关函数
    def _set_lr(self, lr_groups_l):
        """
        Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers.values(), lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """
        Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for _, optimizer in self.optimizers.items():
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    # [similar to scheduler.step()] 相当于 scheduler.step()
    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """
        Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for _, scheduler in self.schedulers.items():
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return {
            key: optimizer.param_groups[0]['lr']
            for key, optimizer in self.optimizers.items()
        }
    #endregion


    #region [Loss] 损失函数相关函数
    def _sum_losses(self, pred, gt, *args, **kwargs):
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        loss = []
        for _pred in pred:
            _loss = self.loss_func(_pred, gt, *args, **kwargs)
            loss.append(_loss)
        loss = torch.stack(loss).sum()
        return loss

    def reduce_loss_dict(self, loss_dict):
        """
        Reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
    #endregion

    #region [Load & Save] 加载和保存模型相关函数
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """
        Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            # missing keys
            logs = []
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logs.append(v)
            if logs:
                logs = list(set([
                    log[:log.find('.')] for log in logs
                ]))
                log = ', '.join(logs)
                self.logger.warning(
                    f"Missing keys in the loaded state_dict (current net - loaded net): {log}"
                )

            # extra keys
            logs = []
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logs.append(v)
            if logs:
                logs = list(set([
                    log[:log.find('.')] for log in logs
                ]))
                log = ', '.join(logs)
                self.logger.warning(
                    f"Redundant keys in the loaded state_dict (Loaded net - current net): {log}"
                )

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    self.logger.warning(f'Size different, ignore [{k}]: crt_net:{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """
        Load network.

        Args:
            load_path (str, LiteralString): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        if not os.path.exists(load_path):
            self.logger.warning(f'Path {load_path} does not exist. Pretrained model will not be loaded.')
            return False

        if not load_path.endswith('.pth'):
            self.logger.warning(f'The loaded path should end with ".pth", but got {load_path}.')
            return False

        net = self.get_bare_model(net)
        device = f'{self.device}:{torch.cuda.current_device()}'

        # load checkpoint
        state_dict = torch.load(load_path, map_location=lambda storage, loc: storage) # .to(device)

        # load params from state_dict by param_key
        if param_key is not None:
            if param_key not in state_dict:
                for prefeb_params_key in ['params', 'state_dict']:
                    if prefeb_params_key in state_dict:
                        self.logger.info(f'Loading: {param_key} does not exist, use {prefeb_params_key}.')
                        param_key = prefeb_params_key
            if param_key in state_dict:
                state_dict = state_dict[param_key]
                self.logger.info(f'Loading {net.__class__.__name__} model from {load_path} at {device}, with param key: [{param_key}].')
            else:
                self.logger.warning(f'Cannot find {param_key} in the loaded state_dict. Use the original state_dict.')
        else:
            self.logger.info(f'Loading {net.__class__.__name__} model from {load_path} at {device}, without param key.')

        # remove unnecessary 'module.'
        for k, v in deepcopy(state_dict).items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
                state_dict.pop(k)
        self._print_different_keys_loading(net, state_dict, strict)

        # only load common keys
        net_dict = net.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items()
            if k in net_dict
        }
        net_dict.update(state_dict)

        # load state_dict to net
        missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=strict)
        # self.logger.info(f'Pretrained model loaded. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.')
        return True

    def resume_training(self, resume_state):
        """
        Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), f'Wrong lengths of optimizers, expect {len(self.optimizers)} in the config, got {len(resume_optimizers)} in the resume_state'
        assert len(resume_schedulers) == len(self.schedulers), f'Wrong lengths of schedulers, expect {len(self.schedulers)} in the config, got {len(resume_schedulers)} in the resume_state'
        for name, o in resume_optimizers.items():
            self.optimizers[name].load_state_dict(o)
        for name, s in resume_schedulers.items():
            self.schedulers[name].load_state_dict(s)
        if 'best_metric_results' in resume_state:
            self.best_metric_results = resume_state['best_metric_results']
        if 'random_state' in resume_state:
            self.random_state = resume_state['random_state']

        # resume amp scaler
        if self.opt['is_train'] and self.opt.get('use_amp', False):
            if resume_state.get('amp_scaler', None):
                self.amp_scaler.load_state_dict(resume_state['amp_scaler'])


    @staticmethod
    def get_checkpoint_path(opt, net_name, net_label='G'):
        """
        Get the checkpoint path.

        Args:
            net_name (str): Network name.
            net_label (str): Network label.
        """
        if net_label is None:
            filename = f'{net_name}.pth'
        else:
            filename = f'{net_name}_{net_label}.pth'
        import os
        save_path = os.path.join(opt['path']['models'], filename)
        return save_path


    @staticmethod
    def check_checkpoint(opt, net_name, net_label='G'):
        """
        Check if the checkpoint exists.

        Args:
            net_name (str): Network name.
            net_label (str): Network label.
        """
        save_path = BaseModel.get_checkpoint_path(opt, net_name, net_label)
        import os.path as osp
        return osp.exists(save_path)


    @staticmethod
    def remove_checkpoint(opt, net_name, net_label='G'):
        """
        Remove the checkpoint.

        Args:
            net_name (str): Network name.
            net_label (str): Network label.
        """
        save_path = BaseModel.get_checkpoint_path(opt, net_name, net_label)
        return os.remove(save_path)


    @master_only
    def save_network(self, net, net_name, net_label='G', param_key='params'):
        """
        Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_name (str): Network name.
            net_label (str): Network label.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        save_path = self.get_checkpoint_path(self.opt, net_name, net_label)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        def on_failure():
            self.logger.warning(f"Failed to save checkpoints to {save_path}. Just skip it.")
        with retry(max_retries=5, task_name="Saving checkpoints", on_failure=on_failure) as signal:
            torch.save(save_dict, save_path)
            signal.success = True
        return save_path

    @master_only
    def save_training_state(self, epoch, current_iter, state_name=None, **other_state_dict):
        """
        Save training states during training, which will be used for resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
            state_name (str): The name of the state. If set to None, use the
                current iteration as the name. Default: None.
            **other_state_dict (dict): Other arguments to be saved.
        """
        if current_iter < 0:
            return

        state = {
            'epoch': epoch,
            'iter': current_iter,
            'optimizers': {},
            'schedulers': {},
            'best_metric_results': self.best_metric_results,
            'random_state': get_random_state()
        }
        if other_state_dict:
            for k, v in other_state_dict.items():
                state[k] = v
        for name, o in self.optimizers.items():
            state['optimizers'][name] = o.state_dict()
        for name, s in self.schedulers.items():
            state['schedulers'][name] = s.state_dict()

        # avoid occasional writing errors
        state_name = state_name or f"{current_iter}"
        save_filename = f'{state_name}.state'
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        def on_failure():
            self.logger.warning(f"Failed to save training state to {save_path}. Just skip it.")
        with retry(max_retries=5, task_name="Saving training states", on_failure=on_failure) as signal:
            torch.save(state, save_path)
            signal.success = True
        return save_path

    @master_only
    def save_latest_training_state(self, epoch, current_iter, **other_state_dict):
        """
        Save latest training state during training, which will be used for resuming
        """
        self.save_training_state(epoch, current_iter, state_name='latest', **other_state_dict)

    #region --[Best Metric Results] 最佳指标相关函数
    @staticmethod
    def _higher_or_lower(better):
        if better in ['higher', '↑', 'high', 'greater', '>']:
            return True
        elif better in ['lower', '↓', 'low', 'less', '<']:
            return False
        else:
            raise ValueError(f'Better type "{better}" is not supported.')

    def _try_initialize_best_metric_results(self, dataset_name):
        """
        Initialize the best metric results dict for recording the best metric value and iteration.

        self.opt must contain:
            - val:
                - metrics:
                    - PSNR(or other metric name):
                        - better: 'higher' or 'lower'
        """
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric_opt in self.opt['val']['metrics']:
            better = metric_opt.get('better', 'higher')
            init_value = float('-inf') if self._higher_or_lower(better) else float('inf')
            record[metric_opt['type']] = dict(better=better, value=init_value, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _is_best_metric_result(self, dataset_name, metric, value):
        """
        Check if the current metric value is the best one.

        Args:
            dataset_name (str): Dataset name.
            metric (str): Metric name.
            value (float): Current metric value.

        Returns:
            bool: Whether the current metric value is the best one.
        """
        metrics = self.best_metric_results[dataset_name][metric]
        if 'value' not in metrics:
            return True
        if self._higher_or_lower(metrics['better']):
            return value >= metrics['value']
        else:
            return value <= metrics['value']

    def _update_best_metric_result(self, dataset_name, metric, value, current_iter):
        metrics = self.best_metric_results[dataset_name][metric]
        if self._is_best_metric_result(dataset_name, metric, value):
            metrics['value'] = value
            metrics['iter'] = current_iter

    def get_current_best_metric_log_str(self, dataset_name, logs: dict=None):
        # 这个 logs 的 values 必须包含 value 和 iter 两个 key
        logs = logs if logs is not None else self.best_metric_results[dataset_name]
        metric_log_str = ', '.join([f"{metric}: {pair['value']}" for metric, pair in logs.items()])
        return metric_log_str
    #endregion

    def load_resume_state(self, option):
        # [load resume state] 加载模型断点状态信息
        import os.path as osp
        import basic.options as options

        if self.is_train:
            if option['path'].get('resume_state', None):
                if osp.exists(option['path']['resume_state']):
                    self.logger.info(f"Loading resume state from {option['path']['resume_state']}...")

                    device_id = torch.cuda.current_device()
                    resume_state = torch.load(option['path']['resume_state'],
                                              map_location=lambda storage, loc: storage.cuda(device_id))
                    options.check_resume(option, self.net_label, resume_state['iter'])

                    self.resume_state = resume_state
                else:
                    self.resume_state = None
                    self.logger.warning(
                        f"Resume state {option['path']['resume_state']} not found, start training from scratch.")
            else:
                self.resume_state = None
                self.logger.warning(f"Start training from scratch.")
        else:
            self.resume_state = None

    def load_pretrain_weights(self, net, option, on_test=None, load_network_kwargs=None):
        # [load pre-trained weights] 加载预训练模型
        import os.path as osp
        from basic.utils.console.log import ColorPrefeb as CP

        if load_network_kwargs is None:
            load_network_kwargs = {}

        path_opt = option['path']
        if 'pretrain_model_G' in path_opt:
            pretrain_model_opt = path_opt['pretrain_model_G']
            strict_load = path_opt['strict_load'] if 'strict_load' in path_opt else True
            if isinstance(pretrain_model_opt, str):
                load_path = osp.join(path_opt['root'], pretrain_model_opt)
                if self.load_network(net, load_path, strict_load, **load_network_kwargs):
                    self.logger.info(f"Pre-trained model loaded from {CP.path(load_path)}.")
            elif isinstance(pretrain_model_opt, dict):
                if self.is_train:
                    if 'train' in pretrain_model_opt:
                        load_path = osp.join(path_opt['root'], pretrain_model_opt['train'])
                        if self.load_network(net, load_path, strict_load, **load_network_kwargs):
                            self.logger.info(f"(Train) Pre-trained model loaded from {CP.path(load_path)}.")
                    else:
                        self.logger.info(f"(Train) Train from scratch")
                if not self.is_train:
                    if 'test' in pretrain_model_opt:
                        load_path = osp.join(path_opt['root'], pretrain_model_opt['test'])
                        if 'debug' in option:
                            strict_load = not option['debug']
                        else:
                            strict_load = True
                        if self.load_network(net, load_path, strict_load, **load_network_kwargs):
                            self.logger.info(f"(Test) Pre-trained model loaded from {CP.path(load_path)}.")
                    else:
                        self.logger.warning(f"(Test) No pre-trained model for test found in options.")
            else:
                raise ValueError(f"Invalid pretrain_model_G: {option['path']['pretrain_model_G']}")
        elif not self.is_train:  # 如果是测试阶段，并且没有指定预训练模型，则报错
            if on_test:
                on_test()
            else:
                raise ValueError(f"No `pretrain_model_G` found in `path`. Please specify `pretrain_model_G` in the configuration.")
        else:
            self.logger.info(f"Train from scratch")

    def val_metrics_settings(self, option):
        from basic.metrics import get_metrics_calculator

        if 'val' in option and 'metrics' in option['val']:
            metrics_opt_list = self.opt['val']['metrics']
            self.metric_funcs = dict()
            for metrics_opt in metrics_opt_list:
                self.metric_funcs[metrics_opt['type']] = get_metrics_calculator(metrics_opt)
        else:
            self.metric_funcs = dict()
    #endregion