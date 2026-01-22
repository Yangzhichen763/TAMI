from copy import deepcopy

import sys
import math

sys.path.append('.')  # 为了在某些服务器中能够导入 basic 包

from basic.utils.misc import set_random_state
from basic.utils.console.log import ColorPrefeb as CP, get_root_logger


#region ==[Preliminary]==
def load_datasets(opt, args, logger=None):
    from basic.datasets import create_dataset, create_dataloader, create_sampler
    logger = logger or get_root_logger()

    ### train dataset
    if 'train' in opt['datasets']:
        train_opt = opt['train']
        train_dataset_opt = opt['datasets']['train']
        # [dataset]
        train_dataset = create_dataset(train_dataset_opt)
        # [dataloader]
        train_loader = create_dataloader(train_dataset, train_dataset_opt, seed=opt['manual_seed'])
        # [training parameters]
        train_size = len(train_loader)
        if 'n_epoch' in train_opt:
            n_epoch = train_opt['n_epoch']
            n_iter = 0
            batch_size = train_dataset_opt['batch_size_per_gpu']

            if 'sampler' in train_dataset_opt:
                sampler_option = train_dataset_opt['sampler']
                sampler = create_sampler(train_dataset, sampler_option, seed=opt['manual_seed'])
                for epoch in range(n_epoch):
                    for _ in sampler:
                        n_iter += 1
                n_iter = n_iter // batch_size
            else:
                n_iter = train_size * n_epoch
            logger.info(f'Total epochs: {n_epoch} ({n_iter} iters)')
        elif 'n_iter' in train_opt:
            n_iter = train_opt['n_iter']
            n_epoch = 0
            batch_size = train_dataset_opt['batch_size_per_gpu']

            if 'sampler' in train_dataset_opt:
                sampler_option = train_dataset_opt['sampler']
                sampler = create_sampler(train_dataset, sampler_option, seed=opt['manual_seed'])
                _n_iter = 0
                while True:
                    for _ in sampler:
                        _n_iter += 1
                    n_epoch += 1
                    if _n_iter // batch_size >= n_iter:
                        break
            else:
                n_epoch = (n_iter - 1) // train_size + 1
            logger.info(f'Total iters: {n_iter} iters ({n_iter // train_size} epochs + {n_iter % train_size} iters)')
        else:
            raise ValueError(f'No "n_epoch" or "n_iter" is provided in the configuration file at {CP.path(args.opt)}#train.')

        if 'warmup_iter' in train_opt:
            warmup_iter = train_opt['warmup_iter']
        elif 'warmup_epoch' in train_opt:
            warmup_iter = train_size * train_opt['warmup_epoch']
        else:
            warmup_iter = 0
    else:
        raise ValueError(
            f'No training dataset is provided in the configuration file at {CP.path(args.opt)}#datasets.train.')

    ### val dataset
    if 'val' in opt['datasets']:
        val_dataset_opt = opt['datasets']['val']
        # [dataset]
        val_dataset = create_dataset(val_dataset_opt)
        # [dataloader]
        val_loader = create_dataloader(val_dataset, val_dataset_opt)
    else:
        val_dataset = None
        val_loader = None
        logger.warning(f'No validation dataset is provided in the configuration file at {CP.path(args.opt)}#datasets.val. '
                       f'The model will not be validated during training.')

    return dict(
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_dataset=val_dataset,
        val_loader=val_loader,
        n_epoch=n_epoch,
        n_iter=n_iter,
        warmup_iter=warmup_iter,
        use_iter=train_opt.get('n_iter', False),
    )


# @process_parallel()
# def _val(val_func, *args, **kwargs):
#     return val_func(*args, **kwargs)


def check_for_validation_and_save_best(
        opt, model, val_loader, current_step, n_iter, epoch,
        logger=None, writer=None, metrics_logger=None,
        val_func=None, save_outputs=True, any_save_best_metrics=True, any_log_best_metrics=True,

        save_val_model_only=False
):
    logger = logger or get_root_logger()
    metrics_logger = metrics_logger or get_root_logger('metrics')

    if val_func is None:
        logger.warning(f'No validation function is provided. The model will not be validated during training.')
        return

    ### validation
    if (current_step >= n_iter
        or ('val' in opt['datasets'] and
            (
                # steps
                (
                    'val_frequency' in opt['train'] and current_step % opt['train']['val_frequency'] == 0   # 验证步数频率到了就验证
                     and ('val_start' not in opt['val'] or current_step >= opt['val']['val_start'])         # 超过验证起始步数才验证
                ) or
                # epochs
                (
                    'val_last_epoch' in opt['train'] and epoch != opt['train']['val_last_epoch']            # 上次验证的 epoch 不是当前 epoch
                     and 'val_frequency_epoch' in opt['train'] and epoch % opt['train']['val_frequency_epoch'] == 0 # 验证 epoch 频率到了就验证
                     and ('val_start_epoch' not in opt['val'] or current_step < opt['val']['val_start_epoch'])      # 超过验证起始 epoch 才验证
                )
            )
        )
    ):
        # temporarily save the current model and load for validation
        save_path, _ = model.save(epoch, current_step, name='val')

        if not save_val_model_only:
            opt_val = deepcopy(opt)
            opt_val['path']['pretrain_model_G'] = save_path
            opt_val['is_train'] = False

            # validate
            output_dir = opt_val['path']['val_images'] if save_outputs else None
            logs = val_func(opt_val, val_loader, current_step, output_dir=output_dir, logger=logger, writer=writer)

            if any_save_best_metrics:
                # update best metrics
                model.update_current_log(logs)

                # save best model
                def on_save_best(metric_name, metric_value):
                    logger.info(
                        f'The best model [{CP.keyword(metric_name)}: {CP.number(metric_value)}] is saved at epoch {epoch}... (step {current_step})')
                model.try_save_best(epoch, current_step, on_save_best)  # 同时也更新了 best_metric_result

                if any_log_best_metrics:
                    # log best metric
                    log_best_metrics(caption=f'step {current_step}', model=model, metrics_logger=metrics_logger)
            else:
                if any_log_best_metrics:
                    metric_log_str = ', '.join([f"{metric}: {value}" for metric, value in logs.items()])
                    log_str = f"step {current_step}, {metric_log_str}"

                    logger.info(log_str)
                    metrics_logger.info(log_str)

    opt['train']['val_last_epoch'] = epoch


def log_best_metrics(caption, model, logger=None, metrics_logger=None, logs=None):
    logger = logger or get_root_logger()
    metrics_logger = metrics_logger or get_root_logger('metrics')

    best_metric_log_str = model.get_current_best_metric_log_str('val', logs)
    log_str = f"{caption}, {best_metric_log_str}"

    logger.info(log_str)
    metrics_logger.info(log_str)


def check_for_save_log(opt, model, current_step, n_iter, metrics_logger=None):
    from basic.metrics.summary import Stats
    metrics_logger = metrics_logger or get_root_logger('metrics')

    ### save log
    if (current_step >= n_iter
            or current_step % opt['logger']['print_frequency'] == 0
    ):
        log = model.get_current_log()
        stats = Stats(f'step {current_step}')
        stats.append(header=current_step, **log)
        metrics_logger.info(stats.summary())


def check_for_save_checkpoint(opt, model, current_step, n_iter, epoch, logger=None):
    logger = logger or get_root_logger()

    ### save step: 保存模型与其 training states，每次都会保存一个 checkpoint，不会被覆盖
    if (current_step >= n_iter
            or (
                    'save_checkpoint_frequency' in opt['logger'] and current_step % opt['logger']['save_checkpoint_frequency'] == 0
            )
            or (
                    'save_checkpoint_last_epoch' in opt['train'] and epoch != opt['train']['save_checkpoint_last_epoch']
                    and 'save_checkpoint_frequency_epoch' in opt['logger'] and epoch % opt['logger']['save_checkpoint_frequency_epoch'] == 0
            )
    ):
        logger.info(f'Saving models and training states at epoch {epoch}... (step {current_step})')
        model.save(epoch, current_step)

    ### save latest: 保存 latest 模型与其 training states，每次都会覆盖上一次保存的 latest checkpoint
    if (current_step >= n_iter
            or ('save_latest_checkpoint_frequency' in opt['logger'] and current_step % opt['logger']['save_latest_checkpoint_frequency'] == 0)
            or ('save_checkpoint_last_epoch' in opt['train'] and epoch != opt['train']['save_checkpoint_last_epoch'] and
                'save_latest_checkpoint_frequency_epoch' in opt['logger'] and epoch % opt['logger']['save_latest_checkpoint_frequency_epoch'] == 0)
    ):
        model.save(epoch, current_step, name='latest')

    opt['train']['save_checkpoint_last_epoch'] = epoch


def save_final_checkpoint(model, current_step, n_iter, epoch, logger=None):
    logger = logger or get_root_logger()

    ### save final: 保存 final 模型权重与其 training states
    if current_step >= n_iter:
        model.save(epoch, current_step, name='final')
        logger.info(f"Final checkpoint is saved at epoch {epoch}... (step {current_step})")
        logger.info(f"Training is done.")


class TrainIterData:
    def __init__(self, model, is_train):
        self.model = model
        self.is_train = is_train
#endregion


#region ==[train]==
from basic.utils.general import future_func
@future_func
def train_video_online(
        name, val,
        summary_input_size=(3, 256, 256), save_outputs=True,
        add_log_to_writer=True,
):
    pass

from basic.utils.general import future_func
@future_func
def train_video_offline(
        name, val,
        summary_input_size=(2, 3, 256, 256), save_outputs=True,
        add_log_to_writer=True,
):
    pass

from basic.utils.general import future_func
@future_func
def train_image(
        name, val,
        summary_input_size=(3, 256, 256), save_outputs=True,
        add_log_to_writer=True,
):
    pass

#endregion

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Video-Net')
    parser.add_argument('--hyperparams', '-hp',         type=lambda s: s.split(','),   default=[],
                        help='Custom hyperparameters for training, e.g. --hyperparams=offline,custom')
    args = parser.parse_known_args()[0]

    from basic.test import val_video_offline, val_video_online
    if 'offline' in args.hyperparams:
        train_video_offline("Video-Net", val_video_offline, save_outputs=True, add_log_to_writer=True)
    if 'online' in args.hyperparams:
        train_video_online("Video-Net", val_video_online, save_outputs=True, add_log_to_writer=True)

