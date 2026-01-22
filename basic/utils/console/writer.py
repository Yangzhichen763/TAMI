
from contextlib import contextmanager

from torch.utils.tensorboard import SummaryWriter



# 用来包装 writer 的日志打印器，使得在分布式计算或其他状况下时，不会报错（也不会显示），writer 还可以手动关闭（不会引起大批量代码的修改）
# noinspection SpellCheckingInspection
class Writer:
    def __init__(self, writer=None, **kwargs):
        if not kwargs.pop('init', True):
            self.writer = None
        else:
            self.writer = writer

    def __getattr__(self, name):
        if self.writer is not None:
            if hasattr(self.writer, name):
                def method(*args, **kwargs):
                    return getattr(self.writer, name)(*args, **kwargs)
                return method
            raise AttributeError(f"'Logger' object has no attribute '{name}'")
        else:
            return self.do_nothing

    def do_nothing(self, *args, **kwargs):
        pass

    @staticmethod
    def get_empty_logger(logger=None):
        return Writer(logger, init=False)


@contextmanager
def writer_context(writer_factory, show_if=True, **kwargs) -> Writer:
    """

    Args:
        writer_factory: A function that returns a writer object.
        show_if (bool): Whether to show the log messages.
    """
    _logger = get_writer_if(writer_factory, condition=show_if, **kwargs)
    try:
        yield _logger
    except Exception as e:
        _logger.exception(e)
        raise e


def get_writer_if(writer_factory, condition=True, **kwargs) -> Writer:
    _writer = Writer(writer_factory(), **kwargs) if condition else Writer.get_empty_logger()
    return _writer


def writer_init_from_config(conf, comment_prefix="train"):
    from basic.utils.console.log import get_striped_time_str
    import os
    import os.path as osp

    # 寻找文件夹 path_opt['tensorboard_log'] 下的最新日志文件
    log_dir = conf['path']['tensorboard_log']
    log_file = f"{comment_prefix}_{get_striped_time_str()}"
    if osp.exists(log_dir):
        for file in os.listdir(log_dir):
            log_file = file
            break
    writer = SummaryWriter(log_dir=f"{log_dir}/{log_file}")
    return writer


#region ==[Packaging]==
def add_param_grad_histogram(writer, model, epoch):
    # 如果会报 np.greater 的错误，那就是 numpy 和 tensorboard 版本不对应的问题，
    # 如果使用的 tensorboard 版本是 2.14.0，那么 numpy 版本可以使用 1.23.5
    for name, param in model.named_parameters():
        writer.add_histogram(f'params/{name}', param.data.cpu().numpy(), epoch)
        if param.grad is not None:
            writer.add_histogram(f'grads/{name}', param.grad.data.cpu().numpy(), epoch)
#endregion