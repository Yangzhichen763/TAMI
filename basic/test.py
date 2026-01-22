import os

import sys

sys.path.append('.')  # 为了在某些服务器中能够导入 basic 包

from basic.utils.io import save_image
from basic.utils.shared_pool import SharedPool


def load_datasets(opt):
    from basic.datasets import create_dataset, create_dataloader
    # [test data] 加载测试数据集
    test_opt = opt['datasets']['test']
    dataset = create_dataset(test_opt)
    dataloader = create_dataloader(dataset, test_opt)
    return dataloader


def save_output(out_dict, output_dir, file_name, **save_opts):
    from basic.utils.convert import tensor2numpy
    # [save pred image]
    if 'pred' in out_dict:
        pred = out_dict['pred']
        pred = tensor2numpy(pred)
        # pred = transforms.ToPILImage()(pred.squeeze(0))  # 原代码是这样处理的（没有使用 (image * 255).round() 而是直接向下取整）
        pred_file_path = os.path.join(output_dir, 'pred', f'{file_name}.png')
        save_image(image=pred, path=pred_file_path)

    # [save other image]
    for key, value in out_dict.items():
        if  f'save_{key}' in save_opts and save_opts[f'save_{key}']:
            value = tensor2numpy(value)
            file_path = os.path.join(output_dir, key, f'{file_name}.png')
            save_image(image=value, path=file_path)


#region ==[test]==
from basic.utils.general import future_func
@future_func
def test_video(
        name, need_gt=False, save_opts=None
):
    pass


from basic.utils.general import future_func
@future_func
def test_image(
        name, need_gt=False, save_opts=None
):
    pass
#endregion


#region ==[val]==
from basic.utils.general import future_func
@future_func
def val_pre(
        name, val_func, save_opts=None, loop=False, **val_kwargs
):
    pass


from basic.utils.general import future_func
@future_func
def val_video_online(
        opt, dataloader, iteration,
        logger=None, writer=None,
        output_dir: str = None, save_opts=None,
        show_model_logs=False,
        **val_kwargs,
):
    pass

from basic.utils.general import future_func
@future_func
def val_video_offline(
        opt, dataloader, iteration,
        logger=None, writer=None,
        output_dir: str = None, save_opts=None,
        show_model_logs=False,
        **val_kwargs,
):
    pass

from basic.utils.general import future_func
@future_func
def val_image(
        opt, dataloader, iteration,
        logger=None, writer=None,
        output_dir: str = None, save_opts=None,
        show_model_logs=False,
        **val_kwargs,
):
    pass
#endregion


import sys
sys.path.append('.')  # 为了在某些服务器中能够导入 basic 包


if __name__ == '__main__':
    # additional args:
    # --gamma:
    test_video("Video-Net", need_gt=True, save_opts=dict(save_gt=False, save_lq=False))

