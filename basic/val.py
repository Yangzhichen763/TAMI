import sys

sys.path.append('..')
sys.path.append('.')

from test import val_pre, val_video_online


if __name__ == '__main__':
    """
    对模型权重路径下的 val.pth 进行验证一次
    """
    val_pre("Video-Net", val_video_online, save_opts=dict(save_gt=False, save_lq=False), show_model_logs=True)

