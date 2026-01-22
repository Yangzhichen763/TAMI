import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import timm
from torchvision import transforms
import math

import sys
sys.path.append('.')
sys.path.append('..')
from basic.utils.io import read_image_as_pil, glob_single_files
from basic.utils.convert import padding
try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass

try:
    from basic.utils.console.log import get_root_logger
    logger = get_root_logger(force_set_info=True)
    def print(*args, **kwargs):
        logger.info(*args, **kwargs)
except ImportError:
    from builtins import print as original_print
    def print(*args, **kwargs):
        original_print(*args, **kwargs)


def build_transform(img_size=None):
    if img_size is None:
        return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.485, 0.456, 0.406),
            #     std=(0.229, 0.224, 0.225)
            # ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),  # 保证稍大于 patch grid
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.485, 0.456, 0.406),
            #     std=(0.229, 0.224, 0.225)
            # ),
        ])


def build_dinov3_model(model_name="vit_small_patch16_dinov3.lvd1689m"):
    """
    可用的 DINOv3 模型名:
        convnext_base.dinov3_lvd1689m, convnext_large.dinov3_lvd1689m, convnext_small.dinov3_lvd1689m,
        convnext_tiny.dinov3_lvd1689m, vit_7b_patch16_dinov3.lvd1689m, vit_7b_patch16_dinov3.sat493m,
        vit_base_patch16_dinov3.lvd1689m, vit_base_patch16_dinov3_qkvb.lvd1689m,
        vit_huge_plus_patch16_dinov3.lvd1689m, vit_huge_plus_patch16_dinov3_qkvb.lvd1689m,
        vit_large_patch16_dinov3.lvd1689m, vit_large_patch16_dinov3.sat493m,
        vit_large_patch16_dinov3_qkvb.lvd1689m, vit_large_patch16_dinov3_qkvb.sat493m,
        vit_small_patch16_dinov3.lvd1689m, vit_small_patch16_dinov3_qkvb.lvd1689m,
        vit_small_plus_patch16_dinov3.lvd1689m, vit_small_plus_patch16_dinov3_qkvb.lvd1689m
    """
    # 使用 python -c "import timm; print([m for m in timm.list_models(pretrained=True) if 'dinov3' in m])" 指令查询可用的 DINOv3 模型名
    model_local_path = os.path.join("./experiments/pretrained_weights/hub", model_name)
    if not os.path.exists(model_local_path) or not os.listdir(model_local_path):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        print(f"Model {model_name} {CP.warning('not found')} in cache, downloading to {model_local_path}")
        os.makedirs(model_local_path, exist_ok=True)
        model = timm.create_model(model_name, pretrained=True, cache_dir=model_local_path)
    else:
        # 如果出现了下载重试问题，可能是因为版本过老的问题，pip install --upgrade huggingface-hub 即可
        os.environ["HF_HUB_OFFLINE"] = "1"

        print(f"Model {model_name} {CP.success('found')} in cache, loading from {model_local_path}")
        model = timm.create_model(model_local_path, pretrained=True, cache_dir=model_local_path)
    return model


def forward_dinov3_model(model, img_tensor, to_2d=False):
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # If tensor has more than 4 dims (e.g. [B, T, C, H, W]), flatten extra dims into batch
    orig_shape = img_tensor.shape
    if img_tensor.dim() > 4:
        # e.g. (B, T, C, H, W) -> (B*T, C, H, W)
        merge_dim = int(torch.prod(torch.tensor(orig_shape[:-3])))
        img_tensor = img_tensor.view(merge_dim, *orig_shape[-3:])

    with torch.no_grad(), padding(img_tensor, 16) as (padded_img_tensor, _):
        # 对于 ViT 模型，通常用 model.forward_features 或直接 model.forward 然后拿特征层
        # timm 新版本自动管理 “features only” 模型，通常 model.forward_features 返回 feature maps
        feats = model.forward_features(padded_img_tensor)  # (B, N, 384), where N = 1(cls) + 4(register) + H * W / (16*16)

    if to_2d:
        from einops import rearrange
        h, w = math.ceil(orig_shape[-2] / 16), math.ceil(orig_shape[-1] / 16)
        feats = feats[:, 1+4:, ...]  # remove cls and reg tokens
        feats = rearrange(feats, 'b (h w) c -> b c h w', h=h, w=w)

    # Restore original batch-like dimensions if needed
    if img_tensor.dim() > 4 or len(orig_shape) > 4:
        # feats shape is (B', ...) → reshape back to (orig_shape[:-3], ...)
        feats = feats.view(*orig_shape[:-3], *feats.shape[1:])

    return feats


if __name__ == "__main__":
    model_name = "vit_small_patch16_dinov3.lvd1689m"
    model = build_dinov3_model(model_name)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M, Trainable parameters: {trainable_params / 1e6:.2f} M")

    #
    print(f"Building model {CP.keyword(model_name)}")
    transform = build_transform()
    print(f"Model built")

    model.eval()
    images_load_path = "~/Dataset/LLIE_dataset/LOL_v2/Synthetic/Test/high/"
    for image_path in glob_single_files(f"{images_load_path}**", 'png'):
        img_pil = read_image_as_pil(image_path)
        img_tensor = transform(img_pil).unsqueeze(0)  # 1 × 3 × H × W
        h, w = img_pil.height, img_pil.width
        print(f"Processing: {image_path} with shape: {img_tensor.shape}")

        feats = forward_dinov3_model(model, img_tensor)
        print(f"Feature tensor shape: {feats.shape}")

        from basic.utils.console.logplot import get_root_plotter, Plotter
        plotter: Plotter = get_root_plotter("./.plotlogs")
        image_name = os.path.basename(image_path).split('.')[0]
        plotter.semantic_feature_map(feats[:, 1+4:], fig_name=f"{model_name}_feature_map/{image_name}", rearrange_option="B (H W) C -> B C H W", H=h // 16, W=w // 16)


