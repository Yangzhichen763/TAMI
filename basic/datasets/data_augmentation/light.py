import math
import torch
import torchvision.transforms.functional as TF

from .util import process_raw_to_srgb, unprocess_srgb_to_raw
from .base import TransformBase


class SensorLightTransform(TransformBase):
    """
    Simulate low-light conditions by converting sRGB to RAW, applying light reduction,
    and converting back to sRGB. This transformation mimics the effect of reduced
    light exposure during image capture.
    """

    def __init__(self, intensity=1.0, p=1.0):
        """
        Initialize the low-light transformation.

        Args:
            intensity (float): Intensity coefficient for controlling light reduction level.
                                     Lower values result in darker images.
            p (float): Probability of applying the transformation. Range: [0, 1]
        """
        super().__init__(p)
        self.light_intensity = intensity

    def call(self, img):
        """
        Apply low-light transformation to the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image in sRGB color space.

        Returns:
            torch.Tensor: Low-light image in sRGB color space with same format as input.
        """
        # sRGB -> RAW (reduce light) -> sRGB （RAW 图像还原时可能会出现色彩偏移）
        clean_raw, metadata = unprocess_srgb_to_raw(img)
        low_raw = apply_linear_darken(clean_raw, self.light_intensity)
        low_img = process_raw_to_srgb(low_raw, metadata['red_gain'], metadata['blue_gain'], metadata['cam2rgb'])

        return low_img


class GammaLightTransform(TransformBase):
    """
    Simulate low-light conditions by applying gamma correction and intensity reduction in sRGB space.
    This provides a simpler alternative to the full sensor simulation.
    """

    def __init__(
            self,
            intensity=None, gamma=None, p=1.0,
            intensity_gamma_function=None,
    ):
        """
        Initialize the gamma and intensity transformation.

        Args:
            intensity: Intensity coefficient for controlling light reduction level.
                Lower values result in darker images. If None, random intensity is used.
            gamma: Gamma value for gamma correction. If None, random gamma is used.
            p (float): Probability of applying the transformation. Range: [0, 1]
            intensity_gamma_function: Function to map intensity and gamma to a combined value.
                e.g.
                    def ig_function(i, g):
                        g_min = 1 / (2.2 * i)
                        g_max = 2.2 / i
                        return min(max(g, g_min), g_max)
                    dict(input="intensity", output="gamma", function=ig_function)
        """
        self.intensity = intensity
        self.gamma = gamma
        super().__init__(p)

        self.intensity_gamma_function = intensity_gamma_function

    def call(self, img):
        """
        Apply gamma correction and intensity reduction to the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image in sRGB color space.

        Returns:
            torch.Tensor: Low-light image in sRGB color space with same format as input.
        """
        # calculate gamma and intensity
        gamma = random_gamma(gamma=self.gamma, device=img.device)
        intensity = random_intensity(intensity=self.intensity, device=img.device)
        if self.intensity_gamma_function is not None:
            if self.intensity_gamma_function['output'] == 'gamma':
                gamma = self.intensity_gamma_function(intensity, gamma)
            elif self.intensity_gamma_function['output'] == 'intensity':
                intensity = self.intensity_gamma_function(intensity, gamma)

        # apply gamma correction first, then intensity reduction
        low_raw = apply_gamma_darken(img, gamma)
        low_img = apply_linear_darken(low_raw, intensity)

        return low_img


"""
低光图像合成过程参考论文：[PR 2016] LLNet: Low-light Image Enhancement with Deep Learning
"""
def random_intensity(intensity=None, device='cpu') -> float:
    """
    Generate random intensity value.

    Args:
        intensity: Intensity value
        device (Any): Device to use for noise generation ('cpu' or 'cuda')

    Returns:
        float: Random intensity value
    """
    if isinstance(intensity, (tuple, list)):
        intensity = dict(min=intensity[0], max=intensity[1])
    if intensity is None:
        intensity = dict(min=0.1, max=0.5)

    if isinstance(intensity, dict):
        intensity = torch.empty(size=(1,)).uniform_(intensity['min'], intensity['max'])
        intensity = intensity.to(device)

    return intensity

def random_gamma(gamma=None, device='cpu') -> [float, int]:
    """
    Generate random gamma value.

    Args:
        gamma: Gamma value
        device (Any): Device to use for noise generation ('cpu' or 'cuda')

    Returns:
        tuple: (lambda_shot_array, lambda_read_array) as numpy arrays
    """
    if isinstance(gamma, (tuple, list)):
        gamma = dict(min=gamma[0], max=gamma[1])
    if gamma is None:
        gamma = dict(min=2, max=5)

    if isinstance(gamma, dict):
        gamma = torch.empty(size=(1,)).uniform_(gamma['min'], gamma['max'])
        gamma = gamma.to(device)

    return gamma

def apply_linear_darken(image, intensity: float):
    darken_image = image * intensity

    return darken_image

def apply_gamma_darken(image, gamma: float):
    darken_image = image ** gamma

    return darken_image


if __name__ == "__main__":
    import sys
    sys.path.append('.')

    import torchvision.transforms as T

    from basic.utils.io import read_image_as_numpy, read_image_as_pil, save_image
    from basic.utils.convert import numpy2tensor, tensor2numpy

    def calcu_psnr(a, b):
        a = a.detach().cpu()
        b = b.detach().cpu()
        return (10 * torch.log10(1 / torch.mean((a - b) ** 2))). item()

    image_path = "~/Dataset/LLVE/DID-1080/test/high/video102/001.jpg"
    image = read_image_as_pil(image_path)
    clean_image = numpy2tensor(read_image_as_numpy(image_path))
    save_image(tensor2numpy(clean_image), '.tmp/clean_image.jpg')

    # # ================================================
    # # 模拟传感器光照变换：调整光照强度
    # # ================================================
    # transform = T.Compose([
    #     T.ToTensor(),
    #     SensorLightTransform(intensity=0.1, p=1.0),
    # ])
    # lowlight_image = transform(image)
    # save_image(tensor2numpy(lowlight_image), '.tmp/lowlight_image_fromraw.jpg')
    # print(f"SensorLight PSNR: {calcu_psnr(clean_image, lowlight_image)}")
    #
    #
    # # ================================================
    # # 模拟Gamma光照变换：调整光照强度和Gamma值
    # # ================================================
    # transform = T.Compose([
    #     T.ToTensor(),
    #     GammaLightTransform(intensity=dict(min=0.1, max=0.5), gamma=1.0, p=1.0),
    # ])
    # lowlight_image = transform(image)
    # save_image(tensor2numpy(lowlight_image), '.tmp/lowlight_image.jpg')
    # print(f"Light       PSNR: {calcu_psnr(clean_image, lowlight_image)}")
    #
    # # LLNet 中的低光合成配置：固定光照强度，调整Gamma值
    # transform = T.Compose([
    #     T.ToTensor(),
    #     GammaLightTransform(intensity=1.0, gamma=dict(min=2, max=5), p=1.0),
    # ])
    # lowlight_image = transform(image)
    # save_image(tensor2numpy(lowlight_image), '.tmp/lowlight_image_llnet.jpg')
    # print(f"LLNet Light PSNR: {calcu_psnr(clean_image, lowlight_image)}")


    # ================================================
    # 可视化Gamma光照变换：不同光照强度和Gamma值的组合
    # ================================================
    import matplotlib.pyplot as plt

    # 定义参数范围
    intensity_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]  # 行
    gamma_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # 列

    n_rows = len(intensity_values)
    n_cols = len(gamma_values)

    # 创建 matplotlib 图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    for i, intensity in enumerate(intensity_values):
        for j, gamma in enumerate(gamma_values):
            transform = T.Compose([
                T.ToTensor(),
                GammaLightTransform(intensity=intensity, gamma=gamma, p=1.0),
            ])

            lowlight_image = transform(image)
            np_img = tensor2numpy(lowlight_image)

            # 绘制到子图
            ax = axes[i, j] if n_rows > 1 else axes[j]
            ax.imshow(np_img)
            ax.axis('off')
            ax.set_title(f"I={intensity:.1f}, γ={gamma:.1f}", fontsize=8)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.tight_layout()
    plt.savefig('.tmp/grid_lowlight_image.pdf', format='pdf', dpi=200, bbox_inches='tight')
    plt.show()