import math
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional_tensor import rgb_to_grayscale

from .util import process_raw_to_srgb, unprocess_srgb_to_raw
from .base import TransformBase


class SensorNoiseTransform(TransformBase):
    """
    Add realistic sensor noise by converting sRGB to RAW, applying noise, and converting back to sRGB.
    This simulates the noise characteristics of camera sensors during image capture.
    """

    def __init__(self, noise_intensity=1.0, p=1.0, lambda_shot=None, lambda_read=None):
        """
        Args:
            noise_intensity (float): Intensity coefficient for controlling noise level.
                                     Higher values result in more noise.
            p (float): Probability of applying the transformation. Range: [0, 1]
            lambda_read (dict, optional): Parameters for read noise distribution.
                                         If None, uses default camera parameters.
            lambda_shot (dict, optional): Parameters for shot noise distribution.
                                         If None, uses default camera parameters.
        """
        super().__init__(p)
        self.noise_intensity = noise_intensity
        self.lambda_shot = lambda_shot
        self.lambda_read = lambda_read

    def call(self, img):
        """
        Apply sensor noise transformation to the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image in sRGB color space.

        Returns:
            torch.Tensor: Noisy image in sRGB color space with same format as input.
        """
        device = img.device

        lambda_shot, lambda_read = random_noise_levels(self.lambda_shot, self.lambda_read, intensity=self.noise_intensity, device=device)
        # sRGB -> RAW (add sensor noise) -> sRGB
        clean_raw, metadata = unprocess_srgb_to_raw(img)
        noisy_raw = generate_gaussian_noise(clean_raw, lambda_shot, lambda_read)
        noisy_img = process_raw_to_srgb(noisy_raw, metadata['red_gain'], metadata['blue_gain'], metadata['cam2rgb'])
        noisy_img = add_noise(img, noisy_img)   # 先生成在 RAW 中处理的噪声，然后再叠加在图像上，可以避免出现 RAW 图像还原时的色彩偏移

        return noisy_img


class GaussianNoiseTransform(TransformBase):
    """
    Apply general noise to images by adding random Gaussian noise.
    """

    def __init__(self, noise_intensity=1.0, p=1.0, lambda_shot=None, lambda_read=None):
        """
        Args:
            noise_intensity (float): Intensity coefficient for controlling noise level.
                                     Higher values result in more noise.
            p (float): Probability of applying the transformation. Range: [0, 1]
            lambda_read (dict, optional): Parameters for read noise distribution.
                                         If None, uses default parameters.
            lambda_shot (dict, optional): Parameters for shot noise distribution.
                                         If None, uses default parameters.
        """
        super().__init__(p)
        self.noise_intensity = noise_intensity
        self.lambda_shot = lambda_shot
        self.lambda_read = lambda_read

    def call(self, img):
        """
        Apply noise transformation to the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Noisy image with same format as input.
        """
        device = img.device

        lambda_shot, lambda_read = random_noise_levels(self.lambda_shot, self.lambda_read, intensity=self.noise_intensity, device=device)
        # add noise
        noisy_img = generate_gaussian_noise(img, lambda_shot, lambda_read)
        noisy_img = add_noise(img, noisy_img)

        return noisy_img


class PoissonNoiseTransform(TransformBase):
    """
    Apply general noise to images by adding random Poisson noise.
    """

    def __init__(self, noise_intensity=1.0, p=1.0, gray_noise=0):
        super().__init__(p)
        self.noise_intensity = noise_intensity
        self.gray_noise = gray_noise

    def call(self, img):
        """
        Apply poisson noise transformation to the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Noisy image with same format as input.
        """
        # add noise
        noisy_img = generate_poisson_noise(img, self.noise_intensity, self.gray_noise)
        noisy_img = add_noise(img, noisy_img)

        return noisy_img

"""
Generate noise
噪声合成的过程参考代码：[CVPR 2019] Unprocessing Images for Learned Raw Denoising (https://github.com/timothybrooks/unprocessing)
"""
def random_noise_levels(lambda_shot=None, lambda_read=None, intensity=None, device='cpu'):
    """
    Generate random noise levels as numpy arrays.

    Args:
        lambda_shot: Shot noise parameters (dict, float, or int)
        lambda_read: Read noise parameters (dict, float, or int)
        intensity: Noise intensity multiplier
        device (Any): Device to use for noise generation ('cpu' or 'cuda')

    Returns:
        tuple: (lambda_shot_array, lambda_read_array) as numpy arrays
    """
    if lambda_shot is None:
        lambda_shot = dict(min=0.0001, max=0.012)
    if lambda_read is None:
        lambda_read = dict(miu=dict(a=2.18, b=1.2), sigma=0.26)
    if intensity is None:
        intensity = 1.0

    if isinstance(intensity, (tuple, list)):
        intensity = dict(min=intensity[0], max=intensity[1])
    if isinstance(intensity, dict):
        intensity = torch.empty(size=(1,)).uniform_(intensity['min'], intensity['max']).item()

    # Calculate the noise variance for each pixel
    if isinstance(lambda_shot, (tuple, list)):
        lambda_shot = dict(min=lambda_shot[0], max=lambda_shot[1])
    if isinstance(lambda_shot, dict):
        a = math.log(lambda_shot["min"] * intensity)
        b = math.log(lambda_shot["max"] * intensity)
        log_lambda_shot = torch.empty(size=(1,)).uniform_(a, b)
        lambda_shot = torch.exp(log_lambda_shot)
        lambda_shot = lambda_shot.to(device)
    elif isinstance(lambda_shot, (float, int)):
        lambda_shot = lambda_shot * intensity

    if isinstance(lambda_read, dict):
        miu_a, miu_b = lambda_read["miu"]['a'], lambda_read["miu"]['b']
        sigma = lambda_read["sigma"]
        miu = miu_a * torch.log(lambda_shot) + miu_b
        log_lambda_read = torch.normal(mean=miu, std=sigma) + miu
        lambda_read = torch.exp(log_lambda_read)
        lambda_read = lambda_read.to(device)
    elif isinstance(lambda_read, (float, int)):
        lambda_read = lambda_read * intensity

    return lambda_shot, lambda_read

def add_noise(image, noise):
    noised_image = image + noise
    noised_image = torch.clamp(noised_image, 0.0, 1.0)

    return noised_image

def generate_gaussian_noise(image, shot_noise=None, read_noise=None):
    """
    Add gaussian noise to a batch of images (PyTorch version).

    Args:
        image (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        shot_noise (float | Tensor): Shot noise scale. Default: 0.01.
        read_noise (float | Tensor): Read noise scale. Default: 0.0005.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1], float32.
    """
    if shot_noise is None:
        shot_noise = 0.01
    if read_noise is None:
        read_noise = 0.0005

    variance = image * shot_noise + read_noise
    noise = torch.normal(mean=0.0, std=torch.sqrt(variance))
    return noise

"""
Adopted from LEDNet(https://github.com/sczhou/LEDNet/blob/master/basicsr/data/degradations.py#L338-L381)
poisson noise (shot)
"""
def generate_poisson_noise(image, intensity=1.0, gray_noise=0):
    """
    Add poisson noise to a batch of images (PyTorch version).

    Poisson noise is a type of noise that occurs in digital images when the number of photons
    hitting the sensor is not constant. It is a non-linear noise that depends on the intensity
    of the image.

    Args:
        image (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        intensity (float | Tensor): Noise scale. Number or Tensor with shape (b). Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b). 0 for False, 1 for True. Default: 0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1], float32.
    """
    b, _, h, w = image.size()

    # Convert gray_noise to appropriate format
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    # Generate gray noise if needed
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(image, num_output_channels=1)
        # Round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.0
        # Use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # Always calculate color noise
    # Round and clip image for counting vals correctly
    img_clipped = torch.clamp((image * 255.0).round(), 0, 255) / 255.0
    # Use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(img_clipped[i, :, :, :])) for i in range(b)]
    vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img_clipped.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img_clipped * vals) / vals
    noise = out - img_clipped

    # Combine gray and color noise if needed
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise

    # Apply scale
    if not isinstance(intensity, (float, int)):
        intensity = intensity.view(b, 1, 1, 1)

    return noise * intensity


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
    # # 将 sRGB 图像转换为 RAW 图像，加噪，再转换回 sRGB 图像
    # # ================================================
    # transform = T.Compose([
    #     T.ToTensor(),
    #     SensorNoiseTransform(noise_intensity=1.0, p=1.0, lambda_shot=dict(a=0.0001, b=0.012)),
    # ])
    # noisy_image = transform(image)
    # save_image(tensor2numpy(noisy_image), '.tmp/noisy_image_fromraw.jpg')
    # print(f"Sensor Noise  PSNR: {calcu_psnr(clean_image, noisy_image)}")
    #
    #
    # # ================================================
    # # 直接在 RGB 图像上面加噪
    # # ================================================
    # transform = T.Compose([
    #     T.ToTensor(),
    #     GaussianNoiseTransform(noise_intensity=1.0, p=1.0, lambda_shot=0.01),
    # ])
    # noisy_image = transform(image)
    # save_image(tensor2numpy(noisy_image), '.tmp/noisy_image.jpg')
    # print(f"Noise         PSNR: {calcu_psnr(clean_image, noisy_image)}")
    #
    #
    # # LLNet[PR16] 和 LEDNet[ECCV22] 中的加噪配置都是如此
    # transform = T.Compose([
    #     T.ToTensor(),
    #     GaussianNoiseTransform(noise_intensity=1.0, p=1.0, lambda_shot=0.0, lambda_read=0.01),
    # ])
    # noisy_image = transform(image)
    # save_image(tensor2numpy(noisy_image), '.tmp/noisy_image_gaussian.jpg')
    # print(f"LLNet Noise   PSNR: {calcu_psnr(clean_image, noisy_image)}")
    #
    #
    # # LEDNet[ECCV22] 中的另一种加噪配置：泊松噪声
    # transform = T.Compose([
    #     T.ToTensor(),
    #     PoissonNoiseTransform(noise_intensity=1.0, p=1.0, gray_noise=0),
    # ])
    # noisy_image = transform(image)
    # save_image(tensor2numpy(noisy_image), '.tmp/noisy_image_poisson.jpg')
    # print(f"Poisson Noise PSNR: {calcu_psnr(clean_image, noisy_image)}")


    # ================================================
    # 可视化传感器噪声：不同曝光时间和读取时间的组合
    # ================================================
    import matplotlib.pyplot as plt

    # 定义参数范围
    lambda_shot_values = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]  # 行
    lambda_read_values = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]  # 列

    n_rows = len(lambda_shot_values)
    n_cols = len(lambda_read_values)

    # 创建 matplotlib 图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    for i, lambda_shot in enumerate(lambda_shot_values):
        for j, lambda_read in enumerate(lambda_read_values):
            transform = T.Compose([
                T.ToTensor(),
                SensorNoiseTransform(noise_intensity=1.0, p=1.0, lambda_shot=math.exp(lambda_shot), lambda_read=math.exp(lambda_read)),
            ])

            noise_image = transform(image)
            np_img = tensor2numpy(noise_image)

            # 绘制到子图
            ax = axes[i, j] if n_rows > 1 else axes[j]
            ax.imshow(np_img)
            ax.axis('off')
            ax.set_title(f"λs={lambda_shot:.1f}, λr={lambda_read:.1f}", fontsize=8)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.tight_layout()
    plt.savefig('.tmp/grid_noise_image.pdf', format='pdf', dpi=200, bbox_inches='tight')
    plt.show()

