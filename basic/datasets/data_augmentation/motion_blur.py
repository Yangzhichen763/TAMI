import math
from random import Random

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .base import TransformBase


class LinearMotionBlurTransform(TransformBase):
    def __init__(self, p=1.0, length=21, angle_degree=0, return_kernel=False, **kwargs):
        super().__init__(p)
        self.length = length
        self.angle_degree = angle_degree
        self.return_kernel = return_kernel
        self.kwargs = kwargs

    def call(self, img):
        # Generate linear PSF kernel
        length = max(3, int(self.length) | 1)
        kernel = LinearMotionBlurTransform.linear_psf(length, self.angle_degree)

        # Add multiplicative noise to the kernel
        noise_range = self.kwargs.get('noise_range', None)
        noise_scale: float = self.kwargs.get('noise_scale', 1.0)
        kernel = add_multiplicative_noise(kernel, noise_range=noise_range, noise_scale=noise_scale)
        kernel = torch.Tensor(kernel).to(img.device)
        blurred_image = apply_motion_blur(img, kernel)

        if self.return_kernel:
            return blurred_image, kernel
        return blurred_image

    @staticmethod
    def linear_psf(length=21, angle_degree=0):
        """
        Generate linear PSF kernel
        """
        # Generate horizontal line kernel
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0

        # Rotate the kernel to the specified angle
        M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_degree, 1.0)
        kernel = cv2.warpAffine(kernel, M, (length, length), flags=cv2.INTER_LINEAR)
        kernel /= kernel.sum()
        return kernel


class RandomTrajectoryMotionBlurTransform(TransformBase):
    def __init__(self, p=1.0, kernel_size=45, steps=300, step_sigma=0.6, return_kernel=False, **kwargs):
        super().__init__(p)
        self.kernel_size = kernel_size
        self.steps = steps
        self.step_sigma = step_sigma
        self.return_kernel = return_kernel
        self.kwargs = kwargs

    def call(self, img):
        kernel = RandomTrajectoryMotionBlurTransform.random_trajectory_psf(self.kernel_size, self.steps, self.step_sigma)
        blurred = apply_motion_blur(img, kernel)

        if self.return_kernel:
            return blurred, kernel
        return blurred

    @staticmethod
    def random_trajectory_psf(size=45, steps=300, step_sigma=0.6, center_bias=True, kernel_blur=False):
        """
        Generate random camera trajectory PSF kernel (nonlinear multi-directional blur)

        Args:
            size: PSF kernel size
            steps: number of trajectory sampling points
            step_sigma: standard deviation of each step's Gaussian random step length
            center_bias: whether to fix the initial point of the trajectory at the center

        Returns:
            kernel: PSF kernel of shape (size, size)
        """
        H = W = size
        kernel = np.zeros((H, W), np.float32)

        if center_bias:
            p = np.array([W // 2, H // 2], np.float32)
        else:
            p = np.array([np.random.uniform(0, W - 1), np.random.uniform(0, H - 1)], np.float32)

        for _ in range(steps):
            step = np.random.randn(2).astype(np.float32) * step_sigma
            p = p + step
            p[0] = np.clip(p[0], 0, W - 1)
            p[1] = np.clip(p[1], 0, H - 1)
            kernel[int(round(p[1])), int(round(p[0]))] += 1.0

        if kernel_blur:
            kernel = cv2.GaussianBlur(kernel, (3,3), 0.8)
        kernel /= max(kernel.sum(), 1e-8)
        return kernel


"""
Adopted and refactored from LEDNet(https://github.com/sczhou/LEDNet/blob/master/basicsr/data/degradations.py#L14-L220)
motion blur
"""
class GaussianMotionBlurTransform(TransformBase):
    def __init__(self, p=1.0, noise_range=None, return_kernel=False, **kwargs):
        super().__init__(p)
        self.noise_range = noise_range
        self.return_kernel = return_kernel
        self.kwargs = kwargs
        if 'kwargs' in self.kwargs: # 防止套娃
            self.kwargs = self.kwargs['kwargs']

        if len(self.kwargs.items()) == 0 or 'type' not in self.kwargs:
            print('MotionBlurTransform kwargs is empty, use default')
            self.kwargs = dict(
                type='anisotropic',
                kernel_size=21,
            )

    def call(self, img):
        device = img.device

        kernel_size = self.kwargs.get('kernel_size', 21)
        strict: bool = self.kwargs.get('strict', False)
        if self.kwargs['type'] == 'anisotropic':
            sigma_x_range = self.kwargs.get('sigma_x_range', [0.6, 3.0])
            sigma_y_range = self.kwargs.get('sigma_y_range', [0.6, 3.0])
            rotation_range = self.kwargs.get('rotation_range', [-math.pi, math.pi])
            kernel = self.random_bivariate_anisotropic_Gaussian(
                kernel_size=kernel_size,
                sigma_x_range=sigma_x_range,
                sigma_y_range=sigma_y_range,
                rotation_range=rotation_range,
                strict=strict
            )
        elif self.kwargs['type'] == 'isotropic':
            sigma_range = self.kwargs.get('sigma_range', [0.6, 3.0])
            kernel = self.random_bivariate_isotropic_Gaussian(
                kernel_size=kernel_size,
                sigma_range=sigma_range,
                strict=strict
            )
        else:
            raise ValueError(f"Unknown motion blur type: {self.kwargs['type']}")

        noise_range = self.kwargs.get('noise_range', None)
        noise_scale: float = self.kwargs.get('noise_scale', 1.0)
        kernel = add_multiplicative_noise(kernel, noise_range=noise_range, noise_scale=noise_scale)
        kernel = torch.Tensor(kernel).to(device)
        blurred_image = apply_motion_blur(img, kernel)

        if self.return_kernel:
            return blurred_image, kernel
        return blurred_image

    @staticmethod
    def sigma_matrix2(sig_x, sig_y, theta):
        """Calculate the rotated sigma matrix (two dimensional matrix).
        Args:
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
        Returns:
            ndarray: Rotated sigma matrix.
        """
        D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(U, np.dot(D, U.T))

    @staticmethod
    def mesh_grid(kernel_size):
        """Generate the mesh grid, centering at zero.
        Args:
            kernel_size (int):
        Returns:
            xy (ndarray): with the shape (kernel_size, kernel_size, 2)
            xx (ndarray): with the shape (kernel_size, kernel_size)
            yy (ndarray): with the shape (kernel_size, kernel_size)
        """
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                               1))).reshape(kernel_size, kernel_size, 2)
        return xy, xx, yy

    @staticmethod
    def pdf2(sigma_matrix, grid):
        """Calculate PDF of the bivariate Gaussian distribution.
        Args:
            sigma_matrix (ndarray): with the shape (2, 2)
            grid (ndarray): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size.
        Returns:
            kernel (ndarrray): un-normalized kernel.
        """
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
        return kernel

    @staticmethod
    def mass_center_shift(kernel_size, kernel):
        """Calculate the shift of the mass center of a kenrel.
        Args:
            kernel_size (int):
            kernel (ndarray): normalized kernel.
        Returns:
            delta_h (float):
            delta_w (float):
        """
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        col_sum, row_sum = np.sum(kernel, axis=0), np.sum(kernel, axis=1)
        delta_h = np.dot(row_sum, ax)
        delta_w = np.dot(col_sum, ax)
        return delta_h, delta_w

    @staticmethod
    def bivariate_anisotropic_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None):
        """Generate a bivariate anisotropic Gaussian kernel.
        Args:
            kernel_size (int):
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None
        Returns:
            kernel (ndarray): normalized kernel.
        """
        if grid is None:
            grid, _, _ = GaussianMotionBlurTransform.mesh_grid(kernel_size)
        sigma_matrix = GaussianMotionBlurTransform.sigma_matrix2(sig_x, sig_y, theta)
        kernel = GaussianMotionBlurTransform.pdf2(sigma_matrix, grid)
        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def bivariate_isotropic_Gaussian(kernel_size, sig, grid=None):
        """Generate a bivariate isotropic Gaussian kernel.
        Args:
            kernel_size (int):
            sig (float):
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None
        Returns:
            kernel (ndarray): normalized kernel.
        """
        if grid is None:
            grid, _, _ = GaussianMotionBlurTransform.mesh_grid(kernel_size)
        sigma_matrix = np.array([[sig ** 2, 0], [0, sig ** 2]])
        kernel = GaussianMotionBlurTransform.pdf2(sigma_matrix, grid)
        kernel = kernel / np.sum(kernel)
        return kernel

    # 各向异性高斯核
    @staticmethod
    def random_bivariate_anisotropic_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range,
            strict=False
    ):
        """Randomly generate bivariate anisotropic Gaussian kernels.
        Args:
            kernel_size (int):
            sigma_x_range (tuple, list): [0.6, 5]
            sigma_y_range (tuple, list): [0.6, 5]
            rotation_range (tuple, list): [-math.pi, math.pi]
            strict (bool):
        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, 'Kernel size must be odd.'

        def parse_arg(x):
            if isinstance(x, (list, tuple)):
                if x[0] != x[1]:
                    return np.random.uniform(x[0], x[1])
                else:
                    return x[0]
            else:
                return x

        sigma_x = parse_arg(sigma_x_range)
        sigma_y = parse_arg(sigma_y_range)
        if strict:
            sigma_max = np.max([sigma_x, sigma_y])
            sigma_min = np.min([sigma_x, sigma_y])
            sigma_x, sigma_y = sigma_max, sigma_min
        rotation = parse_arg(rotation_range)

        if sigma_x == 0 or sigma_y == 0:
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, kernel_size // 2] = 1.0
        else:
            kernel = GaussianMotionBlurTransform.bivariate_anisotropic_Gaussian(kernel_size, sigma_x, sigma_y, rotation)
            kernel = kernel / kernel.sum()

        if strict:
            return kernel, sigma_x, sigma_y, rotation
        else:
            return kernel

    # 各向同性高斯核
    @staticmethod
    def random_bivariate_isotropic_Gaussian(
            kernel_size, sigma_range, strict=False
    ):
        """Randomly generate bivariate isotropic Gaussian kernels.
        Args:
            kernel_size (int):
            sigma_range (tuple, list): [0.6, 5]
            strict (bool):
        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
        assert sigma_range[0] < sigma_range[1], 'Wrong sigma_x_range.'
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])

        kernel = GaussianMotionBlurTransform.bivariate_isotropic_Gaussian(kernel_size, sigma)
        kernel = kernel / np.sum(kernel)
        if strict:
            return kernel, sigma
        else:
            return kernel


"""
More Motion Blur Kernels Generation Method see:
- https://github.com/LeviBorodenko/motionblur/tree/master
"""


# 在核中添加乘性噪声
def add_multiplicative_noise(kernel, noise_range=None, noise_scale=1):
    """Add multiplicative noise to a kernel.
    Args:
        kernel (ndarray): Input kernel.
        noise_range: Range of multiplicative noise.
        noise_scale (float): Scale of multiplicative noise. Default: 1.
    Returns:
        ndarray: Noisy image.
    """
    if noise_range is None:
        return kernel
    assert noise_range[0] < noise_range[1], 'Wrong noise range.'

    if isinstance(noise_range, (tuple, list)):
        noise_range = dict(min=noise_range[0], max=noise_range[1])
    kernel_size = kernel.shape
    _kernel_size = (int(kernel_size[0] * (1 / noise_scale)), int(kernel_size[1] * (1 / noise_scale)))

    noise = np.random.uniform(noise_range['min'], noise_range['max'], size=_kernel_size)
    if noise_scale != 1:
        noise = np.resize(noise, kernel.shape)
    noise[kernel_size[0] // 2, kernel_size[1] // 2] = 1.0
    kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_motion_blur(image, kernel):
    in_channels = image.shape[1]
    kernel_size = kernel.shape[0]

    kernel = kernel.to(image.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)       # (1, 1, K, K)
    kernel = kernel.repeat(in_channels, 1, 1, 1)    # (3, 1, K, K)

    blurred_image = F.conv2d(image, kernel, padding=kernel_size // 2, groups=in_channels)
    return blurred_image


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


    # ================================================
    # 直接在 RGB 图像上面加运动模糊
    # ================================================
    # transform = T.Compose([
    #     T.ToTensor(),
    #     MotionBlurTransform(
    #         p=1.0,
    #         kwargs=dict(
    #             type='anisotropic',
    #             kernel_size = 21,
    #             sigma_x_range = (1.0, 5.0),
    #             sigma_y_range = (0.3, 0.5),
    #             rotation_range = (-math.pi, math.pi),
    #             strict = False
    #         )
    #     )
    # ])
    # transform = T.Compose([
    #     T.ToTensor(),
    #     GaussianMotionBlurTransform(
    #         p=1.0,
    #         kwargs=dict(
    #             type='anisotropic',
    #             kernel_size = 61,
    #             sigma_x_range = 15.0,
    #             sigma_y_range = 1.0,
    #             rotation_range = (-math.pi, math.pi),
    #             noise_range=(0, 1),
    #             noise_scale=8,
    #             strict = False,
    #
    #             return_kernel=True
    #         )
    #     )
    # ])
    # transform = T.Compose([
    #     T.ToTensor(),
    #     RandomTrajectoryMotionBlurTransform(
    #         p=1.0,
    #         kernel_size=49,
    #         steps=10,
    #         step_sigma=3,
    #
    #         return_kernel=True
    #     )
    # ])
    # blurred_image, kernel = transform(image)
    # print(f"Motion Blur PSNR: {calcu_psnr(clean_image, blurred_image)}")
    # save_image(tensor2numpy(kernel.detach().cpu() / kernel.max(), reverse_channels=False), '.tmp/blur_kernel.jpg')
    # save_image(tensor2numpy(blurred_image), f'.tmp/blurred_image.jpg')


    # ================================================
    # 可视化传感器噪声：不同曝光时间和读取时间的组合
    # ================================================
    import matplotlib.pyplot as plt

    # 定义参数范围
    sigma_x_values = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0]  # 行
    sigma_y_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0]  # 列

    n_rows = len(sigma_x_values)
    n_cols = len(sigma_y_values)

    # 创建 matplotlib 图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    random_rotation = np.random.uniform(-math.pi, math.pi)
    for i, sigma_x in enumerate(sigma_x_values):
        for j, sigma_y in enumerate(sigma_y_values):
            transform = T.Compose([
                GaussianMotionBlurTransform(
                    p=1.0,
                    kwargs=dict(
                        type='anisotropic',
                        kernel_size=math.ceil(max(sigma_x, sigma_y)) * 4 + 1,
                        sigma_x_range=sigma_x,
                        sigma_y_range=sigma_y,
                        rotation_range=random_rotation,
                        noise_range=(0, 1),
                        noise_scale=4,
                        strict=False
                    )
                )
            ])

            blur_image = transform(image)
            np_img = tensor2numpy(blur_image)

            # 绘制到子图
            ax = axes[i, j] if n_rows > 1 else axes[j]
            ax.imshow(np_img)
            ax.axis('off')
            ax.set_title(f"σx={sigma_x:.1f}, σy={sigma_y:.1f}", fontsize=8)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.tight_layout()
    plt.savefig('.tmp/grid_blur_image.pdf', format='pdf', dpi=200, bbox_inches='tight')
    plt.show()