

if __name__ == '__main__':
    import sys
    sys.path.append('.')

    import torch
    import torchvision.transforms as T

    from basic.utils.io import read_image_as_numpy, read_image_as_pil, save_image
    from basic.utils.convert import numpy2tensor, tensor2numpy
    import cv2
    import math


    def calcu_psnr(a, b):
        a = a.detach().cpu()
        b = b.detach().cpu()
        return (10 * torch.log10(1 / torch.mean((a - b) ** 2))).item()


    lq_image_path = "~/Dataset/LLVE/SDSD-indoor/eval/input/pair13/0163.png"
    hq_image_path = "~/Dataset/LLVE/SDSD-indoor/eval/GT/pair13/0163.png"
    lq_image = numpy2tensor(read_image_as_numpy(lq_image_path))
    hq_image = numpy2tensor(read_image_as_numpy(hq_image_path))
    save_image(tensor2numpy(lq_image.detach().cpu()), '.tmp/input_image.jpg')
    save_image(tensor2numpy(hq_image.detach().cpu()), '.tmp/clean_image.jpg')


    # ================================================
    # 三种混合
    # ================================================
    from light import GammaLightTransform
    from noise import SensorNoiseTransform, GaussianNoiseTransform
    from motion_blur import GaussianMotionBlurTransform

    transform = T.Compose([
        GammaLightTransform(intensity=dict(min=0.2, max=0.8), gamma=dict(min=2.2, max=2.5), p=1.0),
        GaussianMotionBlurTransform(
            p=1.0,
            noise_range=(0.75, 1.25),
            kwargs=dict(
                type='anisotropic',
                kernel_size=15,
                sigma_x_range=(1.0, 16.0),
                sigma_y_range=(0.1, 0.5),
                rotation_range=(-math.pi, math.pi),
                strict=False
            )
        ),
        SensorNoiseTransform(noise_intensity=0.2, p=1.0, lambda_shot=(0.0001, 0.0012)),
        GaussianNoiseTransform(noise_intensity=0.2, p=1.0, lambda_shot=(0.0001, 0.0012)),
    ])
    for i in range(10):
        mixed_image = transform(hq_image)
        print(f"Motion Blur PSNR: {calcu_psnr(lq_image, mixed_image)}")
        cv2.imshow(f'blurred_image_{i}', tensor2numpy(mixed_image))
    cv2.waitKey(0)
    save_image(tensor2numpy(mixed_image), f'.tmp/mixed_image.jpg')