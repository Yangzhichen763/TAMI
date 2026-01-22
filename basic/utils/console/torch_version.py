import os
import torch


def get_torch_version_str():
    try:
        from .log import ColorPrefeb as CP, Font, color_text
        from .log import get_env_str
        debug = True
    except ImportError:
        print("Warning: Could not import log module. Some features may not work.")
        debug = False

    available = torch.cuda.is_available()
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    num_gpus = torch.cuda.device_count()
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpu_list = [int(i) for i in visible_devices.split(',')] if visible_devices else [i for i in range(num_gpus)]
    compute_capability = torch.cuda.get_device_capability(0)
    if debug:
        return (f"{color_text(get_env_str(), Font.LIGHT_CYAN)}\n"
                f"Torch version: {color_text(torch_version, Font.LIGHT_MAGENTA)}, "
                f"CUDA is {CP.bool(available, 'available', 'not available')}, "
                f"CUDA version: {color_text(cuda_version, Font.LIGHT_MAGENTA)}, "
                f"GPU count: {color_text(num_gpus, Font.LIGHT_MAGENTA)}, "
                f"GPU devices: {color_text(gpu_list, Font.LIGHT_MAGENTA)}, "
                f"Compute Capability: {color_text(f'{compute_capability[0]}.{compute_capability[1]}', Font.LIGHT_MAGENTA)}, "
                f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        return (f"Torch version: {torch_version}, "
                f"CUDA is {available}, "
                f"CUDA version: {cuda_version}, "
                f"GPU count: {num_gpus}, "
                f"GPU devices: {gpu_list}, "
                f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}, "
                f"Using CUDA device: {torch.cuda.get_device_name(0)}")


def assert_cuda_availability():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation or CUDA setup.")
