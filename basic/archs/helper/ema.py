import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import warnings


class EMA:
    """Exponential Moving Average for target network"""
    def __init__(self, model: nn.Module, decay: float = 0.99925):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model):
        for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
            target_params.data.lerp_(online_params.data, 1.0 - self.decay)


class EMAModule(nn.Module):
    """
    Exponential Moving Average for target network.
    Keeps a frozen copy of the online network that updates slowly.
    """
    def __init__(self, model: nn.Module, decay: float = 0.99925):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model: nn.Module):
        online_dict = dict(online_model.named_parameters())
        target_dict = dict(self.model.named_parameters())

        for name, target_param in target_dict.items():
            if name in online_dict:
                online_param = online_dict[name]
                if target_param.shape == online_param.shape:
                    target_param.data.lerp_(online_param.data, 1.0 - self.decay)
                else:
                    warnings.warn(
                        f"[Update] Shape mismatch for '{name}': "
                        f"{online_param.shape} -> {target_param.shape}"
                    )

    @torch.no_grad()
    def reset(self, online_model: nn.Module):
        self.model = copy.deepcopy(online_model)
        self.model.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(False)
        return super().train(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class LazyEMAModule(nn.Module):
    """
    Exponential Moving Average for target network.
    Keeps a frozen copy of the online network that updates slowly.
    """
    def __init__(self, decay: float = 0.99925):
        super().__init__()
        self.decay = decay
        self.model = None

    @torch.no_grad()
    def update(self, online_model: nn.Module):
        if self.model is None:
            self.reset(online_model)

        online_dict = dict(online_model.named_parameters())
        target_dict = dict(self.model.named_parameters())

        for name, target_param in target_dict.items():
            if name in online_dict:
                online_param = online_dict[name]
                if target_param.shape == online_param.shape:
                    target_param.data.lerp_(online_param.data, 1.0 - self.decay)
                else:
                    warnings.warn(
                        f"[Update] Shape mismatch for '{name}': "
                        f"{online_param.shape} -> {target_param.shape}"
                    )

    def reset(self, online_model: nn.Module):
        self.model = copy.deepcopy(online_model)
        self.model.requires_grad_(False)
        device = next(online_model.parameters()).device
        self.model.to(device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        if self.model is None:
            return self
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, *args, **kwargs):
        if self.model is None:
            return self
        self.model.train(False)
        return super().train(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class DynamicEMAModule(nn.Module):
    """
    Exponential Moving Average for target network.
    Keeps a frozen copy of the online network that updates slowly.
    """
    def __init__(self, model: nn.Module, decay_range=(0.99, 0.99925)):
        super().__init__()
        self.decay_range = decay_range
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model: nn.Module, progress: float):
        progress = max(0.0, min(1.0, progress))
        decay = self.decay_range[0] + (self.decay_range[1] - self.decay_range[0]) * progress

        online_dict = dict(online_model.named_parameters())
        target_dict = dict(self.model.named_parameters())

        for name, target_param in target_dict.items():
            if name in online_dict:
                online_param = online_dict[name]
                if target_param.shape == online_param.shape:
                    target_param.data.lerp_(online_param.data, 1.0 - decay)
                else:
                    warnings.warn(
                        f"[Update] Shape mismatch for '{name}': "
                        f"{online_param.shape} -> {target_param.shape}"
                    )

    def reset(self, online_model: nn.Module):
        self.model = copy.deepcopy(online_model)
        self.model.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(False)
        return super().train(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class LazyDynamicEMAModule(nn.Module):
    """
    Exponential Moving Average for target network.
    Keeps a frozen copy of the online network that updates slowly.
    """
    def __init__(self, decay_range=(0.99, 0.99925)):
        super().__init__()
        self.decay_range = decay_range
        self.model = None

    @torch.no_grad()
    def update(self, online_model: nn.Module, progress: float):
        if self.model is None:
            self.reset(online_model)
        progress = max(0.0, min(1.0, progress))
        decay = self.decay_range[0] + (self.decay_range[1] - self.decay_range[0]) * progress

        online_dict = dict(online_model.named_parameters())
        target_dict = dict(self.model.named_parameters())

        for name, target_param in target_dict.items():
            if name in online_dict:
                online_param = online_dict[name]
                if target_param.shape == online_param.shape:
                    target_param.data.lerp_(online_param.data, 1.0 - decay)
                else:
                    warnings.warn(
                        f"[Update] Shape mismatch for '{name}': "
                        f"{online_param.shape} -> {target_param.shape}"
                    )

    def reset(self, online_model: nn.Module):
        self.model = copy.deepcopy(online_model)
        self.model.requires_grad_(False)
        device = next(online_model.parameters()).device
        self.model.to(device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        if self.model is None:
            return self
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, *args, **kwargs):
        if self.model is None:
            return self
        self.model.train(False)
        return super().train(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if self.model is None:
            return self
        return self.model.state_dict(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'model':
            return super().__getattr__(name)
        if self.model is None:
            return super().__getattr__(name)
        return getattr(self.model, name)
