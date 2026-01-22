import torch
import torch.nn as nn
import copy


class Clone:
    """Exponential Moving Average for target network"""
    def __init__(self, model: nn.Module, frozen=True):
        self.model = copy.deepcopy(model)
        self.frozen = frozen
        if frozen:
            self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model):
        # 只有被冻结的才能使用参数更新
        if self.frozen:
            for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
                target_params.copy_(online_params)


class CloneModule(nn.Module):
    def __init__(self, model: nn.Module, frozen=True):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.frozen = frozen
        if self.frozen:
            self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model: nn.Module, force=False):
        # 只有被冻结的才能使用参数更新
        if self.frozen:
            for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
                if not force and torch.equal(target_params, online_params):
                    continue
                target_params.copy_(online_params)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, *args, **kwargs):
        if self.frozen:
            self.model.train(False)
        return super().train(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class LazyCloneModule(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        self.model = None
        self.frozen = frozen

    @torch.no_grad()
    def update(self, online_model: nn.Module, force=False):
        if self.model is None:
            self.reset(online_model)

        # 只有被冻结的才能使用参数更新
        if self.frozen:
            for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
                if not force and torch.equal(target_params, online_params):
                    continue
                target_params.copy_(online_params)

    @torch.no_grad()
    def reset(self, online_model: nn.Module):
        self.model = copy.deepcopy(online_model)
        if self.frozen:
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
        if self.frozen:
            self.model.train(False)
        return super().train(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if self.model is None:
            return self
        return self.model.state_dict(*args, **kwargs)

    def __getattr__(self, name):
        model = super().__getattr__('model')
        if model is not None and hasattr(model, name):
            return getattr(model, name)

        return super().__getattr__(name)