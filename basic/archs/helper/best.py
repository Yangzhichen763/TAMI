import torch
import torch.nn as nn
import copy
import warnings


class Best:
    def __init__(self, model: nn.Module, mode='all'):
        self.model = copy.deepcopy(model)
        self.metrics = []
        self.mode = mode
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model, current_metrics):
        if self.mode == 'all':
            if all(_m > m for _m, m in zip(current_metrics, self.metrics)):
                self._update(online_model)
                self.metrics = current_metrics
                return
        elif self.mode == 'any':
            if any(_m > m for _m, m in zip(current_metrics, self.metrics)):
                self._update(online_model)
                self.metrics = current_metrics
                return
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._setback(online_model)

    def _update(self, online_model):
        for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
            target_params.copy_(online_params)

    def _setback(self, online_model):
        for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
            online_params.copy_(target_params)


class BestModule(nn.Module):
    def __init__(self, model: nn.Module, mode='all'):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.metrics = []
        self.mode = mode
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model, current_metrics):
        if self.mode == 'all':
            if all(_m > m for _m, m in zip(current_metrics, self.metrics)):
                self._update(online_model)
                self.metrics = current_metrics
                return
        elif self.mode == 'any':
            if any(_m > m for _m, m in zip(current_metrics, self.metrics)):
                self._update(online_model)
                self.metrics = current_metrics
                return
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._setback(online_model)

    def _update(self, online_model):
        for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
            target_params.copy_(online_params)

    def _setback(self, online_model):
        for online_params, target_params in zip(online_model.parameters(), self.model.parameters()):
            online_params.copy_(target_params)


class LazyBestModule(nn.Module):
    def __init__(self, mode='all', larger_is_better=None):
        super().__init__()
        self.model = None
        self.metrics = []
        self.mode = mode
        self.larger_is_better = larger_is_better

    @torch.no_grad()
    def update(self, online_model, current_metrics):
        if self.model is None:
            self.reset(online_model, current_metrics)
            return True  # 初始化视为改进

        is_better = False
        if self.mode == 'all':
            is_better = all(self._compare_all(current_metrics))
        elif self.mode == 'any':
            is_better = any(self._compare_all(current_metrics))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if is_better:
            self._update(online_model)
            self.metrics = current_metrics
            return True  # 改进了
        elif not self.no_setback:
            self._setback(online_model)
        return False  # 没有改进

    @torch.no_grad()
    def reset(self, online_model: nn.Module, current_metrics):
        self.model = copy.deepcopy(online_model)
        self.model.requires_grad_(False)
        device = next(online_model.parameters()).device
        self.model.to(device)

        self.metrics = current_metrics

    def _compare_all(self, current_metrics):
        def _compare(a, b):
            if self.larger_is_better is None:
                return a > b
            elif self.larger_is_better:
                return a > b
            else:
                return a < b
        if isinstance(current_metrics, (int, float)):
            return _compare(current_metrics, self.metrics)
        elif isinstance(current_metrics, dict):
            return [_compare(_m, m) for _m, m in zip(list(current_metrics.values()), list(self.metrics.values()))]
        else:
            return [_compare(_m, m) for _m, m in zip(current_metrics, self.metrics)]

    def _update(self, online_model):
        print("[Update] Updating best model parameters.")
        online_dict = dict(online_model.named_parameters())
        target_dict = dict(self.model.named_parameters())

        for name, target_param in target_dict.items():
            if name in online_dict:
                online_param = online_dict[name]
                if target_param.shape == online_param.shape:
                    target_param.data.copy_(online_param.data)
                else:
                    warnings.warn(
                        f"[Update] Shape mismatch for '{name}': "
                        f"{online_param.shape} -> {target_param.shape}"
                    )
            # else:
            #     warnings.warn(f"[Update] Parameter '{name}' not found in online_model")

        # # 检查 online_model 中多出来的参数
        # for name in online_dict:
        #     if name not in target_dict:
        #         warnings.warn(f"[Update] Extra parameter '{name}' in online_model")

    def get_best_state_dict(self):
        """Return the state dict of the best model for saving."""
        if self.model is None:
            return None
        return self.model.state_dict()

    def get_best_metrics(self):
        """Return the best metrics recorded."""
        return self.metrics

    def _setback(self, online_model):
        print("[Setback] Reverting online model parameters to best model.")
        online_dict = dict(online_model.named_parameters())
        target_dict = dict(self.model.named_parameters())

        for name, online_param in online_dict.items():
            if name in target_dict:
                target_param = target_dict[name]
                if online_param.shape == target_param.shape:
                    online_param.data.copy_(target_param.data)
                else:
                    warnings.warn(
                        f"[Setback] Shape mismatch for '{name}': "
                        f"{target_param.shape} -> {online_param.shape}"
                    )
            # else:
            #     warnings.warn(f"[Setback] Parameter '{name}' not found in target model")

        # # 检查 target 中多出来的参数
        # for name in target_dict:
        #     if name not in online_dict:
        #         warnings.warn(f"[Setback] Extra parameter '{name}' in target model")
