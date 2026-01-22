from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.archs.util import set_requires_grad


class MemoryEnhancedNet(nn.Module):
    def __init__(
            self,
            memory_trainable_only=False,
            without_module_init=False,
            **memory_blocks_dict
    ):
        # 适合用于多继承的情况，如果使用多继承，就可以设置 without_module_init=True，防止重复初始化
        if not without_module_init:
            super().__init__()

        # set attributes
        for name, module in memory_blocks_dict.items():
            setattr(self, name, module)

        # set parameters
        # self.memory_blocks = nn.ModuleList(
        #     [memory_block for memory_block in memory_blocks_dict.values()]
        # )
        self.memory_blocks = [memory_block for memory_block in memory_blocks_dict.values() if isinstance(memory_block, nn.Module)]
        self.memory_blocks_map = {name: module for name, module in memory_blocks_dict.items() if isinstance(module, nn.Module)}

        self.memory_trainable_only = memory_trainable_only
        if memory_trainable_only:
            set_requires_grad(self, False)
            self.unfrozen_memory_block()

    def frozen_memory_block(self):
        for memory_block in self.memory_blocks:
            set_requires_grad(memory_block, False)

    def unfrozen_memory_block(self):
        for memory_block in self.memory_blocks:
            set_requires_grad(memory_block, True)

    def reset_memory(self):
        for memory_block in self.memory_blocks:
            memory_block.reset_memory()

    def get_memory_parameters(self):
        memory_parameters = []
        for memory_block in self.memory_blocks:
            memory_parameters.extend(memory_block.parameters())
        return memory_parameters

    def get_non_memory_parameters(self):
        memory_parameters = self.get_memory_parameters()
        non_memory_parameters = []
        for name, param in self.named_parameters():
            if name not in memory_parameters:
                non_memory_parameters.append(param)
        return non_memory_parameters

    def forward_memory(self, x, *args, **kwargs):
        raise NotImplementedError()

    @contextmanager
    def memory_hooker(self):
        memory_outputs = {}

        def get_memory_hook(name):
            def hook(module, input, output):
                memory_outputs[name] = output.detach()

            return hook

        handles = [
            self.memory_block.register_forward_hook(get_memory_hook(name))
            for name, memory_block in self.memory_blocks_map
        ]

        try:
            yield memory_outputs
        finally:
            for handle in handles:
                handle.remove()

