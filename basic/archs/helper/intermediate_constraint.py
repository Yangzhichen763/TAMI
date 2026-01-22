import torch
import torch.nn as nn


class ICHelper(nn.Module):
    def __init__(self):
        super().__init__()

    def init(self, num_constraint, layer_type, **layer_kwargs):
        self.num_constraint = num_constraint

        proj_layers = [self.make_layer(layer_type, **layer_kwargs) for _ in range(num_constraint)]
        self.proj_layers = nn.ModuleList(proj_layers)
