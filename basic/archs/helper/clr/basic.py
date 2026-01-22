import torch
import torch.nn as nn


"""
Contrastive Visual Representation Learning (CRL)

推荐库 https://github.com/lightly-ai/lightly/tree/master，包含各种子监督方法
"""


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim, spatial=False, kernel_size=1):
        super().__init__()
        if spatial:
            # (B, C, H, W)
            self.net = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, projection_dim, kernel_size, padding=kernel_size // 2)
            )
        else:
            # (B, D)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim)
            )

    def forward(self, x):
        return self.net(x)

