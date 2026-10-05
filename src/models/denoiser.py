import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]

        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, emb_dim: int, dilation: int):
        super().__init__()

        groups = 8 if channels % 8 == 0 else 1

        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.emb = nn.Linear(emb_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))

        return x + h

class Denoiser(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_blocks: int = 6, emb_dim: int = 128):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.inp = nn.Conv1d(n_features, hidden, 1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, emb_dim, dilation=2 ** (i % 4)) for i in range(n_blocks)]
        )
        self.out = nn.Conv1d(hidden, n_features, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        emb = self.time_emb(t)
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out(F.silu(h))
