import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims):
    layers = []

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int, window: int, hidden=(256, 128), latent: int = 32):
        super().__init__()

        d = n_features * window
        self.shape = (n_features, window)
        self.encoder = _mlp([d, *hidden, latent])
        self.decoder = _mlp([latent, *reversed(hidden), d])

    def forward(self, x):
        z = self.encoder(x.flatten(1))
        return self.decoder(z).view(-1, *self.shape)

    def loss(self, x):
        return F.mse_loss(self(x), x)

    @torch.no_grad()
    def score(self, x, **_):
        return ((self(x) - x) ** 2).mean(dim=(1, 2))
