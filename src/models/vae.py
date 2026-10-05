import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import _mlp


class VAE(nn.Module):
    def __init__(self, n_features: int, window: int, hidden=(256, 128), latent: int = 32, beta: float = 1.0):
        super().__init__()
        d = n_features * window
        self.shape = (n_features, window)
        self.beta = beta
        self.encoder = nn.Sequential(_mlp([d, *hidden]), nn.ReLU())
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.decoder = _mlp([latent, *reversed(hidden), d])

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def decode(self, z):
        return self.decoder(z).view(-1, *self.shape)

    def loss(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        recon = F.mse_loss(self.decode(z), x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl / x[0].numel()

    @torch.no_grad()
    def score(self, x, **_):
        mu, _ = self.encode(x)
        return ((self.decode(mu) - x) ** 2).mean(dim=(1, 2))
