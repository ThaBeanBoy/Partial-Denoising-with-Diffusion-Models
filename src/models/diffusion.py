import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_betas(T: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02 * 1000 / T, T)

    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]

        return betas.clamp(max=0.999).float()

    raise ValueError(f"Unknown schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    def __init__(self, denoiser: nn.Module, T: int = 100, schedule: str = "cosine"):
        super().__init__()

        self.denoiser = denoiser
        self.T = T

        betas = make_betas(T, schedule)
        alphas = 1.0 - betas
        alpha_bar = torch.cat([torch.ones(1), torch.cumprod(alphas, 0)])

        self.register_buffer("betas", torch.cat([torch.zeros(1), betas]))
        self.register_buffer("alphas", torch.cat([torch.ones(1), alphas]))
        self.register_buffer("alpha_bar", alpha_bar)

    # ---------- forward process ----------
    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(1, self.T + 1, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return F.mse_loss(self.denoiser(xt, t), noise)

    # ---------- reverse process ----------
    def _predict_x0(self, xt, t_int, eps):
        ab = self.alpha_bar[t_int]
        return (xt - (1 - ab).sqrt() * eps) / ab.sqrt()

    @torch.no_grad()
    def reconstruct(self, x0, t_star: int, sampler: str = "ddim", generator=None):
        t_star = int(t_star)

        if not 1 <= t_star <= self.T:
            raise ValueError(f"t_star must be in [1, {self.T}], got {t_star}")

        b = x0.shape[0]
        noise = torch.randn(x0.shape, generator=generator, device=x0.device)
        x = self.q_sample(x0, torch.full((b,), t_star, device=x0.device, dtype=torch.long), noise)

        for t in range(t_star, 0, -1):
            tt = torch.full((b,), t, device=x0.device, dtype=torch.long)
            eps = self.denoiser(x, tt)
            x0_hat = self._predict_x0(x, t, eps).clamp(-10, 10)
            ab_prev = self.alpha_bar[t - 1]

            if sampler == "ddim":
                x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps

            elif sampler == "ddpm":
                ab = self.alpha_bar[t]
                coef0 = ab_prev.sqrt() * self.betas[t] / (1 - ab)
                coeft = self.alphas[t].sqrt() * (1 - ab_prev) / (1 - ab)
                mean = coef0 * x0_hat + coeft * x

                if t > 1:
                    var = self.betas[t] * (1 - ab_prev) / (1 - ab)
                    z = torch.randn(x.shape, generator=generator, device=x.device)
                    x = mean + var.sqrt() * z

                else:
                    x = mean

            else:
                raise ValueError(f"Unknown sampler: {sampler}")

        return x

    @torch.no_grad()
    def score(self, x0, t_star: int, n_samples: int = 4, sampler: str = "ddim", generator=None):
        errs = []

        for _ in range(n_samples):
            recon = self.reconstruct(x0, t_star, sampler=sampler, generator=generator)
            errs.append(((recon - x0) ** 2).mean(dim=(1, 2)))

        return torch.stack(errs).mean(0)
