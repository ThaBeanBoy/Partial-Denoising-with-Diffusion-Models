from .autoencoder import AutoEncoder
from .denoiser import Denoiser
from .diffusion import GaussianDiffusion
from .vae import VAE


def build_model(name: str, cfg: dict, n_features: int, window: int):
    m = cfg.get("model", {})
    if name == "diffusion":
        net = Denoiser(n_features, hidden=m.get("hidden", 64), n_blocks=m.get("n_blocks", 6),
                       emb_dim=m.get("emb_dim", 128))
        return GaussianDiffusion(net, T=m.get("T", 100), schedule=m.get("schedule", "cosine"))
    if name == "autoencoder":
        return AutoEncoder(n_features, window, hidden=tuple(m.get("hidden", (256, 128))),
                           latent=m.get("latent", 32))
    if name == "vae":
        return VAE(n_features, window, hidden=tuple(m.get("hidden", (256, 128))),
                   latent=m.get("latent", 32), beta=m.get("beta", 1.0))
    raise ValueError(f"Unknown model: {name}")
