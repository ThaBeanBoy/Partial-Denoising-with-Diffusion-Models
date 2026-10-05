import numpy as np
import torch

@torch.no_grad()
def score_windows(model, windows: np.ndarray, batch_size: int = 256, device: str = "cpu", seed: int = 0, **score_kwargs) -> np.ndarray:
    model.eval().to(device)

    gen = torch.Generator(device=device).manual_seed(seed)
    out = []

    for i in range(0, len(windows), batch_size):
        x = torch.from_numpy(windows[i:i + batch_size]).float().to(device).transpose(1, 2)

        if "t_star" in score_kwargs:
            s = model.score(x, generator=gen, **score_kwargs)

        else:
            s = model.score(x)

        out.append(s.cpu().numpy())

    return np.concatenate(out) if out else np.zeros(0)
