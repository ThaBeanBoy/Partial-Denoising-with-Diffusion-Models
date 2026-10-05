import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import make_synthetic  # noqa: E402
from src.data.windows import assign_blocks, sliding_windows  # noqa: E402
from src.models import build_model  # noqa: E402
from src.pipeline import evaluate_model, train_model  # noqa: E402


def _tiny_data():
    X, labels, type_names, _ = make_synthetic(n_steps=2000, n_features=4)
    X = (X - X.mean(0)) / X.std(0)
    block_id, split = assign_blocks(len(X), 100, labels, 0.6, 0.2, seed=0)
    tr, tr_y, _, _ = sliding_windows(X, labels, block_id, np.where(split == "train")[0], 16, 4)
    va, va_y, _, _ = sliding_windows(X, labels, block_id, np.where(split == "val")[0], 16, 4)
    te, te_y, te_t, _ = sliding_windows(X, labels, block_id, np.where(split == "test")[0], 16, 4)
    return dict(train=tr, val=va, test=te, test_labels=te_y, test_types=te_t,
                type_names=np.array(type_names)), tr_y, va_y


def test_train_and_val_are_normal_only():
    _, tr_y, va_y = _tiny_data()
    assert tr_y.sum() == 0 and va_y.sum() == 0


def test_partial_denoise_shapes_and_limits():
    cfg = {"model": {"T": 20, "hidden": 16, "n_blocks": 2}}
    model = build_model("diffusion", cfg, n_features=4, window=16)
    x = torch.randn(3, 4, 16)
    for sampler in ["ddim", "ddpm"]:
        assert model.reconstruct(x, t_star=5, sampler=sampler).shape == x.shape
    assert model.score(x, t_star=5, n_samples=2).shape == (3,)
    # alpha_bar is 1 at t=0 and decreasing
    ab = model.alpha_bar
    assert torch.isclose(ab[0], torch.tensor(1.0)) and (ab[1:] < ab[:-1]).all()


def test_end_to_end(tmp_path):
    data, _, _ = _tiny_data()
    for name, extra in [("autoencoder", {}), ("vae", {}),
                        ("diffusion", {"score": {"t_star": 5, "n_samples": 1, "sampler": "ddim"}})]:
        cfg = {"name": name, "model": {"T": 20, "hidden": 16, "n_blocks": 2} if name == "diffusion" else {},
               "train": {"epochs": 2, "batch_size": 64, "lr": 1e-3, "patience": 5}, **extra}
        model = train_model(cfg, data, tmp_path / name, device="cpu", verbose=False)
        res = evaluate_model(model, cfg, data, device="cpu")
        assert 0.0 <= res["overall"]["auroc"] <= 1.0
