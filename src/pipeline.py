import copy
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data.prepare import load_processed
from .evaluation.by_anomaly_type import per_type_metrics
from .evaluation.metrics import evaluate_scores, threshold_from_normal
from .models import build_model
from .scoring.partial_denoise import score_windows
from .utils.common import get_device, save_json, set_seed


def _batches(arr, batch_size, shuffle, rng):
    idx = rng.permutation(len(arr)) if shuffle else np.arange(len(arr))
    for i in range(0, len(arr), batch_size):
        yield torch.from_numpy(arr[idx[i:i + batch_size]]).float().transpose(1, 2)


def train_model(model_cfg: dict, data: dict, run_dir: Path, device="auto", seed=42, epochs=None, verbose=True):
    set_seed(seed)
    device = get_device(device)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train, val = data["train"], data["val"]
    n_features, window = train.shape[2], train.shape[1]
    model = build_model(model_cfg["name"], model_cfg, n_features, window).to(device)

    tc = model_cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0))
    rng = np.random.default_rng(seed)
    n_epochs = epochs or tc["epochs"]
    best, best_state, bad, history = float("inf"), None, 0, []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        model.train()
        tr_losses = []
        for x in _batches(train, tc["batch_size"], True, rng):
            loss = model.loss(x.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            torch.manual_seed(seed)  # same noise every epoch -> comparable val loss
            va_losses = [model.loss(x.to(device)).item()
                         for x in _batches(val, tc["batch_size"], False, rng)]

        tr_l = float(np.mean(tr_losses))
        va_l = float(np.mean(va_losses)) if va_losses else float("nan")
        history.append(dict(epoch=epoch, train_loss=tr_l, val_loss=va_l))
        if verbose:
            print(f"epoch {epoch:3d}  train {tr_l:.4f}  val {va_l:.4f}  ({time.time() - t0:.1f}s)")

        if va_l < best - 1e-5:
            best, best_state, bad = va_l, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc.get("patience", 10):
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"state_dict": model.state_dict(), "n_features": n_features, "window": window},
               run_dir / "model.pt")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(model_cfg, f)

    save_json(history, run_dir / "history.json")

    return model

def load_trained(run_dir: Path, device="auto"):
    run_dir = Path(run_dir)

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")

    model = build_model(cfg["name"], cfg, ckpt["n_features"], ckpt["window"])
    model.load_state_dict(ckpt["state_dict"])

    return model.to(get_device(device)).eval(), cfg

def evaluate_model(model, cfg: dict, data: dict, device="auto", percentile=99.0, score_overrides=None, seed=0) -> dict:
    device = get_device(device)
    kw = dict(cfg.get("score", {})) if cfg["name"] == "diffusion" else {}
    kw.update(score_overrides or {})

    val_scores = score_windows(model, data["val"], device=device, seed=seed, **kw)
    test_scores = score_windows(model, data["test"], device=device, seed=seed, **kw)

    thr = threshold_from_normal(val_scores, percentile)
    overall = evaluate_scores(test_scores, data["test_labels"], thr)
    by_type = per_type_metrics(test_scores, data["test_types"], list(data["type_names"]), thr)

    return dict(model=cfg["name"], score_kwargs=kw, threshold_percentile=percentile,
                overall=overall, by_type=by_type, _val_scores=val_scores, _test_scores=test_scores)


def save_eval(result: dict, out_dir: Path, tag="results"):
    out_dir = Path(out_dir)
    np.savez(out_dir / f"{tag}_scores.npz", val=result["_val_scores"], test=result["_test_scores"])
    save_json({k: v for k, v in result.items() if not k.startswith("_")}, out_dir / f"{tag}.json")


def load_data(path):
    return load_processed(path)
