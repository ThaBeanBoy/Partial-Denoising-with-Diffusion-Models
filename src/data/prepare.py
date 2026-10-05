from pathlib import Path
import numpy as np
import pandas as pd
from . import openstack
from .synthetic import make_synthetic
from .windows import assign_blocks, fit_normaliser, sliding_windows


def prepare(cfg: dict, root: Path, synthetic: bool = False) -> Path:
    lab = cfg["labels"]

    if synthetic or cfg["dataset"] == "synthetic":
        X, labels, type_names, feature_names = make_synthetic(seed=cfg["seed"])
        times = np.arange(len(X)).astype("datetime64[s]")
        steps_per_block = 200
        out_path = root / "data" / "processed" / "synthetic.npz"

    else:
        raw = root / cfg["raw_dir"]
        df = openstack.load_metrics(raw, cfg["resample_seconds"])
        intervals = openstack.load_failure_intervals(raw)

        labels, type_names = openstack.build_point_labels(df.index, intervals, type_by=lab["type_by"], merge_gap_seconds=lab["merge_gap_seconds"], point_max_seconds=lab["point_max_seconds"], pad_seconds=lab["pad_seconds"])
        X = df.to_numpy(np.float32)

        feature_names = list(df.columns)
        times = df.index.to_numpy()
        steps_per_block = int(cfg["block_minutes"] * 60 / cfg["resample_seconds"])
        out_path = root / cfg["processed_path"]
        interim = root / "data" / "interim"

        interim.mkdir(parents=True, exist_ok=True)
        df.assign(label=labels).to_csv(interim / "metrics_merged_labelled.csv")
        intervals.to_csv(interim / "failed_iterations.csv", index=False)

    block_id, split = assign_blocks(len(X), steps_per_block, labels, cfg["train_frac"], cfg["val_frac"], cfg["seed"])

    train_blocks = np.where(split == "train")[0]
    val_blocks = np.where(split == "val")[0]
    test_blocks = np.where(split == "test")[0]

    train_mask = np.isin(block_id, train_blocks)
    mean, std, keep = fit_normaliser(X[train_mask])
    Xn = np.clip((X - mean) / std, -10, 10)[:, keep]
    feature_names = [f for f, k in zip(feature_names, keep) if k]

    w, s = cfg["window"], cfg["stride"]
    tr, tr_y, _, _ = sliding_windows(Xn, labels, block_id, train_blocks, w, s)
    va, va_y, _, _ = sliding_windows(Xn, labels, block_id, val_blocks, w, s)
    te, te_y, te_t, te_start = sliding_windows(Xn, labels, block_id, test_blocks, w, s)

    assert tr_y.sum() == 0 and va_y.sum() == 0, "train/val must contain only normal windows"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path, train=tr, val=va, test=te, test_labels=te_y, test_types=te_t,
        test_start=te_start, type_names=np.array(type_names), feature_names=np.array(feature_names),
        series=Xn.astype(np.float32), point_labels=labels, times=times.astype("datetime64[s]"),
        block_split=np.array(split, dtype=str))

    print(f"Features: {len(feature_names)}  |  timesteps: {len(X)}  |  window: {w}")
    print(f"Windows  train={len(tr)}  val={len(va)}  test={len(te)} "
          f"(anomalous {int(te_y.sum())}, {te_y.mean():.1%})")

    for k, name in enumerate(type_names[1:], 1):
        print(f"  test windows of type '{name}': {int((te_t == k).sum())}")

    print(f"Saved {out_path}")

    return out_path


def load_processed(path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
