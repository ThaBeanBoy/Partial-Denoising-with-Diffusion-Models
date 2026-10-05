"""Figures that show what the models actually do with the test windows.

    python scripts/make_figures.py
    python scripts/make_figures.py --diffusion-run experiments/diffusion_seed1 --baseline-run experiments/vae_seed1

Needs two trained and evaluated runs (see scripts/rq2_compare_baselines.py, which creates them).
Writes to reports/figures/:
  scores_timeline.png  anomaly score of every test window over time, diffusion vs a baseline
  reconstructions.png  partial-denoising reconstructions of a normal and an anomalous window
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _bootstrap import ROOT
from src.pipeline import load_data, load_trained

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "openstack.npz"))
ap.add_argument("--diffusion-run", default=str(ROOT / "experiments" / "diffusion_seed0"))
ap.add_argument("--baseline-run", default=str(ROOT / "experiments" / "autoencoder_seed0"))
ap.add_argument("--t-stars", type=int, nargs="+", default=[10, 30, 70], help="noise levels to compare")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

out_dir = ROOT / "reports" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
data = load_data(args.data)


def scores_timeline(run_dirs_and_titles):
    """Test-window scores on a log axis, red where the window overlaps a failure."""
    starts, labels = data["test_start"], data["test_labels"]
    end_of_window = starts + data["test"].shape[1] - 1
    t = data["times"][end_of_window].astype("datetime64[s]").astype(object)
    order = np.argsort(starts)

    fig, axes = plt.subplots(len(run_dirs_and_titles), 1, figsize=(9, 2 * len(run_dirs_and_titles)), sharex=True)
    for ax, (run_dir, title) in zip(np.atleast_1d(axes), run_dirs_and_titles):
        scores = np.load(Path(run_dir) / "results_scores.npz")
        threshold = np.percentile(scores["val"], 99)
        colours = np.where(labels[order] == 1, "tab:red", "tab:blue")
        ax.scatter(np.array(t)[order], scores["test"][order], s=3, c=colours, linewidths=0)
        ax.axhline(threshold, color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("score")
        ax.set_title(title, fontsize=9)
    axes[-1].set_xlabel("time of the window's last step")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "scores_timeline.png", dpi=200)
    plt.close(fig)


def reconstructions(run_dir, t_stars):
    """Noise a typical normal window and the highest-scoring collective one, then denoise them."""
    model, cfg = load_trained(run_dir, args.device)
    scores = np.load(Path(run_dir) / "results_scores.npz")["test"]
    types = data["test_types"]
    collective_id = list(data["type_names"]).index("collective")

    normal_idx = np.where(types == 0)[0]
    normal = normal_idx[np.argsort(scores[normal_idx])[len(normal_idx) // 4]]
    collective_idx = np.where(types == collective_id)[0]
    anomalous = collective_idx[np.argmax(scores[collective_idx])]

    windows = data["test"][[normal, anomalous]]
    x = torch.from_numpy(windows).float().transpose(1, 2)
    gen = torch.Generator().manual_seed(0)
    recon = {ts: model.reconstruct(x, ts, generator=gen).transpose(1, 2).numpy() for ts in t_stars}

    # plot the feature the model got most wrong on the anomalous window
    worst_err = ((recon[t_stars[len(t_stars) // 2]][1] - windows[1]) ** 2).mean(axis=0)
    feature = int(np.argmax(worst_err))
    name = str(data["feature_names"][feature])

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for ax, k, title in [(axes[0], 0, "normal window"), (axes[1], 1, "collective-anomaly window")]:
        ax.plot(windows[k][:, feature], "k-", lw=1.6, label="input $x_0$")
        for ts in t_stars:
            ax.plot(recon[ts][k][:, feature], lw=1, label=f"reconstruction $t^*={ts}$")
        ax.set_title(f"{title}: {name}", fontsize=9)
        ax.set_xlabel("time step (5 s)")
    axes[0].set_ylabel("z-score")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstructions.png", dpi=200)
    plt.close(fig)


scores_timeline([(args.diffusion_run, "Diffusion"), (args.baseline_run, "Baseline")])
reconstructions(args.diffusion_run, args.t_stars)
print(f"Saved scores_timeline.png and reconstructions.png to {out_dir}")
