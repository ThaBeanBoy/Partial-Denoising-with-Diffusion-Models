import argparse
from pathlib import Path
import pandas as pd
from _bootstrap import ROOT
from src.pipeline import evaluate_model, load_data, load_trained, save_eval, train_model
from src.utils.common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "openstack.npz"))
ap.add_argument("--models", nargs="+", default=["diffusion", "autoencoder", "vae"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--device", default="auto")
args = ap.parse_args()

data = load_data(args.data)
rows = []

for name in args.models:
    for seed in args.seeds:
        run = ROOT / "experiments" / f"{name}_seed{seed}"

        if not (run / "model.pt").exists():
            print(f"--- training {name} seed {seed}")
            train_model(load_config(ROOT / "configs" / f"{name}.yaml"), data, run,
                        args.device, seed, args.epochs, verbose=False)

        model, cfg = load_trained(run, args.device)
        res = evaluate_model(model, cfg, data, args.device, seed=seed)
        save_eval(res, run)
        rows.append({"model": name, "seed": seed, **res["overall"]})

        print(name, seed, {k: round(res["overall"].get(k, float("nan")), 4)
                           for k in ["auroc", "auprc", "best_f1", "f1_val_threshold"]})

df = pd.DataFrame(rows)
df.to_csv(ROOT / "reports" / "rq2_runs.csv", index=False)
metrics = ["auroc", "auprc", "best_f1", "f1_val_threshold", "precision", "recall", "specificity"]

summary = df.groupby("model")[metrics].agg(["mean", "std"]).round(4)
summary.to_csv(ROOT / "reports" / "rq2_comparison.csv")

print(summary.to_string())
