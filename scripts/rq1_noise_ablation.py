import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import ROOT
from src.pipeline import evaluate_model, load_data, load_trained

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "experiments" / "diffusion"))
ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "openstack.npz"))
ap.add_argument("--device", default="auto")
args = ap.parse_args()

model, cfg = load_trained(args.run, args.device)

data = load_data(args.data)
rows = []

for t in cfg["ablation"]["t_star_values"]:
    if t > cfg["model"]["T"]:
        continue

    o = evaluate_model(model, cfg, data, args.device, score_overrides={"t_star": t})["overall"]
    rows.append({"t_star": t, **{k: o.get(k) for k in
                ["auroc", "auprc", "best_f1", "f1_val_threshold", "recall", "specificity", "precision"]}})
    print(rows[-1])

df = pd.DataFrame(rows)
(ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / "reports" / "rq1_noise_ablation.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
for col, label in [("auroc", "AUROC"), ("recall", "Sensitivity (recall)"),
                   ("specificity", "Specificity"), ("f1_val_threshold", "F1")]:
    ax.plot(df["t_star"], df[col], marker="o", label=label)

ax.set_xlabel("t* (forward noising steps before denoising)")
ax.set_ylabel("score")
ax.set_title("RQ1: effect of diffusion noise level")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "reports" / "figures" / "rq1_noise_ablation.png", dpi=150)

print(df.to_string(index=False))
