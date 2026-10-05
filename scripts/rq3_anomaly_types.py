import json
import pandas as pd
from _bootstrap import ROOT

rows = []

for res_file in sorted((ROOT / "experiments").glob("*_seed*/results.json")):
    res = json.loads(res_file.read_text())

    for type_name, m in res["by_type"].items():
        rows.append({"run": res_file.parent.name, "model": res["model"], "anomaly_type": type_name,
                     "n_windows": m.get("n_anomalous"), "auroc": m.get("auroc"),
                     "recall": m.get("recall"), "best_f1": m.get("best_f1")})

if not rows:
    raise SystemExit("No results found. Run scripts/rq2_compare_baselines.py first.")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "reports" / "rq3_runs.csv", index=False)

summary = df.groupby(["anomaly_type", "model"])[["auroc", "recall", "best_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(ROOT / "reports" / "rq3_by_type.csv")

print(summary.to_string())