import argparse
import json
from pathlib import Path
from _bootstrap import ROOT
from src.pipeline import evaluate_model, load_data, load_trained, save_eval

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "openstack.npz"))
ap.add_argument("--t-star", type=int, default=None, help="diffusion only: override t*")
ap.add_argument("--percentile", type=float, default=99.0, help="threshold = this percentile of normal validation scores")
ap.add_argument("--device", default="auto")

args = ap.parse_args()

model, cfg = load_trained(Path(args.run), args.device)
over = {"t_star": args.t_star} if (args.t_star and cfg["name"] == "diffusion") else None
res = evaluate_model(model, cfg, load_data(args.data), args.device, args.percentile, over)

save_eval(res, Path(args.run))
print(json.dumps({"overall": res["overall"], "by_type": res["by_type"]}, indent=2))