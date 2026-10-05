import argparse
from _bootstrap import ROOT
from src.pipeline import load_data, train_model
from src.utils.common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["diffusion", "autoencoder", "vae"])
ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "openstack.npz"))
ap.add_argument("--run-name", default=None)
ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="auto")
args = ap.parse_args()

cfg = load_config(ROOT / "configs" / f"{args.model}.yaml")
data = load_data(args.data)

# Train
run_dir = ROOT / "experiments" / (args.run_name or args.model)

print(f"Training {args.model} on {len(data['train'])} normal windows -> {run_dir}")

train_model(cfg, data, run_dir, device=args.device, seed=args.seed, epochs=args.epochs)
