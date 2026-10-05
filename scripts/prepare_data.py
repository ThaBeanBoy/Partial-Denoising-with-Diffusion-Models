import argparse
from _bootstrap import ROOT
from src.data.prepare import prepare
from src.utils.common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
ap.add_argument("--synthetic", action="store_true")
args = ap.parse_args()

prepare(load_config(args.config), ROOT, synthetic=args.synthetic)