import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/3549604/files/sequential_data.zip?download=1"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
KEEP = ("sequential_data/metrics/", "sequential_data/reports/", "sequential_data/IMPORTANT")


def download(dest: Path):
    print(f"Downloading {URL}\n  -> {dest} (approx. 315 MB)")

    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            sys.stdout.write(f"\r  {min(done, total) / 1e6:7.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, dest, hook)
    print()


def extract(zip_path: Path, out_dir: Path = RAW):
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith(KEEP) and not m.endswith("/")]
        for m in members:
            z.extract(m, out_dir)
    print(f"Extracted {len(members)} files into {out_dir / 'sequential_data'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="Use an already-downloaded sequential_data.zip")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = args.zip or RAW / "sequential_data.zip"

    if not zip_path.exists():
        try:
            download(zip_path)
        except Exception as e:  # network blocked etc.
            print(f"Download failed: {e}\nDownload it manually from "
                  "https://zenodo.org/records/3549604 and run:\n"
                  "  python -m src.data.download --zip path/to/sequential_data.zip")
            sys.exit(1)

    extract(zip_path)

if __name__ == "__main__":
    main()
