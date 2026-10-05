# Diffusion-Based Anomaly Detection for Multivariate Sensor Time-Series

IT18X57 Advanced Artificial Intelligence research project.
Theme: Generative Modelling for Anomaly Detection.

Repository: https://github.com/ThaBeanBoy/Partial-Denoising-with-Diffusion-Models

A diffusion model (DDPM) is trained only on normal windows of multivariate system
metrics. At test time each window is noised forward to step **t\*** and then
denoised back. The reconstruction error is the anomaly score. The approach is
compared with an autoencoder and a VAE baseline.

## Dataset

Nedelkoski, S., Bogatinovski, J., Mandapati, A. K., Becker, S., Cardoso, J., & Kao, O. (2019).
*Multi-Source Distributed System Data for AI-powered Analytics* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.3549604 (CC BY 4.0)

The data is not stored in the repository (`data/raw/` is git-ignored). Download it first
(see [Getting the data](#getting-the-data)). It ends up in `data/raw/sequential_data/`, taken
from `sequential_data.zip`:

| Part | Used for |
|---|---|
| `metrics/wally113…124_metrics.csv` | Model input. There are 5 hosts, and each has cpu.user, mem.used and load.min1/5/15, sampled at about 1 Hz. |
| `reports/j_*.json` (Rally) | Ground truth. Each failed workload iteration marks an injected fault. |
| `IMPORTANT_experiment_start_end_data.txt` | The experiment window and clock offset. |

### Getting the data

```bash
python -m src.data.download
```

This downloads `sequential_data.zip` from Zenodo (about 315 MB) and extracts only the parts used
here (about 80 MB): `metrics/`, `reports/` and the `IMPORTANT_*.txt` file. Everything lands in
`data/raw/sequential_data/`.

If the download is blocked, download `sequential_data.zip` by hand from
https://zenodo.org/records/3549604 and extract it from wherever you saved it:

```bash
python -m src.data.download --zip path/to/sequential_data.zip
```

### How labels are built (`src/data/openstack.py`)

- Rally timestamps are in UTC. The metric timestamps are UTC+1, even though the files label them "CEST". Rally times are therefore shifted by +1 h.
- Only the interval from 2019-11-19 18:38:39 to 2019-11-20 02:30:00 on the metric clock is kept, as the dataset authors instruct.
- The span `[timestamp, timestamp + duration]` of every failed iteration is marked anomalous. That gives 201 failures across Glance, Neutron and Nova.
- For RQ3, failures closer than 30 s are merged into segments. There are 8 segments:
  - **point** anomalies are short bursts of 10–25 s from Glance and Neutron, each shorter than one window.
  - **collective** anomalies are the Nova failures, which last about 5 minutes.
  - Set `labels.type_by: service` in `configs/data.yaml` to group by service instead.

### Split (`src/data/windows.py`)

The faults are bunched in time, so a plain chronological split would leave almost
no anomalies in the test set. Instead the timeline is cut into 10-minute blocks:

- Blocks that contain a fault go to **test**.
- Normal blocks are split 60/15/25 into **train**, **val** and **test**, using a fixed seed.
- Windows never cross a block boundary.
- Normalisation statistics come from the training blocks only.

Current result: 25 features, 2-minute windows (24 × 5 s). There are 2,231 train
windows, 582 val windows and 1,746 test windows, of which 17.5% are anomalous.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests -q        # quick check (synthetic data)
```

## Running the experiments

Run the steps in this order. Each one uses the output of the one before it.

```bash
python -m src.data.download                    # 1. get the raw data into data/raw/ (see above)
python scripts/prepare_data.py                 # 2. raw -> data/processed/openstack.npz

python scripts/train.py --model diffusion      # 3. or autoencoder / vae
python scripts/evaluate.py --run experiments/diffusion

python scripts/rq1_noise_ablation.py           # RQ1: sweep t*  -> reports/rq1_noise_ablation.csv + figure
python scripts/rq2_compare_baselines.py        # RQ2: 3 models x 3 seeds -> reports/rq2_comparison.csv
python scripts/rq3_anomaly_types.py            # RQ3: point vs collective -> reports/rq3_by_type.csv
```

Everything runs on a CPU. One training run takes seconds for AE/VAE and about a
minute for diffusion. A GPU is used automatically if available.

## Metrics

- **AUROC** and **AUPRC** do not depend on a threshold.
- **F1 / precision / recall / specificity** use a threshold set at the 99th percentile of scores on normal validation windows. This is the number you could actually deploy.
- **best_f1** is the best F1 over all thresholds on the test set. It is optimistic and common in the literature, so report it next to the thresholded F1, not instead of it.

## Layout

```
configs/            data.yaml, diffusion.yaml (incl. t* and ablation grid), autoencoder.yaml, vae.yaml
data/raw/           the dataset (metrics + Rally reports)
data/interim/       merged 5 s metrics with labels, list of failed iterations
data/processed/     normalised windows + labels (.npz)
src/data/           download, OpenStack loader + labelling, block split + windowing, synthetic data
src/models/         denoiser (dilated 1D-conv, timestep-conditioned), DDPM, autoencoder, VAE
src/scoring/        batch scoring (partial denoising for the diffusion model)
src/evaluation/     metrics, per-anomaly-type breakdown
src/pipeline.py     training loop (early stopping on val loss) and evaluation
scripts/            prepare / train / evaluate / rq1 / rq2 / rq3
experiments/        checkpoints and results per run
reports/            result tables and figures for the write-up
tests/              end-to-end checks on synthetic data
```
