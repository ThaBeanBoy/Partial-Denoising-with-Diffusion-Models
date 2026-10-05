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

python scripts/make_figures.py                 # score timeline + reconstruction figures -> reports/figures/
```

`make_figures.py` needs the runs created by `rq2_compare_baselines.py` (it uses `diffusion_seed0` and
`autoencoder_seed0` by default; pick others with `--diffusion-run` and `--baseline-run`).

Everything runs on a CPU. One training run takes seconds for AE/VAE and about a
minute for diffusion. A GPU is used automatically if available.

## Example outputs

The tables and figures below come from the runs used in the write-up (OpenStack data, 1,746 test
windows, 305 of them anomalous). The files themselves are committed in `reports/` (CSV tables,
`run_log.txt` and `figures/`), and the commands above re-create them.

**Training** (`python scripts/train.py --model diffusion`) prints one line per epoch:

```
Training diffusion on 2231 normal windows -> experiments/diffusion
epoch   1  train 0.9564  val 0.8856  (0.8s)
epoch   2  train 0.7962  val 0.7021  (0.6s)
...
epoch  60  train 0.1517  val 0.1999  (0.6s)
```

**Evaluation** (`python scripts/evaluate.py --run experiments/diffusion`) prints the metrics as JSON
(excerpt, rounded):

```
"overall": {"n": 1746, "n_anomalous": 305, "auroc": 0.8385, "auprc": 0.4276, "best_f1": 0.5329,
            "threshold": 0.1409, "f1_val_threshold": 0.0179, "precision": 0.0968,
            "recall": 0.0098, "specificity": 0.9806}
```

**RQ2**: diffusion vs baselines, mean ± std over seeds 0-2 (`reports/rq2_comparison.csv`):

| Model | AUROC | AUPRC | best F1 | F1 at val threshold |
|---|---|---|---|---|
| Diffusion | 0.844 ± 0.011 | 0.433 ± 0.014 | 0.552 ± 0.011 | 0.196 ± 0.038 |
| Autoencoder | 0.808 ± 0.010 | 0.362 ± 0.024 | 0.535 ± 0.003 | 0.336 ± 0.013 |
| VAE | 0.806 ± 0.005 | 0.356 ± 0.018 | 0.528 ± 0.006 | 0.347 ± 0.008 |

**RQ1**: effect of the noise level t\* on one diffusion model (`reports/rq1_noise_ablation.csv`):

| t\* | 2 | 5 | 10 | 20 | 30 | 40 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| AUROC | 0.851 | 0.853 | 0.852 | 0.844 | 0.838 | 0.837 | 0.835 | 0.824 | 0.610 |
| F1 at val threshold | 0 | 0 | 0 | 0 | 0.018 | 0.092 | 0.138 | 0.225 | 0.006 |

**RQ3**: AUROC by anomaly type, mean over seeds 0-2 (`reports/rq3_by_type.csv`):

| Type | Diffusion | Autoencoder | VAE |
|---|---|---|---|
| Collective | 0.876 | 0.802 | 0.797 |
| Point | 0.809 | 0.815 | 0.815 |

**Figures** (`python scripts/make_figures.py`, written to `reports/figures/`):

- `scores_timeline.png`: the anomaly score of every test window over time for the diffusion model and a baseline. Red windows overlap a labelled failure and the dashed line is the validation threshold.
- `reconstructions.png`: a normal window and a collective-anomaly window, noised to several t\* values and denoised back. The normal window is reproduced closely, while the anomalous one is pulled towards normal behaviour, which is what makes its error large.
- `rq1_noise_ablation.png` (from `rq1_noise_ablation.py`): AUROC, sensitivity, specificity and F1 against t\*.

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
scripts/            prepare / train / evaluate / rq1 / rq2 / rq3 / make_figures
experiments/        checkpoints and results per run
reports/            result tables, figures and the run log for the write-up (committed)
tests/              end-to-end checks on synthetic data
```
