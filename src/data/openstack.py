import json
from pathlib import Path
import numpy as np
import pandas as pd

METRIC_CLOCK_OFFSET = pd.Timedelta(hours=1)
EXPERIMENT_START = pd.Timestamp("2019-11-19 18:38:39")
EXPERIMENT_END = pd.Timestamp("2019-11-20 02:30:00")

SERVICE_BY_REPORT = {
    "j_image_report.json": "glance_image",
    "j_network_output.json": "neutron_network",
    "j_boot_delete_report.json": "nova_boot_delete",
}

def load_metrics(raw_dir: Path, resample_seconds: int = 5, drop_columns=("load.cpucore",)) -> pd.DataFrame:
    frames = []

    for f in sorted((Path(raw_dir) / "metrics").glob("*_metrics.csv")):
        host = f.stem.replace("_metrics", "")

        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["now"].str.replace(r"\s*CES?T$", "", regex=True))
        df = df.drop(columns=["now", *[c for c in drop_columns if c in df.columns]])
        df = df.groupby("time").mean().sort_index()
        df = df[(df.index >= EXPERIMENT_START) & (df.index <= EXPERIMENT_END)]
        df = df.resample(f"{resample_seconds}s").mean()
        df.columns = [f"{host}.{c}" for c in df.columns]

        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No *_metrics.csv found under {Path(raw_dir) / 'metrics'}")

    merged = pd.concat(frames, axis=1).sort_index()
    merged = merged.interpolate(limit_direction="both").ffill().bfill()

    return merged


def load_failure_intervals(raw_dir: Path) -> pd.DataFrame:
    rows = []

    for f in sorted((Path(raw_dir) / "reports").glob("*.json")):
        report = json.loads(f.read_text())
        service = SERVICE_BY_REPORT.get(f.name, f.stem)
        for task in report["tasks"]:
            for sub in task["subtasks"]:
                for wl in sub["workloads"]:
                    for it in wl["data"]:
                        if not it["error"]:
                            continue

                        start = pd.to_datetime(it["timestamp"], unit="s") + METRIC_CLOCK_OFFSET
                        end = start + pd.to_timedelta(max(it["duration"], 1.0), unit="s")
                        rows.append(dict(start=start, end=end, service=service,error=it["error"][0], scenario=sub["title"]))

    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def merge_into_segments(intervals: pd.DataFrame, merge_gap_seconds: float) -> pd.DataFrame:
    segs = []
    gap = pd.Timedelta(seconds=merge_gap_seconds)

    for _, r in intervals.iterrows():
        if segs and r.start <= segs[-1]["end"] + gap:
            segs[-1]["end"] = max(segs[-1]["end"], r.end)
            segs[-1]["n_failures"] += 1
            segs[-1]["services"].add(r.service)

        else:
            segs.append(dict(start=r.start, end=r.end, n_failures=1, services={r.service}))

    out = pd.DataFrame(segs)
    out["duration_s"] = (out["end"] - out["start"]).dt.total_seconds()
    out["services"] = out["services"].apply(lambda s: "+".join(sorted(s)))

    return out


def build_point_labels(index: pd.DatetimeIndex, intervals: pd.DataFrame, type_by: str = "segment", merge_gap_seconds: float = 30, point_max_seconds: float = 120, pad_seconds: float = 0):
    labels = np.zeros(len(index), dtype=np.int64)
    pad = pd.Timedelta(seconds=pad_seconds)

    if type_by == "segment":
        type_names = ["normal", "point", "collective"]
        segs = merge_into_segments(intervals, merge_gap_seconds)

        for _, s in segs.iterrows():
            k = 1 if s.duration_s <= point_max_seconds else 2
            mask = (index >= s.start - pad) & (index <= s.end + pad)
            labels[mask] = np.maximum(labels[mask], k)

    elif type_by == "service":
        services = sorted(intervals["service"].unique())
        type_names = ["normal", *services]

        for _, r in intervals.iterrows():
            k = type_names.index(r.service)
            mask = (index >= r.start - pad) & (index <= r.end + pad)
            labels[mask] = k

    else:
        raise ValueError(f"type_by must be 'segment' or 'service', got {type_by}")

    return labels, type_names
