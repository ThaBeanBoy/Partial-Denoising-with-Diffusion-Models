import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score

def threshold_from_normal(val_scores: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(val_scores, percentile))

def evaluate_scores(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    out = {"n": int(len(labels)), "n_anomalous": int(labels.sum())}

    if labels.min() == labels.max():
        out["warning"] = "only one class present; AUROC/F1 undefined"
        return out

    out["auroc"] = float(roc_auc_score(labels, scores))
    out["auprc"] = float(average_precision_score(labels, scores))

    p, r, _ = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1))
    out.update(best_f1=float(f1[i]), best_f1_precision=float(p[i]), best_f1_recall=float(r[i]))

    if threshold is not None:
        pred = (scores > threshold).astype(int)
        prec, rec, f, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        out.update(threshold=float(threshold), f1_val_threshold=float(f), precision=float(prec), recall=float(rec), specificity=float(tn / max(tn + fp, 1)))

    return out
