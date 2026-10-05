import numpy as np

from .metrics import evaluate_scores

def per_type_metrics(scores, window_types, type_names, threshold=None) -> dict:
    scores = np.asarray(scores)
    window_types = np.asarray(window_types)
    normal = window_types == 0
    results = {}

    for type_id, name in enumerate(type_names):
        if type_id == 0:
            continue

        mask = window_types == type_id

        if mask.sum() == 0:
            continue

        sel = normal | mask
        results[name] = evaluate_scores(scores[sel], mask[sel].astype(int), threshold)

    return results