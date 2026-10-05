import numpy as np

def assign_blocks(n_steps: int, block_len: int, labels: np.ndarray, train_frac: float, val_frac: float, seed: int):
    n_blocks = int(np.ceil(n_steps / block_len))
    block_id = np.arange(n_steps) // block_len
    has_anom = np.array([labels[block_id == b].any() for b in range(n_blocks)])
    normal_blocks = np.where(~has_anom)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(normal_blocks)
    n_tr = int(round(train_frac * len(normal_blocks)))
    n_va = int(round(val_frac * len(normal_blocks)))
    split = np.full(n_blocks, "test", dtype=object)
    split[normal_blocks[:n_tr]] = "train"
    split[normal_blocks[n_tr:n_tr + n_va]] = "val"

    return block_id, split


def fit_normaliser(X: np.ndarray, eps: float = 1e-8):
    mean, std = X.mean(0), X.std(0)
    keep = std > eps
    return mean, np.where(keep, std, 1.0), keep

def sliding_windows(X: np.ndarray, labels: np.ndarray, block_id: np.ndarray, blocks, window: int, stride: int):
    W, y, types, starts = [], [], [], []

    for b in blocks:
        idx = np.where(block_id == b)[0]

        for s in range(idx[0], idx[-1] - window + 2, stride):
            lab = labels[s:s + window]
            W.append(X[s:s + window])
            nz = lab[lab > 0]
            types.append(int(np.bincount(nz).argmax()) if len(nz) else 0)
            y.append(int(len(nz) > 0))
            starts.append(s)

    if not W:
        return (np.zeros((0, window, X.shape[1]), np.float32), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int))

    return (np.stack(W).astype(np.float32), np.array(y), np.array(types), np.array(starts))
