import numpy as np

def make_synthetic(n_steps: int = 6000, n_features: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    periods = rng.uniform(40, 200, n_features)
    phases = rng.uniform(0, 2 * np.pi, n_features)
    X = np.sin(2 * np.pi * t[:, None] / periods + phases) + 0.1 * rng.standard_normal((n_steps, n_features))
    labels = np.zeros(n_steps, dtype=np.int64)

    for s in rng.choice(np.arange(500, n_steps - 50), 25, replace=False):
        X[s, rng.integers(n_features)] += rng.choice([-1, 1]) * rng.uniform(4, 6)
        labels[s] = 1

    for s in rng.choice(np.arange(500, n_steps - 100, 100), 6, replace=False):
        length = rng.integers(20, 60)
        ch = rng.choice(n_features, 3, replace=False)
        X[s:s + length, ch] = X[s, ch] + 0.05 * rng.standard_normal((length, 3)) + 1.5
        labels[s:s + length] = 2

    names = [f"sensor_{i}" for i in range(n_features)]

    return X.astype(np.float32), labels, ["normal", "point", "collective"], names
