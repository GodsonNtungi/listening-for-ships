"""Quiet-floor predictor: per band, energy clearly above the window's own quietest minutes counts as ship."""
import numpy as np


def share(X, k=2.0, q=0, per_band=False):
    """X: (n, 15, 25) dB with NaN minutes. Floor = q-th quietest observed minute per band; excess over k * floor."""
    L = 10 ** (X.astype(np.float64) / 10)
    obs = np.isfinite(L).all(2)
    Ls = np.sort(np.where(obs[..., None], L, np.inf), 1)
    floor = np.take_along_axis(Ls, np.minimum(q, np.maximum(obs.sum(1) - 1, 0))[:, None, None].repeat(25, 2), 1)
    L0 = np.where(obs[..., None], L, 0)
    ex = np.clip(L0 - k * floor, 0, None)
    if per_band:
        return ex.sum(1) / L0.sum(1)
    return ex.sum((1, 2)) / L0.sum((1, 2))


if __name__ == "__main__":  # flat window -> 0; one loud minute -> its excess share
    X = np.zeros((2, 15, 25))
    X[1, 7] = 10 * np.log10(11)
    X[1, 3] = np.nan
    s = share(X, k=1.0)
    assert abs(s[0]) < 1e-9 and abs(s[1] - 10 / 24) < 1e-9, s
    print("ok", s.round(3))
