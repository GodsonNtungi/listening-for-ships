"""Split each window into 'ship' and 'sea' parts with fixed spectral shapes (semi-supervised NMF).

Ship shapes come from VTUAD minute spectra, sea shapes from SA background minutes; each is
learned on its own, so no assumption about mixing levels, pass shapes or empty windows enters.
Itakura-Saito divergence: scale-invariant, the usual choice for power spectra.
"""
import numpy as np
from sklearn.decomposition import NMF

EPS = 1e-12


def minute_spectra(clips):
    out = [c[:len(c) // 30 * 30].reshape(-1, 30, 25).mean(1) for c in clips if len(c) >= 30]
    return np.concatenate(out)


def learn(spectra, k, seed=0):
    """spectra: (m, 25) linear power -> (k, 25) unit-sum shapes."""
    s = spectra / spectra.sum(1, keepdims=True)
    m = NMF(k, beta_loss="itakura-saito", solver="mu", init="random", max_iter=600, random_state=seed, tol=1e-5)
    m.fit(s + EPS)
    d = m.components_
    return d / d.sum(1, keepdims=True)


def learn_with_fixed(spectra, k, D_fixed, iters=400, seed=0):
    """Learn k new shapes that explain what the fixed shapes cannot (e.g. ship minus its own ambient)."""
    rng = np.random.default_rng(seed)
    V = spectra / spectra.sum(1, keepdims=True) + EPS
    Dn = rng.random((k, 25)) + 0.1
    Dn /= Dn.sum(1, keepdims=True)
    A = np.full((len(V), k + len(D_fixed)), 25 / (k + len(D_fixed)))
    for _ in range(iters):
        D = np.concatenate([Dn, D_fixed])
        R = A @ D + EPS
        A *= ((V / R ** 2) @ D.T) / ((1 / R) @ D.T + EPS)
        R = A @ D + EPS
        Dn *= (A[:, :k].T @ (V / R ** 2)) / (A[:, :k].T @ (1 / R) + EPS)
        Dn /= Dn.sum(1, keepdims=True)
    return Dn


def split(X, D_ship, D_sea, iters=150, ship_penalty=0.0, sea_smooth=0.0, steady_q=None, return_bands=False):
    """X: (n, 15, 25) dB with NaN minutes. Returns per-window ship share and per-minute ship fraction."""
    L = 10 ** (X.astype(np.float64) / 10)
    obs = np.isfinite(L).all(2)
    scale = np.nanmean(L, (1, 2), keepdims=True)
    V = np.where(obs[..., None], L / scale, 1.0)
    D = np.concatenate([D_ship, D_sea])  # (K, 25)
    ks = len(D_ship)
    A = np.full(V.shape[:2] + (len(D),), 25 / len(D))  # V is scaled to mean 1, shapes sum to 1
    for _ in range(iters):  # multiplicative updates for IS divergence, shapes fixed
        R = A @ D + EPS
        pen = np.zeros(len(D))
        pen[:ks] = ship_penalty  # L1 cost on ship activations: only credit ships when they clearly help
        num, den = (V / R ** 2) @ D.T, (1 / R) @ D.T + pen + EPS
        if sea_smooth:  # temporal continuity on sea activations (Virtanen 2007)
            S = A[..., ks:]
            nb = np.concatenate([S[:, 1:2], S[:, :-1]], 1) + np.concatenate([S[:, 1:], S[:, -2:-1]], 1)
            num[..., ks:] += 2 * sea_smooth * nb
            den[..., ks:] += 4 * sea_smooth * S
        A *= num / den
    if steady_q is not None:  # ship activity present in every minute is a steady floor: credit it to sea
        As = np.sort(np.where(obs[..., None], A[..., :ks], np.inf), 1)
        idx = np.minimum(steady_q, np.maximum(obs.sum(1) - 1, 0))[:, None, None].repeat(ks, 2)
        steady = np.take_along_axis(As, idx, 1)
        A = A.copy()
        A[..., :ks] = np.clip(A[..., :ks] - steady, 0, None)
    ship = A[..., :ks] @ D_ship
    total = A @ D + EPS
    mask = ship / total  # Wiener-style: fraction of each cell credited to ship shapes
    e_ship = np.where(obs[..., None], V * mask, 0).sum((1, 2))
    e_all = np.where(obs[..., None], V, 0).sum((1, 2))
    minute_frac = np.where(obs, (V * mask).sum(2) / V.sum(2), np.nan)
    if return_bands:
        band_frac = np.where(obs[..., None], V * mask, 0).sum(1) / np.where(obs[..., None], V, 0).sum(1)
        return e_ship / e_all, minute_frac, band_frac
    return e_ship / e_all, minute_frac


def features(X, D_ship, D_sea):
    share, mf = split(X, D_ship, D_sea)
    with np.errstate(all="ignore"):
        return np.column_stack([share, np.nanmax(mf, 1), np.nanmin(mf, 1), np.nanstd(mf, 1),
                                np.nanmax(mf, 1) - np.nanmedian(mf, 1)]).astype(np.float32), \
            ["nmf_share", "nmf_min_max", "nmf_min_min", "nmf_min_std", "nmf_peak_over_med"]


if __name__ == "__main__":  # self-check: known shapes, known mix -> recovered share
    rng = np.random.default_rng(0)
    Ds, Db = rng.random((3, 25)) ** 4, rng.random((3, 25)) ** 4
    Ds, Db = Ds / Ds.sum(1, keepdims=True), Db / Db.sum(1, keepdims=True)
    As, Ab = rng.random((4, 15, 3)), rng.random((4, 15, 3)) * 3
    L = As @ Ds + Ab @ Db
    X = 10 * np.log10(L)
    X[0, 3] = np.nan
    obs = np.isfinite(X).all(2)
    true = np.array([(As[i] @ Ds)[obs[i]].sum() / L[i][obs[i]].sum() for i in range(4)])
    s, _ = split(X, Ds, Db, iters=3000)
    assert np.abs(true - s).max() < 0.01, (true, s)
    print("ok", s.round(3))
