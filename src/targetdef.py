"""Is the answer key a different kind of share? Score our 11 submitted methods under other share definitions
and see which one ranks them like the public board, with and without uncounted VTUAD ships.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import lbcheck
import mix


def windows(bg, clips, n, rng, cfg):
    """Like mix.make_windows, but keeps the added ship (A) and the mixture (M) per cell."""
    X = np.empty((n, 15, 25), np.float32)
    A, M = np.zeros((n, 15, 25)), np.ones((n, 15, 25))
    want = mix.draw_share(rng, n, cfg)
    for k in range(n):
        B = 10 ** (bg[rng.integers(len(bg))].astype(np.float64) / 10)
        obs = np.isfinite(B).all(1)
        if obs.all() and rng.random() < 0.38:
            obs[rng.integers(0, 15)] = False
        B[~obs] = np.nan
        if rng.random() < cfg["p_bgship"]:
            N = mix.passage_minutes(clips[rng.integers(len(clips))], rng, cfg)
            if N[obs].sum() > 0:
                B = B + 10 ** (rng.uniform(*cfg["bgship_db"]) / 10) * B[obs].sum() / N[obs].sum() * N
        V = mix.passage_minutes(clips[rng.integers(len(clips))], rng, cfg) if want[k] > 0 else np.zeros((15, 25))
        sv, sb = V[obs].sum(), B[obs].sum()
        s = want[k] / (1 - want[k]) * sb / sv if sv > 0 and want[k] > 0 else 0.0
        A[k], M[k] = np.where(obs[:, None], s * V, 0), np.where(obs[:, None], B + s * V, 1)
        X[k] = 10 * np.log10(B + s * V)
    obs = np.isfinite(X).all(2)
    return X, A, M, obs


def targets(A, M, obs):
    o = obs[..., None] * np.ones(25)
    cell = A / M
    return dict(total=A.sum((1, 2)) / (M * o).sum((1, 2)),
                cell=(cell * o).sum((1, 2)) / o.sum((1, 2)),
                minute=np.nanmean(np.where(obs, A.sum(2) / (M * o).sum(2), np.nan), 1),
                band=(A.sum(1) / (M * o).sum(1)).mean(1),
                low16=A[..., :16].sum((1, 2)) / (M * o)[..., :16].sum((1, 2)))


if __name__ == "__main__":
    m = pickle.loads(lbcheck.CACHE.read_bytes())
    names = list(lbcheck.PUBLIC)
    pub = np.array([lbcheck.PUBLIC[n] for n in names])
    for rname, cfg in [("wide z0.2 nat0", dict(ratio_db=(-25, 15), p_zero=0.2, p_bgship=0.0)),
                       ("mid z0.3 nat0", dict(ratio_db=(-20, 10), p_zero=0.3, p_bgship=0.0)),
                       ("faint z0.5 nat0", dict(ratio_db=(-30, 5), p_zero=0.5, p_bgship=0.0)),
                       ("wide z0.2 nat0.7", dict(ratio_db=(-25, 15), p_zero=0.2, p_bgship=0.7, bgship_db=(-10, 10))),
                       ("faint z0.5 nat0.7", dict(ratio_db=(-30, 5), p_zero=0.5, p_bgship=0.7, bgship_db=(-10, 10)))]:
        r = np.random.default_rng(11)
        X, A, M, obs = windows(lbcheck.bg[r.choice(lbcheck.va_bg, 4000)], lbcheck.va_c, 4000, r, {**mix.CFG, **cfg})
        P = lbcheck.predict_all(m, X)
        for tname, y in targets(A, M, obs).items():
            loc = np.array([1 - np.sqrt(np.mean((P[n] - y) ** 2)) for n in names])
            loo = [np.polyval(np.polyfit(np.delete(loc, i), np.delete(pub, i), 1), loc[i]) - pub[i] for i in range(len(loc))]
            print(f"{rname:18s} {tname:7s} y mean {y.mean():.3f} | rank {spearmanr(loc, pub)[0]:+.2f} LOO {np.mean(np.abs(loo)):.3f} "
                  f"max {np.max(np.abs(loo)):.3f} | best local {loc.max():.3f}", flush=True)
