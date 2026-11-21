"""Out-of-sample exam test: calibrate local->public on the 11 older submissions, then predict the 4 new ones
(minute, cell, ctrlnat, ctrl) that no exam was fitted to. An exam earns trust only by beating the old exam's miss.
"""
import copy
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import lbcheck
import mix
import v8

NEW = {"minute": 0.7824, "cell": 0.7829, "ctrlnat": 0.8029, "ctrl": 0.7883}


def windows(bg, clips, n, rng, cfg, mode="plain"):
    """mode plain: old exam style. h5: label from the passage before its tone change. h6: answer also counts hidden low bands."""
    X, y = np.empty((n, 15, 25), np.float32), np.zeros(n)
    want = mix.draw_share(rng, n, cfg)
    for k in range(n):
        B = 10 ** (bg[rng.integers(len(bg))].astype(np.float64) / 10)
        obs = np.isfinite(B).all(1)
        if obs.all() and rng.random() < 0.38:
            obs[rng.integers(0, 15)] = False
        B[~obs] = np.nan
        if rng.random() < cfg["p_bgship"]:
            N = mix.passage_minutes(clips[rng.integers(len(clips))], rng, cfg)
            B = B + 10 ** (rng.uniform(*cfg["bgship_db"]) / 10) * B[obs].sum() / N[obs].sum() * N
        hidden = rng.uniform(*cfg.get("hid", (0, 0))) * B[obs, :2].sum() if mode == "h6" else 0.0
        if want[k] > 0:
            i = rng.integers(len(clips))
            r2 = copy.deepcopy(rng)
            V = mix.passage_minutes(clips[i], rng, cfg)
            s = want[k] / (1 - want[k]) * B[obs].sum() / V[obs].sum()
            M = B + s * V
            num = s * V[obs].sum()
            if mode == "h5":
                U = mix.passage_minutes(clips[i], r2, {**cfg, "tilt_db": 0.0, "bump_db": 0.0})
                num = s * U[obs].sum()
            y[k] = min(num / (M[obs].sum() + hidden), 1.0)
            B = M
        X[k] = 10 * np.log10(B)
    return X, y


def evaluate(tag, X, y, m):
    P = lbcheck.predict_all(m, X)
    F8 = v8.features(X, m["D16"], m["D32"])
    for t in NEW:
        P[t] = np.clip(lgb.Booster(model_file=str(mix.ROOT / f"models/lgb_v8_fold_rich_{t}.txt")).predict(F8), 0, 1)
    old = list(lbcheck.PUBLIC)
    loc = {n: 1 - np.sqrt(np.mean((P[n] - y) ** 2)) for n in P}
    x, p = np.array([loc[n] for n in old]), np.array([lbcheck.PUBLIC[n] for n in old])
    a, b = np.polyfit(x, p, 1)
    miss = {t: a * loc[t] + b - NEW[t] for t in NEW}
    allx = np.array([loc[n] for n in old + list(NEW)])
    allp = np.concatenate([p, list(NEW.values())])
    print(f"{tag:34s} y {y.mean():.3f} | rank15 {spearmanr(allx, allp)[0]:+.2f} | new-4 miss mean {np.mean(np.abs(list(miss.values()))):.3f} ("
          + " ".join(f"{t} {v:+.3f}" for t, v in miss.items()) + f") | broad local {loc['lgb_v8_broad']:.3f}", flush=True)


if __name__ == "__main__":
    m = pickle.loads(lbcheck.CACHE.read_bytes())
    v8.RICH = True
    faint = dict(ratio_db=(-30, 5), p_zero=0.5)
    exams = [("old exam style faint nat0.7", "plain", {**faint, "p_bgship": 0.7, "bgship_db": (-10, 10)}),
             ("old exam style faint nat0.7 (-20,5)", "plain", {**faint, "p_bgship": 0.7, "bgship_db": (-20, 5)})]
    for pr_name, pr in [("faint", faint), ("mid", dict(ratio_db=(-20, 10), p_zero=0.3))]:
        for nat in (0.0, 0.4):
            base = {**pr, "p_bgship": nat, "bgship_db": (-10, 10)}
            exams += [(f"h5 {pr_name} nat{nat}", "h5", base), (f"h5x2 {pr_name} nat{nat}", "h5", {**base, "tilt_db": 16.0, "bump_db": 6.0}),
                      (f"h6 {pr_name} nat{nat} hid0.5-3", "h6", {**base, "hid": (0.5, 3)}),
                      (f"h6 {pr_name} nat{nat} hid2-10", "h6", {**base, "hid": (2, 10)})]
    for tag, mode, cfg in exams:
        r = np.random.default_rng(11)
        X, y = windows(lbcheck.bg[r.choice(lbcheck.va_bg, 4000)], lbcheck.va_c, 4000, r, {**mix.CFG, **cfg}, mode)
        evaluate(tag, X, y, m)
