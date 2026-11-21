"""H2: does the answer key leave out the Canadian sea noise inside each VTUAD clip?
Mixture always gets the full clip; the counted part is the clip minus its quiet floor (or minus typical VTUAD ambient).
Ranks our 13 submitted methods on each version.
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

PUBLIC = {**lbcheck.PUBLIC, "lgb_v8_minute": 0.7824, "lgb_v8_cell": 0.7829}
AMB = np.median([np.median(np.load(f), 0) for f in (mix.ROOT / "data/vtuad/bands_background").glob("*.npy")], 0)


def floors(clips, how):
    if how == "full":
        return [np.zeros(25) for _ in clips]
    if how == "amb":
        return [AMB for _ in clips]
    return [np.percentile(c, float(how[1:]), 0) for c in clips]  # "p5", "p20": the clip's own quiet floor


def windows(bg, clips, fl, n, rng, cfg):
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
        if want[k] == 0:
            X[k] = 10 * np.log10(B)
            continue
        i = rng.integers(len(clips))
        r2 = copy.deepcopy(rng)
        V = mix.passage_minutes(clips[i], rng, cfg)
        S = mix.passage_minutes(np.maximum(clips[i] - fl[i], 1e-6 * clips[i]), r2, cfg)  # same random draws
        s = want[k] / (1 - want[k]) * B[obs].sum() / V[obs].sum()  # loudness set on the full clip
        M = B + s * V
        y[k] = s * S[obs].sum() / M[obs].sum()
        X[k] = 10 * np.log10(M)
    return X, y


if __name__ == "__main__":
    m = pickle.loads(lbcheck.CACHE.read_bytes())
    names = list(PUBLIC)
    pub = np.array([PUBLIC[n] for n in names])
    v8.RICH = True
    for rname, cfg in [("wide z0.2 nat0", dict(ratio_db=(-25, 15), p_zero=0.2, p_bgship=0.0)),
                       ("mid z0.3 nat0", dict(ratio_db=(-20, 10), p_zero=0.3, p_bgship=0.0)),
                       ("faint z0.5 nat0", dict(ratio_db=(-30, 5), p_zero=0.5, p_bgship=0.0)),
                       ("faint z0.5 nat0.7", dict(ratio_db=(-30, 5), p_zero=0.5, p_bgship=0.7, bgship_db=(-10, 10)))]:
        for how in ("full", "amb", "p5", "p20"):
            r = np.random.default_rng(11)
            X, y = windows(lbcheck.bg[r.choice(lbcheck.va_bg, 4000)], lbcheck.va_c, floors(lbcheck.va_c, how), 4000, r, {**mix.CFG, **cfg})
            P = lbcheck.predict_all(m, X)
            F8 = v8.features(X, m["D16"], m["D32"])
            for t in ("minute", "cell"):
                P[f"lgb_v8_{t}"] = np.clip(lgb.Booster(model_file=str(mix.ROOT / f"models/lgb_v8_fold_rich_{t}.txt")).predict(F8), 0, 1)
            loc = np.array([1 - np.sqrt(np.mean((P[n] - y) ** 2)) for n in names])
            loo = [np.polyval(np.polyfit(np.delete(loc, i), np.delete(pub, i), 1), loc[i]) - pub[i] for i in range(len(loc))]
            print(f"{rname:18s} {how:5s} y mean {y.mean():.3f} | rank {spearmanr(loc, pub)[0]:+.2f} LOO {np.mean(np.abs(loo)):.3f} "
                  f"max {np.max(np.abs(loo)):.3f} | best local {loc.max():.3f} ({names[loc.argmax()]}) | minute,cell,broad "
                  f"{loc[-2]:.3f} {loc[-1]:.3f} {loc[names.index('lgb_v8_broad')]:.3f}", flush=True)
