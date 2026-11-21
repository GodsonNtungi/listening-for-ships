"""H4: are some windows "controls" where a VTUAD background clip (sea only, no ship) is added and not counted?
Ranks our 13 submitted methods on exams with control additions, and checks the minute/cell pair lands ~0.028 under broad.
"""
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import ambient
import lbcheck
import mix
import v8

BGCLIPS = [np.load(f).astype(np.float64) for f in sorted((mix.ROOT / "data/vtuad/bands_background").glob("*.npy"))]


def windows(bg, clips, n, rng, cfg):
    """SA background, then (p_ctrl) an uncounted VTUAD sea-only clip, then the counted vessel clip at the drawn share."""
    X, y = np.empty((n, 15, 25), np.float32), np.zeros(n)
    want = mix.draw_share(rng, n, cfg)
    for k in range(n):
        B = 10 ** (bg[rng.integers(len(bg))].astype(np.float64) / 10)
        obs = np.isfinite(B).all(1)
        if obs.all() and rng.random() < 0.38:
            obs[rng.integers(0, 15)] = False
        B[~obs] = np.nan
        if rng.random() < cfg["p_ctrl"]:
            C = mix.passage_minutes(BGCLIPS[rng.integers(len(BGCLIPS))], rng, cfg)
            B = B + 10 ** (rng.uniform(*cfg["ctrl_db"]) / 10) * B[obs].sum() / C[obs].sum() * C
        if want[k] > 0:
            V = mix.passage_minutes(clips[rng.integers(len(clips))], rng, cfg)
            s = want[k] / (1 - want[k]) * B[obs].sum() / V[obs].sum()
            y[k] = s * V[obs].sum() / (B + s * V)[obs].sum()
            B = B + s * V
        X[k] = 10 * np.log10(B)
    return X, y


if __name__ == "__main__":
    m = pickle.loads(lbcheck.CACHE.read_bytes())
    names = list(ambient.PUBLIC)
    pub = np.array([ambient.PUBLIC[n] for n in names])
    v8.RICH = True
    grid = [("faint", (-30, 5), 0.5), ("mid", (-20, 10), 0.3), ("wide", (-25, 15), 0.2)], (0.3, 0.6), ((-10, 10), (-20, 5))
    if "--zoom" in sys.argv:  # around faint + controls, the only region that got the minute/cell pair right
        grid = [("faint", (-30, 5), 0.5), ("vfaint", (-35, 0), 0.5)], (0.6, 0.9), ((-10, 10), (-5, 15), (0, 15))
    for prior_name, prior, pz in grid[0]:
        for p_ctrl in grid[1]:
            for cdb in grid[2]:
                cfg = {**mix.CFG, "ratio_db": prior, "p_zero": pz, "p_ctrl": p_ctrl, "ctrl_db": cdb}
                r = np.random.default_rng(11)
                X, y = windows(lbcheck.bg[r.choice(lbcheck.va_bg, 4000)], lbcheck.va_c, 4000, r, cfg)
                P = lbcheck.predict_all(m, X)
                F8 = v8.features(X, m["D16"], m["D32"])
                for t in ("minute", "cell"):
                    P[f"lgb_v8_{t}"] = np.clip(lgb.Booster(model_file=str(mix.ROOT / f"models/lgb_v8_fold_rich_{t}.txt")).predict(F8), 0, 1)
                loc = np.array([1 - np.sqrt(np.mean((P[n] - y) ** 2)) for n in names])
                loo = [np.polyval(np.polyfit(np.delete(loc, i), np.delete(pub, i), 1), loc[i]) - pub[i] for i in range(len(loc))]
                gap = (loc[-2] + loc[-1]) / 2 - loc[names.index("lgb_v8_broad")]
                print(f"{prior_name:5s} z{pz} ctrl{p_ctrl} {str(cdb):10s} y mean {y.mean():.3f} | rank {spearmanr(loc, pub)[0]:+.2f} "
                      f"LOO {np.mean(np.abs(loo)):.3f} max {np.max(np.abs(loo)):.3f} | pair-broad {gap:+.3f} (board -0.028) | "
                      f"best {loc.max():.3f} ({names[loc.argmax()]})", flush=True)
