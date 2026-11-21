"""Held-out check: NMF share alone, trees alone, trees + NMF features. Fold: stations 13+9 and 20% of hulls."""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mix
import nmf
import train

K_SHIP, K_SEA = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(train.SEED)
bg, st, _ = mix.load_backgrounds()
clips, hulls, _ = mix.load_passages()
uh = np.unique(hulls)
va_h = set(rng.choice(uh, len(uh) // 5, replace=False))
tr_c = [c for c, h in zip(clips, hulls) if h not in va_h]
va_c = [c for c, h in zip(clips, hulls) if h in va_h]
tr_bg = np.where(~np.isin(st, [13, 9]))[0]
va_bg = np.where(np.isin(st, [13, 9]))[0]

t = time.time()
D_ship = nmf.learn(nmf.minute_spectra(tr_c), K_SHIP)
sea_min = 10 ** (bg[rng.choice(train.balanced_bg(np.arange(len(bg)), st, tr_bg, rng, 20000), 20000)].reshape(-1, 25).astype(np.float64) / 10)
sea_min = sea_min[np.isfinite(sea_min).all(1)]
D_sea = nmf.learn(sea_min[rng.choice(len(sea_min), 60000, replace=False)], K_SEA)
print(f"shapes learned in {time.time() - t:.0f}s", flush=True)

cfg = {**mix.CFG, "ratio_db": (-30, 5)}
for name, extra in [("base", {}), ("bgship", {"p_bgship": 0.4})]:
    Xv, yv, _, _ = mix.make_windows(bg[rng.choice(va_bg, 15000)], va_c, 15000, rng, cfg={**cfg, **extra})
    s, _ = nmf.split(Xv, D_ship, D_sea)
    print(f"[{name}] NMF share alone: 1-RMSE {1 - train.rmse(np.clip(s, 0, 1), yv):.4f}  corr {np.corrcoef(s, yv)[0, 1]:.3f}", flush=True)
    if name == "base":
        Xt, yt, _, _ = mix.make_windows(train.balanced_bg(bg, st, tr_bg, rng, 80000), tr_c, 80000, rng, cfg=cfg)
        Ft, _ = train.build(Xt)
        Nt, _ = nmf.features(Xt, D_ship, D_sea)
    Fv, _ = train.build(Xv)
    Nv, _ = nmf.features(Xv, D_ship, D_sea)
    for label, a, b in [("trees", Ft, Fv), ("trees+nmf", np.hstack([Ft, Nt]), np.hstack([Fv, Nv]))]:
        key = (label,)
        if key not in globals().setdefault("models", {}):
            models[key] = lgb.train(train.PARAMS, lgb.Dataset(a, yt), 800)
        p = np.clip(models[key].predict(b), 0, 1)
        print(f"[{name}] {label:10s} 1-RMSE {1 - train.rmse(p, yv):.4f}", flush=True)
