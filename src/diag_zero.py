"""What does the model say about untouched held-out backgrounds (true answer 0)?"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mix
import train

rng = np.random.default_rng(train.SEED)
bg, st, _ = mix.load_backgrounds()
clips, _, _ = mix.load_passages()
cfg = {**mix.CFG, **train.VARIANTS["lgb_v3_faint"]}
for held in train.STATION_FOLDS:
    tr = np.where(~np.isin(st, held))[0]
    F, y, _ = train.dataset(train.balanced_bg(bg, st, tr, rng, 80_000), clips, 80_000, rng, cfg)
    m = lgb.train(train.PARAMS, lgb.Dataset(F, y), 800)
    for s in held:
        idx = rng.choice(np.where(st == s)[0], min(3000, (st == s).sum()), replace=False)
        Fz, _ = train.build(bg[idx])
        p = np.clip(m.predict(Fz), 0, 1)
        print(f"station {s:2d}: n={len(idx):5d} mean pred {p.mean():.3f}  rmse-if-true-0 {np.sqrt((p ** 2).mean()):.3f}  "
              f"p50 {np.median(p):.3f} p90 {np.percentile(p, 90):.3f}  share>0.2 {(p > 0.2).mean():.2f}", flush=True)
