"""Train on one way of building mixes, score on others: which training recipe survives an unknown test recipe?"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mix
import train

ALL_GENS = {
    "base": dict(ratio_db=(-30, 5)),
    "bgship": dict(ratio_db=(-30, 5), p_bgship=0.4),
    "natural": dict(ratio_db=(-30, 5), stretch=(1.0, 1.0), tilt_db=3.0, bump_db=1.5),
    "wide": dict(ratio_db=(-30, 10), p_bgship=0.25, stretch=(0.7, 1.4), tilt_db=6.0, p_zero=0.3),
    "env": dict(ratio_db=(-30, 5), p_env=0.8),
    "env_all": dict(ratio_db=(-30, 5), p_env=1.0, env_width_min=(0.7, 5.0)),
}
GENS = {k: ALL_GENS[k] for k in (sys.argv[1:] or ["base", "bgship", "natural", "wide"])}
rng = np.random.default_rng(train.SEED)
bg, st, _ = mix.load_backgrounds()
clips, hulls, _ = mix.load_passages()
held = [13, 9]
uh = np.unique(hulls)
va_h = set(rng.choice(uh, len(uh) // 5, replace=False))
tr_c = [c for c, h in zip(clips, hulls) if h not in va_h]
va_c = [c for c, h in zip(clips, hulls) if h in va_h]
tr_bg = np.where(~np.isin(st, held))[0]
va_bg = np.where(np.isin(st, held))[0]
vals = {g: train.dataset(bg[rng.choice(va_bg, 15000)], va_c, 15000, rng, {**mix.CFG, **c})[:2] for g, c in GENS.items()}
print("train \\ eval  " + "  ".join(f"{g:>8s}" for g in GENS) + "   mean")
for g, c in GENS.items():
    F, y, _ = train.dataset(train.balanced_bg(bg, st, tr_bg, rng, 80_000), tr_c, 80_000, rng, {**mix.CFG, **c})
    m = lgb.train(train.PARAMS, lgb.Dataset(F, y), 800)
    sc = [1 - train.rmse(np.clip(m.predict(Fv), 0, 1), yv) for Fv, yv in vals.values()]
    print(f"{g:12s}  " + "  ".join(f"{v:8.4f}" for v in sc) + f"   {np.mean(sc):.4f}", flush=True)
