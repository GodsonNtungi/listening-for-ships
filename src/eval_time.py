"""Time-aware splitter variants on recipe-free anchors (plain bg -> 0, ship-only -> 1) and held-out mixes, alone and floor-capped."""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import floor
import mix
import nmf
import train

rng = np.random.default_rng(train.SEED)
bg, st, _ = mix.load_backgrounds()
clips, hulls, _ = mix.load_passages()
uh = np.unique(hulls)
va_h = set(rng.choice(uh, len(uh) // 5, replace=False))
tr_c = [c for c, h in zip(clips, hulls) if h not in va_h]
va_c = [c for c, h in zip(clips, hulls) if h in va_h]
tr_bg, va_bg = np.where(~np.isin(st, [13, 9]))[0], np.where(np.isin(st, [13, 9]))[0]
bg_va = bg[rng.choice(va_bg, 4000)]
ship_X = np.stack([10 * np.log10(mix.passage_minutes(va_c[i], rng, {**mix.CFG, "tilt_db": 0, "bump_db": 0, "stretch": (1, 1)}) + 1e-30)
                   for i in rng.integers(0, len(va_c), 1500)])
ship_X[ship_X < -200] = np.nan
ship_X = ship_X[np.isfinite(ship_X).all(2).sum(1) >= 3]
Xm, ym, _, _ = mix.make_windows(bg[rng.choice(va_bg, 6000)], va_c, 6000, rng, cfg={**mix.CFG, "ratio_db": (-30, 5)})
D_ship = nmf.learn(nmf.minute_spectra(tr_c), 16)
sea = 10 ** (bg[train.balanced_bg(np.arange(len(bg)), st, tr_bg, rng, 20000)].reshape(-1, 25).astype(np.float64) / 10)
sea = sea[np.isfinite(sea).all(1)]
D_sea = nmf.learn(sea[rng.choice(len(sea), 60000, replace=False)], 32)
fz, fm = floor.share(bg_va, 3.0, 2), floor.share(Xm, 3.0, 2)


def ev(m):
    return f"1-RMSE {1 - train.rmse(np.clip(m, 0, 1), ym):.4f} corr {np.corrcoef(m, ym)[0, 1]:.3f} mean {m.mean():.3f}"


GRID = [("steady q0", dict(steady_q=0))] + [(f"steady q0 + ship penalty {p}", dict(steady_q=0, ship_penalty=p)) for p in (0.002, 0.005, 0.01, 0.02)]
for name, kw in GRID if "--pen" in sys.argv else [("v1", {}), ("sea smooth 0.1", dict(sea_smooth=0.1)), ("sea smooth 1", dict(sea_smooth=1.0)),
                 ("steady q0", dict(steady_q=0)), ("steady q2", dict(steady_q=2)),
                 ("sea smooth 0.1 + steady q2", dict(sea_smooth=0.1, steady_q=2))]:
    z, o, m = (nmf.split(x, D_ship, D_sea, **kw)[0] for x in (bg_va, ship_X, Xm))
    print(f"{name:28s} bg {z.mean():.3f} ship {o.mean():.3f} | alone {ev(m)} | capped bg {np.minimum(z, fz).mean():.3f} {ev(np.minimum(m, fm))}", flush=True)
