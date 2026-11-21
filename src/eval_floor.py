"""Floor predictor vs NMF v1 on held-out stations/hulls. k is picked on plain training backgrounds only."""
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

bg_tr = bg[train.balanced_bg(np.arange(len(bg)), st, tr_bg, rng, 6000)]
bg_va = bg[rng.choice(va_bg, 4000)]
Xm, ym, _, _ = mix.make_windows(bg[rng.choice(va_bg, 6000)], va_c, 6000, rng, cfg={**mix.CFG, "ratio_db": (-30, 5)})


def row(name, z, m):
    print(f"{name:28s} plain bg {z.mean():.3f} | mixes 1-RMSE {1 - train.rmse(np.clip(m, 0, 1), ym):.4f} "
          f"corr {np.corrcoef(m, ym)[0, 1]:.3f} mean {m.mean():.3f}", flush=True)


for q in [0, 2]:
    for k in [1.0, 2.0, 3.0, 5.0]:
        print(f"q={q} k={k}: train plain bg {floor.share(bg_tr, k, q).mean():.3f}", end="  ")
        row("", floor.share(bg_va, k, q), floor.share(Xm, k, q))

D_ship = nmf.learn(nmf.minute_spectra(tr_c), 16)
sea = 10 ** (bg[train.balanced_bg(np.arange(len(bg)), st, tr_bg, rng, 20000)].reshape(-1, 25).astype(np.float64) / 10)
sea = sea[np.isfinite(sea).all(1)]
D_sea = nmf.learn(sea[rng.choice(len(sea), 60000, replace=False)], 32)
nz, nm = nmf.split(bg_va, D_ship, D_sea)[0], nmf.split(Xm, D_ship, D_sea)[0]
row("nmf v1", nz, nm)
K, Q = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0, 0
fz, fm = floor.share(bg_va, K, Q), floor.share(Xm, K, Q)
row(f"average nmf + floor k={K}", (nz + fz) / 2, (nm + fm) / 2)
row(f"min(nmf, floor) k={K}", np.minimum(nz, fz), np.minimum(nm, fm))
for q in [0, 2]:
    for k in [1.0, 2.0, 3.0, 5.0]:
        row(f"min(nmf, floor) q={q} k={k}", np.minimum(nz, floor.share(bg_va, k, q)), np.minimum(nm, floor.share(Xm, k, q)))
