"""Frozen exam with sea-only controls from held-out VTUAD background clips 12-15."""
import sys
import numpy as np
sys.path.insert(0, "src")
import controls, lbcheck, mix
controls.BGCLIPS = controls.BGCLIPS[12:]
R = {"faint z0.5 ctrl0.9 (-5,15)": ((-30, 5), 0.9, (-5, 15)), "vfaint z0.5 ctrl0.9 (0,15)": ((-35, 0), 0.9, (0, 15)),
     "faint z0.5 ctrl0.6 (0,15)": ((-30, 5), 0.6, (0, 15))}
Xs, ys, rs = [], [], []
for k, (prior, pc, cdb) in enumerate(R.values()):
    r = np.random.default_rng(2000 + k)
    X, y = controls.windows(lbcheck.bg[r.choice(lbcheck.va_bg, 6000)], lbcheck.va_c, 6000, r,
                            {**mix.CFG, "ratio_db": prior, "p_zero": 0.5, "p_ctrl": pc, "ctrl_db": cdb})
    Xs.append(X), ys.append(y), rs.append(np.full(6000, k))
np.savez_compressed(mix.ROOT / "data/localval_ctrl.npz", X=np.concatenate(Xs), y=np.concatenate(ys), recipe=np.concatenate(rs), names=np.array(list(R)))
print("built")
