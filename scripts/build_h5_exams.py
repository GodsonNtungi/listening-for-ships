"""Frozen exams for the two worlds that passed the out-of-sample check."""
import sys
import numpy as np
sys.path.insert(0, "src")
import examtest, lbcheck, mix
for fname, mode, cfg in [("localval_h5.npz", "h5", dict(ratio_db=(-20, 10), p_zero=0.3, p_bgship=0.4, bgship_db=(-10, 10), tilt_db=16.0, bump_db=6.0)),
                         ("localval_quietnat.npz", "plain", dict(ratio_db=(-30, 5), p_zero=0.5, p_bgship=0.7, bgship_db=(-20, 5)))]:
    r = np.random.default_rng(3000)
    X, y = examtest.windows(lbcheck.bg[r.choice(lbcheck.va_bg, 12000)], lbcheck.va_c, 12000, r, {**mix.CFG, **cfg}, mode)
    np.savez_compressed(mix.ROOT / "data" / fname, X=X, y=y, recipe=np.zeros(len(y), int), names=np.array([fname]))
    print("built", fname, y.mean(), flush=True)
