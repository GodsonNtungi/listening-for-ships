"""Local scorer that ranks methods like the public board.

  python3 src/score_local.py build   # freeze validation windows + fit local->public line (once)
  python3 src/score_local.py score   # score every method in lbcheck.predict_all

Validation: held-out stations 13+9 and 20% of hulls (lbcheck fold), averaged over the recipes in
RECIPES, chosen because they rank our 7 submissions like the public board. Methods must be trained
on the lbcheck fold only (lbcheck.tr_c / lbcheck.tr_bg), never on va_c / va_bg.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lbcheck
import mix
import train

VAL = mix.ROOT / "data" / __import__("os").environ.get("LOCALVAL", "localval.npz")
CALIB = mix.ROOT / "models/localval_calib.json"
N_PER = 6000
# v3 (2026-10-25): calibrated on the 8 older submissions, these recipes predict lgb_v8's public 0.786 within 0.011;
# v2 (vfaint, localval_v2.npz) over-rated it by 0.07 because v8 trains on that region. v1: localval_v1.npz.
RECIPES = {
    f"{name} z{pz} bg{bgs} {bgdb}": dict(ratio_db=ratio, p_zero=pz, p_bgship=bgs, bgship_db=bgdb)
    for name, ratio, pz, bgs, bgdb in [("faint", (-30, 5), 0.2, 0.7, (-10, 10)), ("faint", (-30, 5), 0.5, 0.7, (-20, 5)),
                                       ("mid", (-30, 10), 0.5, 0.7, (-10, 10)), ("mid", (-30, 10), 0.5, 1.0, (-10, 10)),
                                       ("faint", (-30, 5), 0.5, 0.4, (-10, 10))]
}


def build():
    Xs, ys, rs = [], [], []
    for k, (name, cfg) in enumerate(RECIPES.items()):
        r = np.random.default_rng(1000 + k)
        X, y, _, _ = mix.make_windows(lbcheck.bg[r.choice(lbcheck.va_bg, N_PER)], lbcheck.va_c, N_PER, r, cfg={**mix.CFG, **cfg})
        Xs.append(X), ys.append(y), rs.append(np.full(N_PER, k))
    np.savez_compressed(VAL, X=np.concatenate(Xs), y=np.concatenate(ys), recipe=np.concatenate(rs), names=np.array(list(RECIPES)))


def local_scores(preds):
    z = np.load(VAL)
    return {n: float(np.mean([1 - train.rmse(p[z["recipe"] == k], z["y"][z["recipe"] == k]) for k in range(len(z["names"]))]))
            for n, p in preds.items()}


def calibrate(loc):
    names = list(lbcheck.PUBLIC)
    x, y = np.array([loc[n] for n in names]), np.array([lbcheck.PUBLIC[n] for n in names])
    a, b = np.polyfit(x, y, 1)
    loo = [np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]) - y[i] for i in range(len(x))]
    from scipy.stats import spearmanr
    out = dict(slope=a, intercept=b, rank_agreement=float(spearmanr(x, y)[0]), loo_mean=float(np.mean(np.abs(loo))),
               loo_max=float(np.max(np.abs(loo))), loo={n: float(e) for n, e in zip(names, loo)})
    CALIB.write_text(json.dumps(out, indent=1))
    return out


def predicted_public(local):
    c = json.loads(CALIB.read_text())
    return c["slope"] * local + c["intercept"]


if __name__ == "__main__":
    m = pickle.loads(lbcheck.CACHE.read_bytes())
    if sys.argv[1] == "build":
        build()
    z = np.load(VAL)
    loc = local_scores(lbcheck.predict_all(m, z["X"]))
    if sys.argv[1] == "build":
        c = calibrate(loc)
        print(f"rank agreement {c['rank_agreement']:+.2f} | leave-one-out miss mean {c['loo_mean']:.3f} max {c['loo_max']:.3f}")
    for n, v in sorted(loc.items(), key=lambda t: -t[1]):
        pub = lbcheck.PUBLIC.get(n)
        print(f"{n:26s} local {v:.4f} -> predicted public {predicted_public(v):.4f}" + (f"  (actual {pub:.4f})" if pub else ""))
