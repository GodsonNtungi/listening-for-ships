"""Do real SA ship passes, as the uncounted natural ship, match the board better than VTUAD clips?

Each scorer v3 recipe is run with both natural-ship sources; methods are calibrated on the 8 older submissions and
must predict both v8 public scores.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import lbcheck
import mix
import score_local
import train

if __name__ == "__main__":
    m = pickle.loads(lbcheck.CACHE.read_bytes())
    z = np.load(mix.ROOT / "data/sa_windows_clean.npz")
    pool = mix.sa_events(z["X"][np.isin(z["station"], [13, 9])], np.random.default_rng(5), 20_000)
    print("validation SA events", len(pool), flush=True)
    names = list(lbcheck.PUBLIC)
    pub = np.array([lbcheck.PUBLIC[n] for n in names])
    old = np.array([not n.startswith("lgb_v8") for n in names])
    for rname, cfg in score_local.RECIPES.items():
        for src in ("vtuad", "sa"):
            r = np.random.default_rng(7)
            X, y, _, _ = mix.make_windows(lbcheck.bg[r.choice(lbcheck.va_bg, 4000)], lbcheck.va_c, 4000, r, cfg={**mix.CFG, **cfg},
                                          nat_pool=pool if src == "sa" else None)
            P = lbcheck.predict_all(m, X)
            loc = np.array([1 - train.rmse(P[n], y) for n in names])
            a, b = np.polyfit(loc[old], pub[old], 1)
            miss = np.polyval([a, b], loc[~old]) - pub[~old]
            loo = [np.polyval(np.polyfit(np.delete(loc, i), np.delete(pub, i), 1), loc[i]) - pub[i] for i in range(len(loc))]
            print(f"{rname:32s} {src:5s} rank {spearmanr(loc, pub)[0]:+.2f} | v8 misses " + " ".join(f"{v:+.3f}" for v in miss)
                  + f" | LOO mean {np.mean(np.abs(loo)):.3f} max {np.max(np.abs(loo)):.3f} | " + " ".join(f"{v:.3f}" for v in loc), flush=True)
