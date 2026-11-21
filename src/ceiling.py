"""Best score possible under a recipe for a model that knows the total ship energy but must split added vs natural."""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import lbcheck
import mix
import train


def ceiling(cfg, n=8000, seed=5):
    r = np.random.default_rng(seed)
    out = []
    for idx, clips in ((lbcheck.tr_bg, lbcheck.tr_c), (lbcheck.va_bg, lbcheck.va_c)):
        X, y, _, _, nat = mix.make_windows(lbcheck.bg[r.choice(idx, n)], clips, n, r, cfg={**mix.CFG, **cfg}, return_nat=True)
        out.append((np.column_stack([y + nat]), y))
    m = lgb.train({**train.PARAMS, "num_leaves": 15}, lgb.Dataset(*out[0]), 300)
    return 1 - train.rmse(np.clip(m.predict(out[1][0]), 0, 1), out[1][1])


if __name__ == "__main__":
    grids = {}
    for flag in ("", "--grid2"):
        sys.argv = ["x", flag]
        import importlib
        importlib.reload(lbcheck)
        grids.update(lbcheck.RECIPES)
    for name, cfg in grids.items():
        if cfg.get("p_bgship", 0) > 0:
            print(f"{name:52s} ceiling {ceiling(cfg):.4f}", flush=True)
