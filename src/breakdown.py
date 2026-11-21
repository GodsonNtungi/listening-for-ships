"""Where do our points go? Error of v8 broad and the splitter, split by kind of window, on scorer v3 recipes.

  python3 src/breakdown.py   # writes models/breakdown.json, then viz/breakdown.html from viz/breakdown_template.html
"""
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import floor
import lbcheck
import mix
import nmf
import score_local
import v8

GROUPS = [  # (key, label, test on true share y and natural-ship share nat)
    ("empty", "No ship at all", lambda y, n: (y == 0) & (n < 0.02)),
    ("trap", "Only a ship that doesn't count", lambda y, n: (y == 0) & (n >= 0.02)),
    ("faint", "Faint added ship (under 10%)", lambda y, n: (y > 0) & (y < 0.1)),
    ("medium", "Medium added ship (10-40%)", lambda y, n: (y >= 0.1) & (y < 0.4)),
    ("loud", "Loud added ship (over 40%)", lambda y, n: y >= 0.4),
]

if __name__ == "__main__":
    m = pickle.loads(lbcheck.CACHE.read_bytes())
    Xs, ys, ns = [], [], []
    for k, cfg in enumerate(score_local.RECIPES.values()):
        r = np.random.default_rng(500 + k)
        X, y, _, _, nat = mix.make_windows(lbcheck.bg[r.choice(lbcheck.va_bg, 3000)], lbcheck.va_c, 3000, r, cfg={**mix.CFG, **cfg}, return_nat=True)
        Xs.append(X), ys.append(y), ns.append(nat)
    X, y, nat = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ns)
    v8.RICH = True
    pred = {"v8": np.clip(lgb.Booster(model_file=str(mix.ROOT / "models/lgb_v8_fold_rich.txt")).predict(v8.features(X, m["D16"], m["D32"])), 0, 1),
            "splitter": np.minimum(np.clip(nmf.split(X, m["D16"], m["D32"], steady_q=0)[0], 0, 1), floor.share(X, 3.0, 2))}
    total = {k: ((p - y) ** 2).sum() for k, p in pred.items()}
    out = dict(n=len(y), score={k: float(1 - np.sqrt(((p - y) ** 2).mean())) for k, p in pred.items()}, groups=[], examples=[])
    r = np.random.default_rng(0)
    for key, label, test in GROUPS:
        g = test(y, nat)
        row = dict(key=key, label=label, frac=float(g.mean()), truth=float(y[g].mean()))
        for k, p in pred.items():
            row[k] = dict(guess=float(p[g].mean()), err=float(np.sqrt(((p[g] - y[g]) ** 2).mean())), lost=float(((p[g] - y[g]) ** 2).sum() / total[k]))
        out["groups"].append(row)
        i = r.choice(np.flatnonzero(g & np.isfinite(X).all((1, 2))))
        out["examples"].append(dict(key=key, db=np.round(X[i], 1).tolist(), truth=float(y[i]), nat=float(nat[i]), v8=float(pred["v8"][i])))
    (mix.ROOT / "models/breakdown.json").write_text(json.dumps(out, indent=1))
    for g in out["groups"]:
        print(f"{g['label']:34s} {g['frac']:5.1%} of windows | truth {g['truth']:.3f} | v8 guess {g['v8']['guess']:.3f} err {g['v8']['err']:.3f} "
              f"lost {g['v8']['lost']:5.1%} | splitter guess {g['splitter']['guess']:.3f} lost {g['splitter']['lost']:5.1%}")
    print(out["score"])
    tpl = mix.ROOT / "viz/breakdown_template.html"
    if tpl.exists():
        (mix.ROOT / "viz/breakdown.html").write_text(tpl.read_text().replace("/*DATA*/null", json.dumps(out)))
