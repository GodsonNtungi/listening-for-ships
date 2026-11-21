"""Grouped CV + final fit + submission for the share model.

Folds hold out whole SA stations (backgrounds) and whole VTUAD hulls (passages) together,
so the score reflects new places and new ships. Nothing here reads Test.csv except predict().
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import features
import mix

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
STATION_FOLDS = [[1], [5], [13, 9], [7, 2], [4, 10]]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=10, deterministic=True)


def balanced_bg(bg, st, idx, rng, n):
    """Sample background rows so each station gets weight ~ sqrt(size), not size."""
    sts, counts = np.unique(st[idx], return_counts=True)
    w = np.sqrt(counts) / np.sqrt(counts).sum()
    pick = []
    for s, ws in zip(sts, w):
        pool = idx[st[idx] == s]
        pick.append(rng.choice(pool, int(round(ws * n)), replace=True))
    return bg[np.concatenate(pick)]


ABSOLUTE = ("_med", "_min", "_max", "_p25", "_p75", "_linmean")


def build(X, relative_only=True):
    F, names = features.build(X)
    if relative_only:  # absolute dB levels identify the recorder, not the ship
        keep = [i for i, n in enumerate(names) if not (n.startswith("b") and n.endswith(ABSOLUTE))]
        F, names = F[:, keep], [names[i] for i in keep]
    return F, names


VARIANTS = {
    "lgb_v1": {},
    "lgb_v2_zeros": dict(p_zero=0.4),
    "lgb_v3_faint": dict(ratio_db=(-30, 5)),
    "lgb_v4_natural": dict(stretch=(1.0, 1.0), tilt_db=3.0, bump_db=1.5),
    "lgb_v5_clean": dict(ratio_db=(-30, 5)),  # same as v3, on blip-free backgrounds
    "lgb_v6_randstart": dict(ratio_db=(-30, 5)),  # same as v3, windows start at random minutes
    "lgb_v7_env": dict(ratio_db=(-30, 5), p_env=0.8),  # passes swell and fade with distance
}


def dataset(bg, clips, n, rng, cfg=mix.CFG):
    X, y, _, _ = mix.make_windows(bg, clips, n, rng, cfg=cfg)
    F, names = build(X)
    return F, y, names


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cv(n_train=120_000, n_val=20_000, rounds=1500):
    rng = np.random.default_rng(SEED)
    bg, st, _ = mix.load_backgrounds()
    clips, hulls, _ = mix.load_passages()
    uh = np.unique(hulls)
    hull_fold = dict(zip(uh, rng.permutation(len(uh)) % len(STATION_FOLDS)))
    clip_fold = np.array([hull_fold[h] for h in hulls])
    res = []
    for k, held in enumerate(STATION_FOLDS):
        tr_bg = np.where(~np.isin(st, held))[0]
        va_bg = np.where(np.isin(st, held))[0]
        tr_clips = [c for c, f in zip(clips, clip_fold) if f != k]
        va_clips = [c for c, f in zip(clips, clip_fold) if f == k]
        Ftr, ytr, names = dataset(balanced_bg(bg, st, tr_bg, rng, n_train), tr_clips, n_train, rng)
        Fva, yva, _ = dataset(bg[rng.choice(va_bg, n_val)], va_clips, n_val, rng)
        m = lgb.train(PARAMS, lgb.Dataset(Ftr, ytr), rounds, valid_sets=[lgb.Dataset(Fva, yva)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = np.clip(m.predict(Fva, num_iteration=m.best_iteration), 0, 1)
        ex = np.clip(Fva[:, names.index("excess_p20")], 0, 1)
        r = dict(fold=k, held=held, clips=len(va_clips), best_iter=m.best_iteration, model=rmse(p, yva),
                 const_mean=rmse(np.full_like(yva, ytr.mean()), yva), excess_p20=rmse(ex, yva))
        res.append(r)
        print(json.dumps(r), flush=True)
    df = pd.DataFrame(res)
    print("mean score (1-RMSE): model %.4f | constant %.4f | floor-excess %.4f" % tuple(1 - df[c].mean() for c in ["model", "const_mean", "excess_p20"]))
    return df


def fit_and_predict(n_train=200_000, rounds=None, tag="lgb_v1"):
    rng = np.random.default_rng(SEED)
    bg, st, _ = mix.load_backgrounds()
    clips, _, _ = mix.load_passages()
    cfg = {**mix.CFG, **VARIANTS[tag]}
    F, y, names = dataset(balanced_bg(bg, st, np.arange(len(bg)), rng, n_train), clips, n_train, rng, cfg)
    m = lgb.train(PARAMS, lgb.Dataset(F, y), rounds)
    test = pd.read_csv(ROOT / "data/Test.csv", dtype={"ID": str})
    Xt = test.drop(columns="ID").to_numpy(np.float32).reshape(-1, 15, 25)
    Ft, names_t = build(Xt)
    assert names_t == names
    p = np.clip(m.predict(Ft), 0, 1)
    sub = pd.DataFrame({"ID": test["ID"], "share_added_vessel": p, "score_added_vessel": p})
    out = ROOT / f"submissions/{tag}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    m.save_model(str(ROOT / f"models/{tag}.txt")) if (ROOT / "models").mkdir(exist_ok=True) is None else None
    print(f"wrote {out}: mean share {p.mean():.3f}, zeros-ish {(p < 0.02).mean():.2f}")


if __name__ == "__main__":
    t = time.time()
    if sys.argv[1] == "cv":
        cv()
    else:
        fit_and_predict(rounds=int(sys.argv[2]), tag=sys.argv[3] if len(sys.argv) > 3 else "lgb_v1")
    print(f"{time.time() - t:.0f}s")
