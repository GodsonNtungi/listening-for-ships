"""v8: trees trained on the board-matching recipe family, with NMF split and quiet-floor outputs as inputs.

  python3 src/v8.py local    # train on the lbcheck fold, score on the frozen local scorer
  python3 src/v8.py submit   # train on everything, write submissions/lgb_v8_broad.csv (--region: lgb_v8_region.csv)
"""
import pickle
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import floor
import mix
import nmf
import train

# broader than the scorer's 6 recipes: shares, zero rate, natural-ship rate and level drawn per chunk
TRAIN_SPACE = dict(ratio_lo=(-45, -25), ratio_hi=(-5, 10), p_zero=(0.1, 0.6), p_bgship=(0.2, 1.0), bg_lo=(-25, -10), bg_hi=(0, 12))
if "--region" in sys.argv:  # centred on the recipe region that ranks our submissions like the board
    TRAIN_SPACE = dict(ratio_lo=(-45, -35), ratio_hi=(-5, 5), p_zero=(0.3, 0.7), p_bgship=(0.3, 1.0), bg_lo=(-25, -10), bg_hi=(0, 12))

SPACE = __import__("json").loads(__import__("os").environ.get("V8_SPACE", "{}"))  # sweep overrides, e.g. {"p_env": [0, 0]}
TRAIN_SPACE = {**TRAIN_SPACE, **{k: tuple(v) for k, v in SPACE.items()}}


def draw_cfg(r):
    u = lambda k: r.uniform(*TRAIN_SPACE[k])
    return {**__import__("json").loads(__import__("os").environ.get("V8_EXTRA", "{}")), **dict(ratio_db=(u("ratio_lo"), u("ratio_hi")), p_zero=u("p_zero"), p_bgship=u("p_bgship"), bgship_db=(u("bg_lo"), u("bg_hi")),
                p_env=u("p_env") if "p_env" in TRAIN_SPACE else 0.5, p_phys=u("p_phys") if "p_phys" in TRAIN_SPACE else 0.0,
                target=__import__("os").environ.get("V8_TARGET", "total"),
                p_ctrl=u("p_ctrl") if "p_ctrl" in TRAIN_SPACE else 0.0, ctrl_db=(u("ctrl_lo"), u("ctrl_hi")) if "p_ctrl" in TRAIN_SPACE else None,
                ctrl_clips=(0, 12) if __import__("sys").argv[1] == "local" else (0, 16))}


def make_train(bg, st, bg_idx, clips, n, r, chunk=2000, stations=None):
    pool = None
    if SANAT:  # half the chunks take the uncounted natural ship from real SA passes
        z = np.load(mix.ROOT / "data/sa_windows_clean.npz")
        pool = mix.sa_events(z["X"][np.isin(z["station"], stations)], r, 60_000)
        del z
    Xs, ys = [], []
    for _ in range(n // chunk):
        X, y, _, _ = mix.make_windows(bg[train.balanced_bg(np.arange(len(bg)), st, bg_idx, r, chunk)], clips, chunk, r,
                                      cfg={**mix.CFG, **draw_cfg(r)}, nat_pool=pool if pool is not None and r.random() < 0.5 else None)
        Xs.append(X), ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def features(X, D_ship, D_sea, chunk=20_000):
    if len(X) > chunk:  # NMF split holds several (n, 15, K) float64 arrays: chunk to stay in memory
        return np.concatenate([features(X[i:i + chunk], D_ship, D_sea) for i in range(0, len(X), chunk)])
    F, _ = train.build(X)
    N, _ = nmf.features(X, D_ship, D_sea)
    st_share = nmf.split(X, D_ship, D_sea, steady_q=0)[0]
    fl = np.column_stack([floor.share(X, k, q) for k, q in [(1, 0), (2, 2), (3, 2), (5, 2), (10, 2)]])
    v1 = N[:, :1]
    if RICH:
        _, mf, bf = nmf.split(X, D_ship, D_sea, return_bands=True)
        mf_sorted = np.sort(np.nan_to_num(mf, nan=-1.0), 1)  # pass shape, independent of where it sits
        fb = floor.share(X, 3.0, 2, per_band=True)
        F = np.hstack([F, mf_sorted, bf, fb])
    return np.hstack([F, N, st_share[:, None], fl, np.minimum(v1, fl[:, 2:3]), np.minimum(st_share[:, None], fl[:, 2:3])]).astype(np.float32)


RICH = "--rich" in sys.argv
SANAT = "--sanat" in sys.argv
ROUNDS, N_TRAIN = (2000, 240_000) if RICH else (1200, 120_000)

if __name__ == "__main__":
    mode = sys.argv[1]
    r = np.random.default_rng(train.SEED + 8)
    if mode == "local":
        import lbcheck
        import score_local
        m = pickle.loads(lbcheck.CACHE.read_bytes())
        D_ship, D_sea = m["D16"], m["D32"]
        X, y = make_train(lbcheck.bg, lbcheck.st, lbcheck.tr_bg, lbcheck.tr_c, N_TRAIN, r, stations=np.setdiff1d(lbcheck.st, [13, 9]))
        model = lgb.train(train.PARAMS, lgb.Dataset(features(X, D_ship, D_sea), y), ROUNDS)
        z = np.load(score_local.VAL)
        p = np.clip(model.predict(features(z["X"], D_ship, D_sea)), 0, 1)
        loc = score_local.local_scores({"lgb_v8": p})["lgb_v8"]
        print(f"lgb_v8 {SPACE} local {loc:.4f} -> predicted public {score_local.predicted_public(loc):.4f}")
        np.save(mix.ROOT / "models/v8_local_pred.npy", p)
        model.save_model(str(mix.ROOT / f"models/lgb_v8_fold{'_rich' if RICH else ''}{'_region' if '--region' in sys.argv else ''}{'_sanat' if SANAT else ''}{'_' + __import__('os').environ['V8_TAG'] if 'V8_TAG' in __import__('os').environ else ''}.txt"))
    else:
        tag = __import__("os").environ.get("V8_TAG") or ("lgb_v8_region" if "--region" in sys.argv else "lgb_v8_broad") + ("_sanat" if SANAT else "")
        bg, st, _ = mix.load_backgrounds()
        clips, _, _ = mix.load_passages()
        sh = np.load(mix.ROOT / "models/nmf_v1_shapes.npz")
        X, y = make_train(bg, st, np.arange(len(bg)), clips, N_TRAIN, r, stations=np.unique(st))
        model = lgb.train(train.PARAMS, lgb.Dataset(features(X, sh["D_ship"], sh["D_sea"]), y), ROUNDS)
        test = pd.read_csv(mix.ROOT / "data/Test.csv", dtype={"ID": str})
        p = np.clip(model.predict(features(test.drop(columns="ID").to_numpy(np.float32).reshape(-1, 15, 25), sh["D_ship"], sh["D_sea"])), 0, 1)
        pd.DataFrame({"ID": test.ID, "share_added_vessel": p, "score_added_vessel": p}).to_csv(mix.ROOT / f"submissions/{tag}.csv", index=False)
        model.save_model(str(mix.ROOT / f"models/{tag}.txt"))
        print(f"wrote {tag}.csv mean {p.mean():.3f}")
