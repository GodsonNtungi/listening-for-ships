"""Final splitter: ship16 + sea64 (ship-like sea shapes pruned), optional isotonic curve. All shapes from training data only."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import mix
import nmf
import train

rng = np.random.default_rng(train.SEED)
bg, st, _ = mix.load_backgrounds()
clips, _, _ = mix.load_passages()
allbg = np.arange(len(bg))
D_ship = nmf.learn(nmf.minute_spectra(clips), 16)
sea = 10 ** (bg[train.balanced_bg(allbg, st, allbg, rng, 60000)].reshape(-1, 25).astype(np.float64) / 10)
sea = sea[np.isfinite(sea).all(1)]
D_sea = nmf.learn(sea[rng.choice(len(sea), 150000, replace=False)], 64)


def usage(X):
    L = 10 ** (X.astype(np.float64) / 10)
    obs = np.isfinite(L).all(2)
    V = np.where(obs[..., None], L / np.nanmean(L, (1, 2), keepdims=True), 1.0)
    D = np.concatenate([D_ship, D_sea])
    A = np.full(V.shape[:2] + (len(D),), 25 / len(D))
    for _ in range(150):
        R = A @ D + nmf.EPS
        A *= ((V / R ** 2) @ D.T) / ((1 / R) @ D.T + nmf.EPS)
    e = np.where(obs[..., None], A, 0).sum((0, 1))[len(D_ship):]
    return e / e.sum()


ship_X = np.stack([10 * np.log10(mix.passage_minutes(clips[i], rng, {**mix.CFG, "tilt_db": 0, "bump_db": 0, "stretch": (1, 1)}) + 1e-30)
                   for i in rng.integers(0, len(clips), 2000)])
ship_X[ship_X < -200] = np.nan
keep = usage(ship_X) / (usage(bg[rng.choice(len(bg), 5000)]) + 1e-9) < 3
D_sea = D_sea[keep]
print("sea shapes kept", keep.sum(), "of", len(keep))
np.savez(mix.ROOT / "models/nmf_v2_shapes.npz", D_ship=D_ship, D_sea=D_sea)

Xc, yc, _, _ = mix.make_windows(train.balanced_bg(bg, st, allbg, rng, 20000), clips, 20000, rng, cfg={**mix.CFG, "ratio_db": (-30, 5)})
iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(nmf.split(Xc, D_ship, D_sea)[0], yc)

test = pd.read_csv(mix.ROOT / "data/Test.csv", dtype={"ID": str})
raw = np.clip(nmf.split(test.drop(columns="ID").to_numpy(np.float32).reshape(-1, 15, 25), D_ship, D_sea)[0], 0, 1)
for tag, p in [("nmf_v2_pruned", raw), ("nmf_v2_pruned_curve", iso.predict(raw))]:
    pd.DataFrame({"ID": test.ID, "share_added_vessel": p, "score_added_vessel": raw}).to_csv(mix.ROOT / f"submissions/{tag}.csv", index=False)
    print(f"{tag}: mean {p.mean():.3f} pct10/50/90 {np.percentile(p, [10, 50, 90]).round(2)}")
v1 = pd.read_csv(mix.ROOT / "submissions/nmf_v1.csv").share_added_vessel.to_numpy()
print(f"corr raw vs nmf_v1 {np.corrcoef(raw, v1)[0, 1]:.2f}, nmf_v1 mean {v1.mean():.3f}")
