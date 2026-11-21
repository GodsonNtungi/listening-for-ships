"""Recipe-free checks for the splitter: plain held-out SA backgrounds should read 0, held-out ship-only windows 1."""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import mix
import nmf
import train

rng = np.random.default_rng(train.SEED)
bg, st, _ = mix.load_backgrounds()
clips, hulls, _ = mix.load_passages()
amb = [np.load(f).astype(float) for f in sorted((mix.ROOT / "data/vtuad/bands_background").glob("*.npy"))]
uh = np.unique(hulls)
va_h = set(rng.choice(uh, len(uh) // 5, replace=False))
tr_c = [c for c, h in zip(clips, hulls) if h not in va_h]
va_c = [c for c, h in zip(clips, hulls) if h in va_h]
tr_bg, va_bg = np.where(~np.isin(st, [13, 9]))[0], np.where(np.isin(st, [13, 9]))[0]


def sea_minutes(n_windows):
    m = 10 ** (bg[train.balanced_bg(np.arange(len(bg)), st, tr_bg, rng, n_windows)].reshape(-1, 25).astype(np.float64) / 10)
    return m[np.isfinite(m).all(1)]


zero_X = bg[rng.choice(va_bg, 4000)]
ship_X = np.stack([10 * np.log10(mix.passage_minutes(va_c[i], rng, {**mix.CFG, "tilt_db": 0, "bump_db": 0, "stretch": (1, 1)}) + 1e-30)
                   for i in rng.integers(0, len(va_c), 1500)])
ship_X[ship_X < -200] = np.nan  # minutes outside the pass carry no energy: treat as missing
ship_X = ship_X[np.isfinite(ship_X).all(2).sum(1) >= 3]
Xm, ym, _, _ = mix.make_windows(bg[rng.choice(va_bg, 6000)], va_c, 6000, rng, cfg={**mix.CFG, "ratio_db": (-30, 5)})
sea20 = sea_minutes(20000)
sea60 = sea_minutes(60000)


def report(name, D_ship, D_sea, pen=0.0):
    z, _ = nmf.split(zero_X, D_ship, D_sea, ship_penalty=pen)
    o, _ = nmf.split(ship_X, D_ship, D_sea, ship_penalty=pen)
    m, _ = nmf.split(Xm, D_ship, D_sea, ship_penalty=pen)
    print(f"{name:34s} plain bg (want 0): {z.mean():.3f} | ship only (want 1): {o.mean():.3f} | "
          f"mixes 1-RMSE {1 - train.rmse(np.clip(m, 0, 1), ym):.4f} corr {np.corrcoef(m, ym)[0, 1]:.3f}", flush=True)


ship_min = nmf.minute_spectra(tr_c)
D16 = nmf.learn(ship_min, 16)
if "--prune" not in sys.argv: report("v1: ship16 sea32", D16, nmf.learn(sea20[rng.choice(len(sea20), 60000, replace=False)], 32))
D_sea64 = nmf.learn(sea60[rng.choice(len(sea60), 150000, replace=False)], 64)
report("sea64, 3x more sea minutes", D16, D_sea64)
D_amb = nmf.learn(nmf.minute_spectra(amb), 8)
D_excess = nmf.learn_with_fixed(ship_min, 16, D_amb)
if "--prune" not in sys.argv: report("ship shapes minus VTUAD ambient", D_excess, D_sea64)
for pen in ([] if "--prune" in sys.argv else [0.05, 0.2]):
    report(f"sea64 + ship penalty {pen}", D16, D_sea64, pen)


def atom_usage(X, D_ship, D_sea):
    """Energy share each sea shape takes in windows X (iterate the split, read activations)."""
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


if "--prune" in sys.argv:
    ship_tr = np.stack([10 * np.log10(mix.passage_minutes(tr_c[i], rng, {**mix.CFG, "tilt_db": 0, "bump_db": 0, "stretch": (1, 1)}) + 1e-30)
                        for i in rng.integers(0, len(tr_c), 1500)])
    ship_tr[ship_tr < -200] = np.nan
    sea_tr = bg[rng.choice(tr_bg, 4000)]
    u_ship, u_sea = atom_usage(ship_tr, D16, D_sea64), atom_usage(sea_tr, D16, D_sea64)
    ratio = u_ship / (u_sea + 1e-9)
    for thr in [3.0, 1.5, 1.0]:
        keep = ratio < thr
        report(f"sea64 minus ship-like shapes (<{thr}x) n={keep.sum()}", D16, D_sea64[keep])
    for thr in [3.0, 1.5]:
        move = ratio >= thr
        report(f"ship-like sea shapes moved to ship (>={thr}x)", np.concatenate([D16, D_sea64[move]]), D_sea64[~move])

if "--calib" in sys.argv:
    from sklearn.isotonic import IsotonicRegression
    ship_tr = np.stack([10 * np.log10(mix.passage_minutes(tr_c[i], rng, {**mix.CFG, "tilt_db": 0, "bump_db": 0, "stretch": (1, 1)}) + 1e-30)
                        for i in rng.integers(0, len(tr_c), 1500)])
    ship_tr[ship_tr < -200] = np.nan
    ratio = atom_usage(ship_tr, D16, D_sea64) / (atom_usage(bg[rng.choice(tr_bg, 4000)], D16, D_sea64) + 1e-9)
    Xc, yc, _, _ = mix.make_windows(train.balanced_bg(bg, st, tr_bg, rng, 15000), tr_c, 15000, rng, cfg={**mix.CFG, "ratio_db": (-30, 5)})
    for name, Dsea in [("sea64", D_sea64), ("sea64 pruned <3x", D_sea64[ratio < 3])]:
        sc, _ = nmf.split(Xc, D16, Dsea)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(sc, yc)
        m, _ = nmf.split(Xm, D16, Dsea)
        z, _ = nmf.split(zero_X, D16, Dsea)
        print(f"{name:18s} raw 1-RMSE {1 - train.rmse(np.clip(m, 0, 1), ym):.4f} | curve 1-RMSE {1 - train.rmse(iso.predict(m), ym):.4f} "
              f"| plain bg after curve {iso.predict(z).mean():.3f}", flush=True)
