"""Which local validation recipe ranks our submitted methods the way the public board does?

Every method is rebuilt on a training fold (stations 13+9 and 20% of hulls held out), then scored on
held-out windows built under each candidate recipe. Only public scores of whole submissions are used.
"""
import pickle
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import floor
import mix
import nmf
import train

PUBLIC = {"lgb_v1": 0.6836, "lgb_v3_faint": 0.7003, "lgb_v5_clean": 0.6909, "nmf_v1": 0.7662, "nmf_v2_pruned_curve": 0.7461,
          "nmf_v2_pruned": 0.6993, "nmf_v1_floorcap": 0.7748, "nmf_v1_steady_floorcap": 0.7776,
          "lgb_v8_region": 0.7859, "lgb_v8_broad": 0.8109, "lgb_v8_noenv": 0.8044}
CACHE = mix.ROOT / "models/lbcheck_methods.pkl"

rng = np.random.default_rng(train.SEED)
bg, st, _ = mix.load_backgrounds()
clips, hulls, _ = mix.load_passages()
uh = np.unique(hulls)
va_h = set(rng.choice(uh, len(uh) // 5, replace=False))
tr_c = [c for c, h in zip(clips, hulls) if h not in va_h]
va_c = [c for c, h in zip(clips, hulls) if h in va_h]
tr_bg, va_bg = np.where(~np.isin(st, [13, 9]))[0], np.where(np.isin(st, [13, 9]))[0]


def usage(X, D_ship, D_sea):
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


def build_methods():
    r = np.random.default_rng(train.SEED)
    m = {}
    for tag, extra in [("lgb_v1", {}), ("lgb_v3_faint", dict(ratio_db=(-30, 5)))]:
        F, y, _ = train.dataset(train.balanced_bg(bg, st, tr_bg, r, 80_000), tr_c, 80_000, r, {**mix.CFG, **extra})
        m[tag] = lgb.train(train.PARAMS, lgb.Dataset(F, y), 800).model_to_string()
        print("trained", tag, flush=True)
    ship_min = nmf.minute_spectra(tr_c)
    m["D16"] = nmf.learn(ship_min, 16)
    sea = 10 ** (bg[train.balanced_bg(np.arange(len(bg)), st, tr_bg, r, 20000)].reshape(-1, 25).astype(np.float64) / 10)
    sea = sea[np.isfinite(sea).all(1)]
    m["D32"] = nmf.learn(sea[r.choice(len(sea), 60000, replace=False)], 32)
    sea = 10 ** (bg[train.balanced_bg(np.arange(len(bg)), st, tr_bg, r, 60000)].reshape(-1, 25).astype(np.float64) / 10)
    sea = sea[np.isfinite(sea).all(1)]
    D64 = nmf.learn(sea[r.choice(len(sea), 150000, replace=False)], 64)
    ship_X = np.stack([10 * np.log10(mix.passage_minutes(tr_c[i], r, {**mix.CFG, "tilt_db": 0, "bump_db": 0, "stretch": (1, 1)}) + 1e-30)
                       for i in r.integers(0, len(tr_c), 2000)])
    ship_X[ship_X < -200] = np.nan
    keep = usage(ship_X, m["D16"], D64) / (usage(bg[r.choice(tr_bg, 5000)], m["D16"], D64) + 1e-9) < 3
    m["D64p"] = D64[keep]
    Xc, yc, _, _ = mix.make_windows(train.balanced_bg(bg, st, tr_bg, r, 20000), tr_c, 20000, r, cfg={**mix.CFG, "ratio_db": (-30, 5)})
    m["iso"] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(nmf.split(Xc, m["D16"], m["D64p"])[0], yc)
    return m


def add_v5(m):
    """lgb_v5: v3 recipe on blip-free backgrounds (sa_windows.py --clean)."""
    z = np.load(mix.ROOT / "data/sa_windows_clean.npz")
    cbg, cst = z["X"], z["station"]
    r = np.random.default_rng(train.SEED)
    idx = np.where(~np.isin(cst, [13, 9]))[0]
    F, y, _ = train.dataset(train.balanced_bg(cbg, cst, idx, r, 80_000), tr_c, 80_000, r, {**mix.CFG, "ratio_db": (-30, 5)})
    m["lgb_v5_clean"] = lgb.train(train.PARAMS, lgb.Dataset(F, y), 800).model_to_string()


def predict_all(m, X):
    F, _ = train.build(X)
    out = {t: np.clip(lgb.Booster(model_str=m[t]).predict(F), 0, 1) for t in ("lgb_v1", "lgb_v3_faint", "lgb_v5_clean")}
    v1 = np.clip(nmf.split(X, m["D16"], m["D32"])[0], 0, 1)
    pr = np.clip(nmf.split(X, m["D16"], m["D64p"])[0], 0, 1)
    out.update(nmf_v1=v1, nmf_v2_pruned=pr, nmf_v2_pruned_curve=m["iso"].predict(pr),
               nmf_v1_floorcap=np.minimum(v1, floor.share(X, 3.0, 2)))
    # not yet submitted: out-of-sample check for whichever recipe wins
    out["nmf_v1_steady_floorcap"] = np.minimum(np.clip(nmf.split(X, m["D16"], m["D32"], steady_q=0)[0], 0, 1), floor.share(X, 3.0, 2))
    import v8
    v8.RICH = True
    F8 = v8.features(X, m["D16"], m["D32"])
    for tag, f in [("lgb_v8_region", "lgb_v8_fold_rich_region.txt"), ("lgb_v8_broad", "lgb_v8_fold_rich.txt"),
                   ("lgb_v8_noenv", "lgb_v8_fold_rich_lgb_v8_noenv.txt")]:
        out[tag] = np.clip(lgb.Booster(model_file=str(mix.ROOT / "models" / f)).predict(F8), 0, 1)
    return out


RECIPES = {}
if "--grid2" in sys.argv:  # around the first grid's best: faint shares, natural ships inside backgrounds
    for prior_name, prior in [("faint(-30,5)", (-30, 5)), ("vfaint(-40,0)", (-40, 0)), ("mid(-30,10)", (-30, 10))]:
        for bgs in (0.4, 0.7, 1.0):
            for bgdb in ((-20, 5), (-10, 10)):
                for pz in (0.2, 0.5):
                    RECIPES[f"{prior_name} zeros{pz} bgship{bgs} bgdb{bgdb}"] = dict(ratio_db=prior, p_zero=pz, p_bgship=bgs, bgship_db=bgdb)
if "--mixstyle" in sys.argv:  # same loudness recipe, but the counted and uncounted ships mixed in different ways
    base = dict(ratio_db=(-30, 5), p_zero=0.5, p_bgship=0.7, bgship_db=(-10, 10))
    for name, extra in [("same", {}), ("nat long", dict(nat=dict(stretch=(3, 6)))), ("nat short", dict(nat=dict(stretch=(0.5, 1)))),
                        ("nat swell", dict(nat=dict(p_env=1.0))), ("added swell", dict(p_env=1.0, nat=dict(p_env=0.0))),
                        ("both swell", dict(p_env=1.0)), ("added centred", dict(center=True, nat=dict(center=False))),
                        ("nat centred", dict(nat=dict(center=True))), ("added long", dict(stretch=(3, 6), nat=dict(stretch=(0.5, 2)))),
                        ("added short", dict(stretch=(0.5, 1), nat=dict(stretch=(0.5, 2)))), ("nat flat colour", dict(nat=dict(tilt_db=0, bump_db=0))),
                        ("added flat colour", dict(tilt_db=0, bump_db=0, nat=dict(tilt_db=8.0, bump_db=3.0)))]:
            RECIPES[name] = {**base, **extra}
for prior_name, prior in [] if RECIPES else [("wide(-25,15)", dict(ratio_db=(-25, 15))), ("faint(-30,5)", dict(ratio_db=(-30, 5))),
                          ("vfaint(-40,0)", dict(ratio_db=(-40, 0))), ("mid(-20,0)", dict(ratio_db=(-20, 0)))]:
    for pz in (0.2, 0.5):
        for env in (0.0, 1.0):
            for bgs in (0.0, 0.4):
                RECIPES[f"{prior_name} zeros{pz} env{env:.0f} bgship{bgs}"] = {**prior, "p_zero": pz, "p_env": env, "p_bgship": bgs}

if __name__ == "__main__":
    if CACHE.exists():
        m = pickle.loads(CACHE.read_bytes())
    else:
        m = build_methods()
        CACHE.write_bytes(pickle.dumps(m))
    if "lgb_v5_clean" not in m:
        add_v5(m)
        CACHE.write_bytes(pickle.dumps(m))
    names = list(PUBLIC)
    pub = np.array([PUBLIC[n] for n in names])
    rows = []
    for rname, cfg in RECIPES.items():
        r = np.random.default_rng(7)
        X, y, _, _ = mix.make_windows(bg[r.choice(va_bg, 4000)], va_c, 4000, r, cfg={**mix.CFG, **cfg})
        P = predict_all(m, X)
        loc = np.array([1 - train.rmse(P[n], y) for n in names])
        rho, pr = spearmanr(loc, pub)[0], pearsonr(loc, pub)[0]
        mae = np.abs(loc - pub).mean()
        steady = 1 - train.rmse(P["nmf_v1_steady_floorcap"], y)
        # leave-one-out: fit public = a + b * local on the other methods, predict the held-out one
        loo = np.array([np.polyval(np.polyfit(np.delete(loc, i), np.delete(pub, i), 1), loc[i]) - pub[i] for i in range(len(loc))])
        pred = np.polyval(np.polyfit(loc, pub, 1), steady)
        rows.append((rho, pr, np.abs(loo).max(), rname, loc, steady, y.mean(), pred, np.abs(loo).mean()))
        print(f"{rname:42s} rank-agree {rho:+.2f} lin {pr:+.2f} gap {mae:.3f} | y mean {y.mean():.3f} | "
              + " ".join(f"{v:.3f}" for v in loc) + f" | steady {steady:.3f} -> public {pred:.3f} | LOO err mean {np.abs(loo).mean():.3f} max {np.abs(loo).max():.3f}", flush=True)
    print("order:", names)
    rows.sort(key=lambda t: (-t[0], -t[1]))
    print("\nbest by rank agreement then linear agreement:")
    for t in rows[:6]:
        print(f"  {t[3]:52s} rank {t[0]:+.2f} lin {t[1]:+.2f} LOO mean {t[8]:.3f} max {t[2]:.3f} | steady local {t[5]:.3f} -> predicted public {t[7]:.4f}")
