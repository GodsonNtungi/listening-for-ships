"""Make labelled training windows: real SA background + a VTUAD passage at a known energy share."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEG_PER_MIN = 30  # 2 s segments
WIN = 15
LOGF = np.log10(10.0 ** (1.5 + 0.1 * np.arange(25)))


def load_backgrounds():
    z = np.load(ROOT / "data" / __import__("os").environ.get("SA_WINDOWS", "sa_windows.npz"), allow_pickle=False)
    return z["X"], z["station"], z["recorder"]


def load_passages():
    meta = pd.read_csv(ROOT / "data/vtuad/registry_anonymized.csv")
    meta["n"] = meta["path"].str.extract(r"vessel/(\d+)\.wav")[0]
    meta = meta.dropna(subset=["n"]).set_index("n")
    clips, hulls, labels = [], [], []
    for f in sorted((ROOT / "data/vtuad/bands").glob("*.npy"), key=lambda p: int(p.stem)):
        p = np.load(f)
        if len(p) < SEG_PER_MIN or not np.isfinite(p).all():
            continue
        clips.append(p.astype(np.float64))
        hulls.append(str(meta.loc[f.stem, "hull_id"]) if f.stem in meta.index else f"clip{f.stem}")
        labels.append(meta.loc[f.stem, "label"] if f.stem in meta.index else "unknown")
    return clips, np.array(hulls), np.array(labels)


_BGC = []


def bg_clips():
    if not _BGC:
        _BGC.extend(np.load(f).astype(np.float64) for f in sorted((ROOT / "data/vtuad/bands_background").glob("*.npy")))
    return _BGC


CFG = dict(p_zero=0.2, ratio_db=(-25, 15), stretch=(0.5, 2.0), tilt_db=8.0, bump_db=3.0, p_bgship=0.0, bgship_db=(-20, 5),
           p_env=0.0, env_width_min=(1.0, 10.0))


def pass_gain_db(rng, total):
    """(total, 25) dB gain of a shallow-water pass: spreading, absorption, surface reflection, low cut, louder stern.
    Ranges from the physics notes (JOMOPANS-ECHO, Thorp, Lloyd's mirror); shallow-site angle rule is a rough guess."""
    f = 10.0 ** LOGF
    d, v, H = np.exp(rng.uniform(np.log(100), np.log(3000))), 0.514 * rng.uniform(3, 18), rng.uniform(15, 35)
    ts = (np.arange(total) / SEG_PER_MIN - rng.uniform(-3, WIN + 3)) * 60
    R = np.sqrt(d ** 2 + (v * ts) ** 2 + H ** 2)[:, None]
    fk = f / 1000
    absorb = 0.11 * fk ** 2 / (1 + fk ** 2) + 44 * fk ** 2 / (4100 + fk ** 2) + 2.75e-4 * fk ** 2 + 0.003
    absorb = absorb + rng.uniform(0.5, 3) * np.clip(np.log10(fk) / 0.9, 0, 1)  # seabed loss above 1 kHz
    y = 1.41 * 2 * np.pi * f / 1500 * rng.uniform(2, 7) * np.minimum(0.47, np.sqrt(H / R))
    w = np.sqrt(d ** 2 + H ** 2) / v
    stern = rng.uniform(3, 8) * (-1 if rng.random() < 0.2 else 1) / 2 * np.tanh(ts / (0.5 * w))[:, None]
    g = (-rng.uniform(15, 20) * np.log10(R) - absorb * R / 1000 + 10 * np.log10(2 * y ** 2 / (1 + y ** 2))
         - 10 * np.log10(1 + (rng.uniform(15, 60) / f) ** 8) * (1 - np.exp(-R / (5 * H))) + stern)
    return g - g.max()


def passage_minutes(p, rng, cfg=CFG):
    """One clip -> (15, 25) minute power, with random time stretch, placement and channel colouring."""
    stretch = np.exp(rng.uniform(np.log(cfg["stretch"][0]), np.log(cfg["stretch"][1])))
    n = max(SEG_PER_MIN, int(round(len(p) * stretch)))
    src = np.linspace(0, len(p) - 1, n)
    logp = np.log(p + 1e-30)
    q = np.exp(np.stack([np.interp(src, np.arange(len(p)), logp[:, b]) for b in range(25)], 1))
    total = WIN * SEG_PER_MIN
    if n >= total:
        s = (n - total) // 2 if cfg.get("center") else rng.integers(0, n - total + 1)
        seq = q[s:s + total]
    else:
        seq = np.zeros((total, 25))
        o = (total - n) // 2 if cfg.get("center") else rng.integers(-n + 2 * SEG_PER_MIN, total - 2 * SEG_PER_MIN + 1)  # at least 2 minutes inside
        a, b = max(o, 0), min(o + n, total)
        seq[a:b] = q[a - o:b - o]
    if rng.random() < cfg.get("p_phys", 0.0):
        seq = seq * 10 ** (pass_gain_db(rng, total) / 10)
    elif rng.random() < cfg["p_env"]:
        # strict clips sit at closest approach and are nearly flat; a real pass swells and fades with
        # distance: intensity ~ 1 / (cpa^2 + (v t)^2), i.e. a Lorentzian in time
        t = np.arange(total) / SEG_PER_MIN
        t0 = rng.uniform(-3, WIN + 3)
        w = np.exp(rng.uniform(*np.log(cfg["env_width_min"])))
        seq = seq * (1 / (1 + ((t - t0) / w) ** 2))[:, None]
    tilt = rng.uniform(-cfg["tilt_db"], cfg["tilt_db"]) * (LOGF - LOGF.mean())
    bumps = sum(rng.uniform(-cfg["bump_db"], cfg["bump_db"]) * np.cos(np.pi * rng.uniform(0.5, 3) * (LOGF - LOGF[0]) / 2.4 + rng.uniform(0, 2 * np.pi)) for _ in range(2))
    seq = seq * 10 ** ((tilt + bumps) / 10)
    return seq.reshape(WIN, SEG_PER_MIN, 25).mean(1)


def draw_share(rng, n, cfg=CFG):
    ratio_db = rng.uniform(*cfg["ratio_db"], n)
    s = 1 / (1 + 10 ** (-ratio_db / 10))
    s[rng.random(n) < cfg["p_zero"]] = 0.0
    return s


def sa_events(X, rng, n, min_share=0.2):
    """Real SA ship-like passes: (n, 15, 25) linear power above 3x each band's 3rd quietest minute, low-pitched and peaked."""
    X = X[np.isfinite(X).all((1, 2))]
    out = []
    for i in range(0, len(X), 20_000):
        L = 10 ** (X[i:i + 20_000].astype(np.float64) / 10)
        ex = np.clip(L - 3 * np.sort(L, 1)[:, 2:3], 0, None)
        pm = ex.sum(2)
        keep = (ex.sum((1, 2)) / L.sum((1, 2)) > min_share) & (ex[:, :, :16].sum((1, 2)) > 0.7 * ex.sum((1, 2))) & (pm.max(1) > 2 * pm.mean(1))
        out.append(ex[keep])
    ev = np.concatenate(out)
    return ev[rng.choice(len(ev), min(n, len(ev)), replace=False)]


def make_windows(bg, clips, n, rng, extra_drop=0.38, cfg=CFG, return_nat=False, nat_pool=None):
    """bg: (m, 15, 25) dB backgrounds to sample from; clips: list of passages. Returns X dB, y share, clip idx, bg idx.
    nat_pool: (k, 15, 25) linear SA events used for the uncounted natural ship instead of VTUAD clips."""
    bi = rng.integers(0, len(bg), n)
    ci = rng.integers(0, len(clips), n)
    want = draw_share(rng, n, cfg)
    X = np.empty((n, WIN, 25), np.float32)
    y = np.zeros(n)
    nat = np.zeros(n)
    for k in range(n):
        B = 10 ** (bg[bi[k]].astype(np.float64) / 10)
        obs = np.isfinite(B).all(1)
        if obs.all() and rng.random() < extra_drop:
            obs[rng.integers(0, WIN)] = False
        B[~obs] = np.nan
        if rng.random() < cfg["p_bgship"]:  # a natural ship already in the background: not part of the answer
            N = nat_pool[rng.integers(len(nat_pool))] if nat_pool is not None else passage_minutes(clips[rng.integers(0, len(clips))], rng, {**cfg, **cfg.get("nat", {})})
            r = 10 ** (rng.uniform(*cfg["bgship_db"]) / 10)
            if N[obs].sum() > 0:
                add = r * B[obs].sum() / N[obs].sum() * N
                B = B + add
                nat[k] = add[obs].sum()
        if cfg.get("p_ctrl", 0) > 0 and rng.random() < cfg["p_ctrl"]:  # uncounted VTUAD sea-only clip
            pool = bg_clips()[slice(*cfg.get("ctrl_clips", (0, 16)))]
            C = passage_minutes(pool[rng.integers(len(pool))], rng, cfg)
            add = 10 ** (rng.uniform(*cfg["ctrl_db"]) / 10) * B[obs].sum() / C[obs].sum() * C
            B = B + add
            nat[k] += add[obs].sum()
        r2 = __import__("copy").deepcopy(rng) if cfg.get("label_raw") and want[k] > 0 else None
        V = passage_minutes(clips[ci[k]], rng, cfg) if want[k] > 0 else np.zeros((WIN, 25))
        sv, sb = V[obs].sum(), B[obs].sum()
        scale = want[k] / (1 - want[k]) * sb / sv if sv > 0 and want[k] > 0 else 0.0
        M = B + scale * V
        if r2 is not None:  # H5: the answer measures the passage before its tone change
            U = passage_minutes(clips[ci[k]], r2, {**cfg, "tilt_db": 0.0, "bump_db": 0.0})
            y[k] = min(scale * U[obs].sum() / M[obs].sum(), 1.0)
            nat[k] /= M[obs].sum()
            X[k] = 10 * np.log10(M)
            continue
        t = cfg.get("target", "total")
        if t == "minute" and want[k] > 0:
            y[k] = np.mean(scale * V[obs].sum(1) / M[obs].sum(1))
        elif t == "cell" and want[k] > 0:
            y[k] = np.mean(scale * V[obs] / M[obs])
        else:
            y[k] = scale * sv / M[obs].sum()
        nat[k] /= M[obs].sum()
        X[k] = 10 * np.log10(M)
    return (X, y, ci, bi, nat) if return_nat else (X, y, ci, bi)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bg, st, _ = load_backgrounds()
    clips, hulls, _ = load_passages()
    X, y, _, _ = make_windows(bg, clips, 2000, rng)
    assert X.shape == (2000, 15, 25) and np.all((y >= 0) & (y < 1))
    assert np.allclose(np.isnan(X).any(2).sum(1) <= 2, True)
    print(f"{len(clips)} clips, {len(bg)} backgrounds; y mean {y.mean():.3f}, zeros {np.mean(y == 0):.2f}, "
          f"missing-minute windows {np.isnan(X).any((1, 2)).mean():.2f}")
