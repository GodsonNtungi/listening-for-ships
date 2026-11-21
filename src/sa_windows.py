"""Cut the South African recordings into 15-minute x 25-band background windows (dB).

Output data/sa_windows.npz: X (n, 15, 25) float32 dB with NaN for missing minutes,
station (n,) int, recorder (n,) str, t0 (n,) datetime64[m].
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["31.6", "39.8", "50.1", "63.1", "79.4", "100", "125.9", "158.5", "199.5", "251.2", "316.2", "398.1", "501.2",
         "631", "794.3", "1000", "1258.9", "1584.9", "1995.3", "2511.9", "3162.3", "3981.1", "5011.9", "6309.6", "7943.3"]
COLS = [f"TOL_{b}" for b in BANDS]
WIN = 15
VALID_DB = (20.0, 180.0)  # Richards Bay carries +-156 dB sentinels; tighten if new junk shows up


TILED = "--tiled" in __import__("sys").argv
DROP_BLIPS = "--clean" in __import__("sys").argv  # lgb_v5 tried dropping blip minutes: public 0.691 vs 0.700, the test keeps them
rng = np.random.default_rng(7)


def tiled_for(df):
    """15-minute blocks from each break; bursts shorter than a window are glued to the next (minute positions ignore gaps)."""
    df = df.assign(t=df["t"].dt.round("min")).drop_duplicates("t").sort_values("t")
    x = df[COLS].to_numpy(np.float32)
    bad = (~np.isfinite(x)).any(1) | (x < VALID_DB[0]).any(1) | (x > VALID_DB[1]).any(1) | (df["dur"].to_numpy() < 55)
    x[bad] = np.nan
    seg = (df["t"].diff().dt.total_seconds().ne(60) | df["src_file"].ne(df["src_file"].shift())).cumsum().to_numpy()
    t = df["t"].to_numpy("datetime64[m]")
    w, t0, buf = [], [], []
    for _, idx in pd.Series(np.arange(len(x))).groupby(seg):
        idx = idx.to_numpy()
        if len(idx) >= WIN:
            buf = []
        buf.extend(idx)
        while len(buf) >= WIN:
            blk, buf = buf[:WIN], buf[WIN:]
            if np.isnan(x[blk]).any(1).sum() <= 1:
                w.append(x[blk]), t0.append(t[blk[0]])
    return (np.stack(w), np.array(t0)) if w else (np.empty((0, WIN, 25), np.float32), np.array([], "datetime64[m]"))


def windows_for(df):
    df = df.assign(t=df["t"].dt.round("min")).drop_duplicates("t").sort_values("t")
    # minutes at either edge of a pause or file change carry blips (up to +18 dB)
    restart = (df["t"].diff().dt.total_seconds().ne(60) | df["src_file"].ne(df["src_file"].shift())).to_numpy()
    x = df[COLS].to_numpy(np.float32)
    bad = (~np.isfinite(x)).any(1) | (x < VALID_DB[0]).any(1) | (x > VALID_DB[1]).any(1) | (df["dur"].to_numpy() < 55)
    x[bad] = np.nan
    minute = ((df["t"] - df["t"].iloc[0]).dt.total_seconds() // 60).to_numpy(np.int64)
    grid = np.full((minute[-1] + 1, 25), np.nan, np.float32)
    grid[minute] = x
    if DROP_BLIPS:
        hard = np.zeros(len(grid), bool)
        hard[minute[restart]] = True
        hard[minute[np.roll(restart, -1)]] = True  # and the last minute before one
        # some recorders blip at fixed minutes of every hour (Richards Bay: +10-20 dB at :00)
        lv = pd.Series(np.where(bad, np.nan, 10 * np.log10((10 ** (x.astype(np.float64) / 10)).sum(1))), index=df.index)
        rel = lv - lv.rolling(31, center=True, min_periods=5).median()
        moh = df["t"].dt.minute
        blip_moh = rel.groupby(moh).mean()
        hard[minute[moh.isin(blip_moh.index[blip_moh > 3]).to_numpy()]] = True
        grid[hard] = np.nan
    # random window starts, so recorder blips land at any minute position, as they can in the test
    gap = np.concatenate([[0], np.cumsum(np.isnan(grid).any(1))])
    ok = np.flatnonzero(gap[WIN:] - gap[:-WIN] <= 1)  # the test has at most a few missing minutes, nearly always 0 or 1
    starts = np.sort(rng.choice(ok, min(len(ok), len(grid) // WIN), replace=False)) if len(ok) else ok
    starts = np.array(starts, dtype=np.int64)
    w = grid[starts[:, None] + np.arange(WIN)] if len(starts) else np.empty((0, WIN, 25), np.float32)
    t0 = (df["t"].iloc[0] + pd.to_timedelta(starts, "min")).to_numpy("datetime64[m]")
    return w, t0


def main():
    Xs, st, rec, t0s = [], [], [], []
    for f in sorted((ROOT / "data/sa/acoustics").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["site", "recorder", "src_file", "t", "dur"] + COLS)
        sid = int(f.name.split("_")[0])
        for r, g in d.groupby("recorder"):
            w, t0 = tiled_for(g) if TILED else windows_for(g)
            Xs.append(w); t0s.append(t0)
            st += [sid] * len(w); rec += [r] * len(w)
            print(f"{f.name:28s} {r:45s} rows={len(g):7d} windows={len(w):6d}")
    X = np.concatenate(Xs)
    np.savez_compressed(ROOT / ("data/sa_windows_tiled.npz" if TILED else "data/sa_windows_clean.npz" if DROP_BLIPS else "data/sa_windows.npz"), X=X, station=np.array(st), recorder=np.array(rec), t0=np.concatenate(t0s))
    print("total windows", len(X), "with a missing minute", int(np.isnan(X).any((1, 2)).sum()))


if __name__ == "__main__":
    main()
