"""Find each test window's background minutes in the released SA archive (exact 0.1 dB match)."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIN, NB = 15, 25


def load_test():
    t = pd.read_csv(ROOT / "data/Test.csv")
    bands = [c[4:] for c in t.columns if c.startswith("m01_")]
    Y = t[[f"m{m:02d}_{b}" for m in range(1, WIN + 1) for b in bands]].to_numpy().reshape(-1, WIN, NB)
    return t["ID"].to_numpy(), Y, bands


def load_archive(bands):
    segs = []
    for f in sorted((ROOT / "data/sa/acoustics").glob("*.parquet")):
        a = pd.read_parquet(f)
        segs.append((f.stem, a[bands].to_numpy(np.float64)))
    return segs


def key(row, idx):
    return tuple(np.round(row[idx] * 10).astype(int)) if np.isfinite(row[idx]).all() else None


def match(Y, segs, kb=(22, 23, 24), tol=0.051):
    """Per test window: best (seg, start, n_exact) where test >= archive and most cells equal."""
    idx = list(kb)
    table = defaultdict(list)
    for s, A in enumerate(segs):
        for r in range(len(A[1])):
            k = key(A[1][r], idx)
            if k is not None:
                table[k].append((s, r))
    out = []
    for w in range(len(Y)):
        votes = defaultdict(int)
        for m in range(WIN):
            k = key(Y[w, m], idx)
            for s, r in table.get(k, ()) if k else ():
                votes[(s, r - m)] += 1
        best = (None, None, -1)
        for (s, st), _ in sorted(votes.items(), key=lambda kv: -kv[1])[:20]:
            A = segs[s][1]
            if st < 0 or st + WIN > len(A):
                continue
            B = A[st:st + WIN]
            d = Y[w] - B
            ok = np.isfinite(d)
            n_exact = int((np.abs(d[ok]) < tol).sum())
            if (d[ok] >= -tol).mean() > 0.98 and n_exact > best[2]:
                best = (s, st, n_exact)
        out.append(best)
    return out


def share(Ydb, Bdb):
    y, b = 10 ** (Ydb / 10), 10 ** (Bdb / 10)
    ok = np.isfinite(y) & np.isfinite(b)
    return float(np.clip((y[ok] - b[ok]).sum() / y[ok].sum(), 0, 1))


if __name__ == "__main__":
    ids, Y, bands = load_test()
    segs = load_archive(bands)
    res = match(Y, segs)
    found = [r for r in res if r[0] is not None]
    print(f"matched {len(found)}/{len(Y)}; exact cells median {np.median([r[2] for r in found]) if found else 0}")
    sh = [share(Y[w], segs[s][1][st:st + WIN]) if s is not None else np.nan for w, (s, st, _) in enumerate(res)]
    pd.DataFrame({"ID": ids, "seg": [r[0] for r in res], "start": [r[1] for r in res],
                  "n_exact": [r[2] for r in res], "share_total": sh}).to_csv(ROOT / "logs/bgmatch.csv", index=False)
