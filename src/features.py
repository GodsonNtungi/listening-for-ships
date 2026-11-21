"""Window features from a (n, 15, 25) dB array with NaN minutes. Uses only the window itself."""
import warnings

import numpy as np

GROUPS = {"low": slice(0, 9), "mid": slice(9, 16), "high": slice(16, 25)}  # 31.6-200 Hz, 250-1k, 1.26-7.9k


def _db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def _series_stats(s, name):
    """s: (n, 15) dB time series with NaN -> dict of shape-of-passage stats."""
    mx, mn = np.nanmax(s, 1), np.nanmin(s, 1)
    med = np.nanmedian(s, 1)
    edges = np.nanmean(np.concatenate([s[:, :3], s[:, -3:]], 1), 1)
    filled = np.where(np.isnan(s), med[:, None], s)
    d1 = np.diff(filled, axis=1)
    return {
        f"{name}_range": mx - mn, f"{name}_peak_over_med": mx - med, f"{name}_med_over_min": med - mn,
        f"{name}_std": np.nanstd(s, 1), f"{name}_argmax": np.nanargmax(np.where(np.isnan(s), -np.inf, s), 1),
        f"{name}_peak_over_edges": mx - edges, f"{name}_rough": np.abs(d1).mean(1),
        f"{name}_smooth": np.abs(d1).mean(1) / (mx - mn + 1e-3),
    }


def build(X):
    X = X.astype(np.float64)
    L = 10 ** (X / 10)
    S = np.sort(X, 1)  # NaN sorts last
    nobs = np.isfinite(X).sum(1, keepdims=True)

    def pct(q):  # nearest-rank percentile over time, per band
        return np.take_along_axis(S, np.round(q * (nobs - 1)).astype(int), 1)[:, 0, :]

    f = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_b = pct(0.5)
        for name, v in {"med": med_b, "min": S[:, 0, :], "max": pct(1.0), "std": np.nanstd(X, 1),
                        "p25": pct(0.25), "p75": pct(0.75),
                        "linmean": _db(np.nanmean(L, 1))}.items():
            for b in range(25):
                f[f"b{b:02d}_{name}"] = v[:, b]
        rel = med_b - med_b.mean(1, keepdims=True)
        for b in range(25):
            f[f"b{b:02d}_shape"] = rel[:, b]
            f[f"b{b:02d}_maxmin"] = f[f"b{b:02d}_max"] - f[f"b{b:02d}_min"]
        tot = np.nansum(L, (1, 2))
        for q, floor in {"min": 10 ** (S[:, :1, :] / 10), "p20": 10 ** (pct(0.2)[:, None, :] / 10),
                         "p40": 10 ** (pct(0.4)[:, None, :] / 10)}.items():
            ex = np.nansum(np.maximum(L - floor, 0), (1, 2))
            f[f"excess_{q}"] = ex / tot
            for g, sl in GROUPS.items():
                f[f"excess_{q}_{g}"] = np.nansum(np.maximum(L[:, :, sl] - floor[:, :, sl], 0), (1, 2)) / tot
        # energy share of each group, and of the loudest minute
        for g, sl in GROUPS.items():
            f[f"share_{g}"] = np.nansum(L[:, :, sl], (1, 2)) / tot
        minute_tot = np.nansum(L, 2)
        minute_tot[np.isnan(X).all(2)] = np.nan
        f["loudest_minute_share"] = np.nanmax(minute_tot, 1) / tot
        series = {"all": _db(minute_tot)}
        for g, sl in GROUPS.items():
            s = _db(np.nansum(L[:, :, sl], 2))
            s[np.isnan(X).all(2)] = np.nan
            series[g] = s
        for name, s in series.items():
            f.update(_series_stats(s, name))
        f["low_minus_high_range"] = f["low_range"] - f["high_range"]
        # do bands rise and fall together (a passing source) or independently
        Z = X - np.nanmean(X, 1, keepdims=True)
        Z = np.where(np.isnan(Z), 0, Z)
        Z = Z / (np.sqrt((Z ** 2).sum(1, keepdims=True)) + 1e-9)
        C = np.einsum("ntb,ntc->nbc", Z, Z)
        iu = np.triu_indices(25, 1)
        f["corr_all"] = C[:, iu[0], iu[1]].mean(1)
        lo = np.triu_indices(9, 1)
        f["corr_low"] = C[:, :9, :9][:, lo[0], lo[1]].mean(1)
        f["corr_low_high"] = C[:, :9, 16:].mean((1, 2))
        slope_x = np.arange(25) - 12
        f["slope_med"] = (med_b * slope_x).sum(1) / (slope_x ** 2).sum()
        f["slope_maxmin"] = ((f_mm := np.stack([f[f"b{b:02d}_maxmin"] for b in range(25)], 1)) * slope_x).sum(1) / (slope_x ** 2).sum()
        f["maxmin_mean"] = f_mm.mean(1)
        f["n_missing"] = np.isnan(X).all(2).sum(1)
        grid = X - med_b[:, None, :]
        for t in range(15):
            for b in range(0, 25, 2):
                f[f"g{t:02d}_{b:02d}"] = grid[:, t, b]
    names = list(f)
    return np.column_stack([np.asarray(f[k], np.float32) for k in names]), names
