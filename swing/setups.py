"""Stage 2 setup detection + per-symbol raw factors. Every column at row t uses data up to t only."""
import numpy as np
import pandas as pd

from . import config as C
from .indicators import atr, bb_width, clv, ema, lin, rsi, sma, trend_r2, windows


def _masked(win, mask, fn):
    return fn(np.where(mask, win, np.nan), axis=1)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def features(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    with _quiet(), np.errstate(all="ignore"):
        return _features(df, spy_close)


def _features(df, spy_close):
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["atr"] = a = atr(df)
    f["vol50"] = vol50 = sma(v, 50).shift(1)
    f["rvol"] = rvol = v / vol50
    f["dollar_vol"] = sma(c * v, 50)
    f["clv"] = cl = clv(df)
    s50, s150, s200, e21 = sma(c, 50), sma(c, 150), sma(c, 200), ema(c, 21)
    f["sma50"], f["sma200"], f["ema21"] = s50, s200, e21
    hi52, lo52 = h.rolling(252, min_periods=252).max(), l.rolling(252, min_periods=252).min()
    f["trend_template"] = ((c > s50) & (s50 > s150) & (s150 > s200) & (s200 > s200.shift(20))
                           & (c >= 1.3 * lo52) & (c >= 0.75 * hi52))

    # --- raw factors (ranked cross-sectionally later) ---
    spy = spy_close.reindex(df.index).ffill()
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["mom_6_1"] = c.shift(21) / c.shift(126) - 1
    rel = lambda n: (c / c.shift(n)) - (spy / spy.shift(n))
    f["rs_raw"] = 0.4 * rel(63) + 0.2 * rel(126) + 0.2 * rel(189) + 0.2 * rel(252)
    f["high52"] = c / hi52
    f["trend_q"] = trend_r2(c, 63)
    up = (c > c.shift()).astype(float)
    f["vol_accum"] = (v * up).rolling(50).sum() / (v * (1 - up)).rolling(50).sum().replace(0, np.nan)

    n = len(df)
    H, L, V = windows(h, 65), windows(l, 65), windows(v, 65)
    av, cv = a.to_numpy(), c.to_numpy()
    pos = np.arange(65)

    # --- Setup A: base breakout ---
    valid = ~np.isnan(H).any(axis=1)
    am = np.where(valid, np.argmax(np.nan_to_num(H, nan=-1), axis=1), 0)
    pivot = np.where(valid, H[np.arange(n), am], np.nan)
    base_len = 65 - am
    post = pos >= am[:, None]
    base_low = _masked(L, post, np.nanmin)
    depth = (pivot - base_low) / pivot
    b1, b2 = am + base_len // 3, am + 2 * base_len // 3
    seg = lambda lo, hi: (pos >= lo[:, None]) & (pos < hi[:, None])
    s1, s2, s3 = seg(am, b1), seg(b1, b2), seg(b2, np.full(n, 65))
    rng = lambda m: (_masked(H, m, np.nanmax) - _masked(L, m, np.nanmin)) / pivot
    r1, r2, r3 = rng(s1), rng(s2), rng(s3)
    seg3_low = _masked(L, s3, np.nanmin)
    bbw = bb_width(c)
    bbw_pct = (bbw.rolling(252, min_periods=126).rank(pct=True) * 100).shift(1).to_numpy()
    # quietest 5-day stretch in the final 10 days of the base vs. normal volume
    dryup = (v.rolling(5).mean().rolling(10).min().shift(1) / vol50).to_numpy()
    contraction = ((r1 > r2) & (r2 > r3)) | (bbw_pct <= 20)
    a_sig = (f["trend_template"].to_numpy() & valid & (base_len >= C.BASE_MIN) & (base_len <= C.BASE_MAX)
             & (depth <= C.BASE_MAX_DEPTH) & contraction & (dryup < 0.75)
             & (cv > pivot) & (rvol.to_numpy() >= C.A_RVOL) & (cl.to_numpy() >= C.A_CLV)
             & (cv <= pivot + C.A_MAX_EXT_ATR * av))
    f["A"] = a_sig
    f["A_pivot"], f["A_depth"], f["A_base_len"] = pivot, depth, base_len
    f["A_contraction"] = r3 / r1
    f["A_bbw_pct"], f["A_dryup"] = bbw_pct, dryup
    f["A_quality"] = np.nanmean([lin(depth, 0.30, 0.08), lin(bbw_pct, 60, 0), lin(dryup, 1.0, 0.4),
                                 lin(rvol, 1.0, 3.0)], axis=0)
    f["A_stop"] = seg3_low - C.STOP_BUFFER_ATR * av
    f["A_t2_add"] = pivot - base_low          # measured move
    f["A_max_entry"] = pivot + C.A_MAX_EXT_ATR * av

    # --- Setup B: pullback in uptrend ---
    H20, V20, L20, C20 = windows(h, 20), windows(v, 20), windows(l, 20), windows(c, 20)
    v20 = ~np.isnan(H20).any(axis=1)
    am20 = np.where(v20, np.argmax(np.nan_to_num(H20, nan=-1), axis=1), 0)
    hi20 = np.where(v20, H20[np.arange(n), am20], np.nan)
    pb = pos[:20] >= am20[:, None]
    prev_c = c.shift(1).to_numpy()
    pullback = 1 - prev_c / hi20
    pb_vol = _masked(V20, pb, np.nanmean) / vol50.to_numpy()
    bad = ((c < s50) & (rvol > 1.5)).astype(float)
    bad_in_pb = _masked(windows(bad, 20), pb, np.nanmax) > 0
    rsi2_min = _masked(windows(rsi(c, 2), 20), pb, np.nanmin)
    r14 = rsi(c, 14).shift(1).to_numpy()
    touch = lambda ma: ((l <= ma + 0.5 * a) & (c >= ma - 0.5 * a) & (ma > ma.shift(5)))
    touched = (touch(e21) | touch(s50)).astype(float).rolling(5).max().shift(1).to_numpy() > 0
    dist_ma = (np.minimum((prev_c - e21.shift(1)).abs(), (prev_c - s50.shift(1)).abs()) / a.shift(1)).to_numpy()
    b_sig = (f["trend_template"].to_numpy() & v20 & (pullback >= 0.03) & (pullback <= 0.12) & touched
             & (pb_vol < 0.9) & ~bad_in_pb & ((rsi2_min <= 15) | ((r14 >= 38) & (r14 <= 52)))
             & (cv > h.shift(1).to_numpy()) & (cl.to_numpy() >= 0.5))
    f["B_raw"] = b_sig      # RS-rank >= 70 is applied after cross-sectional ranking
    f["B_pullback"], f["B_pb_vol"], f["B_dist_ma"] = pullback, pb_vol, dist_ma
    f["B_quality_part"] = np.nanmean([lin(pb_vol, 1.0, 0.5), lin(dist_ma, 1.5, 0)], axis=0)
    low5 = l.rolling(5).min()
    f["B_stop"] = low5 - C.STOP_BUFFER_ATR * a
    f["B_t2"] = hi20 + av
    f["B_max_entry"] = c + 0.5 * a

    # --- Setup C: post-earnings drift (price/volume proxy; earnings confirmed live) ---
    gap = o / c.shift(1) - 1
    gap_ok = (gap >= np.maximum(C.C_GAP_MIN, C.C_GAP_ATR * a.shift(1) / c.shift(1))) & (rvol >= C.C_RVOL)
    mid = (h + l) / 2
    c_sig = np.zeros(n, bool)
    gap_size, gap_rvol, gap_atr, gap_age = (np.full(n, np.nan) for _ in range(4))
    gap_date = np.full(n, None, object)
    idx = df.index
    for k in range(9, 3, -1):          # later iterations (smaller k = more recent gap) win
        g_ok = gap_ok.shift(k, fill_value=False).to_numpy(bool)
        held_low = (l.rolling(k - 1).min().shift(1) > l.shift(k)).to_numpy()
        held_mid = (c.rolling(k - 1).min().shift(1) >= mid.shift(k)).to_numpy()
        brk = (c > h.rolling(k).max().shift(1)).to_numpy()
        hit = g_ok & held_low & held_mid & brk & (cl.to_numpy() >= 0.5)
        c_sig |= hit
        gs = (o - c.shift(1)).shift(k).to_numpy()
        gap_size = np.where(hit, gs, gap_size)
        gap_rvol = np.where(hit, rvol.shift(k).to_numpy(), gap_rvol)
        gap_atr = np.where(hit, (gs / a.shift(k + 1)).to_numpy(), gap_atr)
        gap_age = np.where(hit, k, gap_age)
        if hit.any():
            src = np.where(hit)[0]
            gap_date[src] = [str(idx[i - k].date()) for i in src]
    f["C"] = c_sig
    f["C_gap_size"], f["C_gap_rvol"], f["C_gap_atr"], f["C_gap_age"] = gap_size, gap_rvol, gap_atr, gap_age
    f["C_gap_date"] = gap_date
    f["C_quality_part"] = np.nanmean([lin(gap_atr, 1.5, 4.0), lin(gap_rvol, 3, 8)], axis=0)
    f["C_stop"] = f["B_stop"]
    f["C_max_entry"] = f["B_max_entry"]
    return f
