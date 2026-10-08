"""Dip-buy rating (0-100) for the most-traded names, grouped by sector for the heat map.

rating = 0.40 x dip + 0.35 x trend + 0.25 x relative strength
  dip    how oversold/stretched below its 20-day high (RSI(2) 60 -> 0 pts, 5 -> 100; dip 0% -> 0, 10% -> 100)
  trend  100 in a confirmed uptrend, else distance below a rising-SMA200 uptrend (max 50)
  RS     cross-sectional relative-strength percentile
Not in an uptrend -> capped at 45 (the strategy never buys it). All dip rules met -> at least 75."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config as C, yahoo
from .indicators import lin

SECTORS = C.CACHE_DIR / "sectors.json"
HEATMAP = C.DATA_DIR / "heatmap.json"


def _sectors(symbols):
    cache = json.loads(SECTORS.read_text()) if SECTORS.exists() else {}
    missing = [s for s in symbols if s not in cache]
    if missing:
        def get(s):
            info = yahoo._safe(lambda: __import__("yfinance").Ticker(s).info, {}) or {}
            return s, {"sector": info.get("sector") or "Other", "name": info.get("shortName") or s}
        with ThreadPoolExecutor(6) as ex:
            cache.update(dict(ex.map(get, missing)))
        SECTORS.write_text(json.dumps(cache))
    return cache


def rating(row, rs_rank):
    up = bool(row["D_uptrend"])
    dip = float(np.nanmean([lin(row["D_rsi2"], 60, 5), lin(row["D_dip"], 0.0, 0.10)]))
    trend = 100.0 if up else float(lin((row["close"] - row["sma200"]) / row["sma200"], -0.15, 0.0)) * 0.5
    rs = 50.0 if rs_rank is None or not np.isfinite(rs_rank) else float(rs_rank)
    score = 0.40 * dip + 0.35 * trend + 0.25 * rs
    if not up:
        score = min(score, 45.0)
    if row["D"]:
        score = max(score, 75.0)
    return round(score), {"dip": round(dip), "trend": round(trend), "rs": round(rs)}


def build(feats, ranks, asof, dip_report=None, n=C.HEATMAP_N):
    """feats: {symbol: (df, features)} at the latest bar; ranks: DataFrame of factor ranks by symbol."""
    top = sorted(feats, key=lambda s: feats[s][1]["dollar_vol"].iloc[-1], reverse=True)[:n]
    meta = _sectors(top)
    per = {}
    if dip_report:
        for p in dip_report["styles"]["target"]["per_ticker"]:
            per[p["symbol"]] = p
    tiles = []
    for s in top:
        df, f = feats[s]
        row = f.iloc[-1]
        sc, parts = rating(row, ranks.loc[s, "rs_raw"] if s in ranks.index else None)
        tiles.append({
            "symbol": s, "name": meta.get(s, {}).get("name", s), "sector": meta.get(s, {}).get("sector", "Other"),
            "rating": sc, "parts": parts, "signal": bool(row["D"]), "uptrend": bool(row["D_uptrend"]),
            "close": round(float(row["close"]), 2), "chg_1d": round(float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1), 4),
            "rsi2": round(float(row["D_rsi2"]), 1), "dip": round(float(row["D_dip"]), 4),
            "past_year": per.get(s),
        })
    out = {"asof": asof, "n": len(tiles), "tiles": tiles, "method": __doc__}
    HEATMAP.write_text(json.dumps(out))
    return out


def annotate(status):
    """Attach the scan's verdict (pick / runner-up / rejection reason) to heat-map tiles."""
    if not HEATMAP.exists():
        return
    h = json.loads(HEATMAP.read_text())
    for t in h["tiles"]:
        t["status"] = status.get(t["symbol"])
    HEATMAP.write_text(json.dumps(h))
