"""Stage 0: tradable universe and cached daily bars."""
import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import alpaca, config as C

NY = ZoneInfo("America/New_York")
BENCH = ["SPY", "QQQ"] + sorted(set(C.SECTOR_ETF.values()))


def _clean_name(a):
    sym, name = a["symbol"], (a.get("name") or "").lower()
    if not a.get("tradable") or a.get("exchange") not in C.EXCHANGES:
        return False
    if any(ch in sym for ch in "./-") or len(sym) > 5:
        return False
    return not any(w in name for w in C.NAME_EXCLUDE)


def symbols(progress=None):
    """Liquid common-stock-like symbols, cached per day."""
    path = C.CACHE_DIR / f"universe_{date.today()}.json"
    if path.exists():
        return json.loads(path.read_text())
    syms = sorted(a["symbol"] for a in alpaca.assets() if _clean_name(a))
    start = (date.today() - timedelta(days=45)).isoformat()
    recent = alpaca.bars(syms, start, chunk=400, progress=progress)
    keep = []
    for s, df in recent.items():
        if len(df) >= 15:
            dv = (df["close"] * df["volume"]).tail(20).mean()
            if df["close"].iloc[-1] >= C.MIN_PRICE and dv >= 0.75 * C.MIN_DOLLAR_VOL:
                keep.append(s)
    keep.sort()
    path.write_text(json.dumps(keep))
    return keep


def completed_sessions(df):
    """Drop today's partial bar while the market is still open."""
    now = datetime.now(NY)
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 20):
        return df.iloc[:-1]
    return df


def history(syms, start, tag, progress=None):
    """Bars since `start` for syms (+ benchmarks), cached as one parquet per tag/day."""
    now = datetime.now(NY)
    session = "eod" if (now.hour, now.minute) >= (16, 20) else "intraday"   # never reuse a partial-day cache
    path = C.CACHE_DIR / f"bars_{tag}_{now.date()}_{session}.parquet"
    if path.exists():
        long = pd.read_parquet(path)
    else:
        frames = alpaca.bars(sorted(set(syms) | set(BENCH)), start, progress=progress)
        long = pd.concat({s: d for s, d in frames.items()}, names=["symbol", "t"]).reset_index()
        long.to_parquet(path)
    out = {}
    for s, d in long.groupby("symbol", sort=False):
        out[s] = completed_sessions(d.drop(columns="symbol").set_index("t").sort_index())
    return out
