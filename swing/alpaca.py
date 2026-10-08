"""Alpaca market data + paper trading (free plan: historical SIP bars, IEX quotes, news)."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

DATA = "https://data.alpaca.markets"
PAPER = "https://paper-api.alpaca.markets"

_session = requests.Session()
if os.environ.get("ALPACA_API_KEY_ID"):
    _session.headers.update({
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_API_SECRET_KEY", ""),
    })


def _get(url, params=None, tries=8):
    for i in range(tries):
        r = _session.get(url, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:   # free plan: ~200 requests/minute
            time.sleep(float(r.headers.get("Retry-After") or min(60, 2 ** (i + 1))))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def assets():
    return _get(f"{PAPER}/v2/assets", {"status": "active", "asset_class": "us_equity"})


def _bars_chunk(symbols, start, end):
    out, token = {}, None
    while True:
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start,
                  "limit": 10000, "adjustment": "all", "feed": "sip"}
        if end:
            params["end"] = end
        if token:
            params["page_token"] = token
        j = _get(f"{DATA}/v2/stocks/bars", params)
        for sym, rows in (j.get("bars") or {}).items():
            out.setdefault(sym, []).extend(rows)
        token = j.get("next_page_token")
        if not token:
            return out


def bars(symbols, start, end=None, chunk=200, workers=3, progress=None):
    """Daily adjusted bars -> {symbol: DataFrame[open, high, low, close, volume]} indexed by date."""
    if end is None:  # free plan cannot query the most recent 15 minutes of SIP data
        end = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
    chunks = [symbols[i:i + chunk] for i in range(0, len(symbols), chunk)]
    raw = {}
    with ThreadPoolExecutor(workers) as ex:
        for n, part in enumerate(ex.map(lambda c: _bars_chunk(c, start, end), chunks), 1):
            raw.update(part)
            if progress:
                progress(n / len(chunks))
    frames = {}
    for sym, rows in raw.items():
        df = pd.DataFrame(rows)
        df["t"] = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        frames[sym] = df.set_index("t")[["open", "high", "low", "close", "volume"]].astype(float)
    return frames


def news(symbol, days=30, limit=50):
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    j = _get(f"{DATA}/v1beta1/news", {"symbols": symbol, "start": start, "limit": limit, "sort": "desc"})
    return [{"time": n["created_at"][:16].replace("T", " "), "headline": n["headline"],
             "summary": (n.get("summary") or "")[:300], "source": n.get("source")}
            for n in j.get("news", [])]


def account():
    return _get(f"{PAPER}/v2/account")


def bracket_order(symbol, qty, take_profit, stop_loss):
    """Paper-account market buy with attached take-profit and stop-loss."""
    body = {"symbol": symbol, "qty": str(int(qty)), "side": "buy", "type": "market",
            "time_in_force": "gtc", "order_class": "bracket",
            "take_profit": {"limit_price": round(take_profit, 2)},
            "stop_loss": {"stop_price": round(stop_loss, 2)}}
    r = _session.post(f"{PAPER}/v2/orders", json=body, timeout=30)
    r.raise_for_status()
    return r.json()
