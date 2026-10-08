"""Fundamentals, earnings, insiders, short interest and VIX via yfinance (unofficial Yahoo data)."""
import logging
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def vix_history(start="2014-01-01"):
    df = _safe(lambda: yf.download(["^VIX", "^VIX3M"], start=start, progress=False, auto_adjust=False))
    if df is None or df.empty:
        return None, None
    cl = df["Close"]
    cl.index = pd.to_datetime(cl.index).tz_localize(None).normalize()
    return cl.get("^VIX"), cl.get("^VIX3M")


def _row(stmt, names):
    for n in names:
        if stmt is not None and n in stmt.index:
            return stmt.loc[n].dropna().sort_index()
    return None


def fundamentals(symbol):
    """Everything the fundamental/positioning factors and LLM agents need, as plain JSON."""
    tk = yf.Ticker(symbol)
    info = _safe(lambda: tk.info, {}) or {}
    fd = {
        "symbol": symbol, "name": info.get("shortName"), "quote_type": info.get("quoteType"),
        "sector": info.get("sector"), "industry": info.get("industry"),
        "market_cap": info.get("marketCap"), "short_pct_float": info.get("shortPercentOfFloat"),
        "short_ratio_days": info.get("shortRatio"), "float_shares": info.get("floatShares"),
        "profit_margin": info.get("profitMargins"), "debt_to_equity": info.get("debtToEquity"),
        "forward_pe": info.get("forwardPE"), "summary": (info.get("longBusinessSummary") or "")[:600],
    }
    cal = _safe(lambda: tk.calendar, {}) or {}
    nxt = cal.get("Earnings Date") if isinstance(cal, dict) else None
    if nxt:
        d = nxt[0] if isinstance(nxt, list) else nxt
        fd["next_earnings"] = str(d)
        fd["days_to_earnings"] = (pd.Timestamp(d).date() - date.today()).days

    eh = _safe(lambda: tk.earnings_history)
    if eh is not None and len(eh):
        eh = eh.sort_index()
        last = eh.iloc[-1]
        fd["last_quarter_end"] = str(pd.Timestamp(eh.index[-1]).date())
        fd["surprise_pct"] = _num(last.get("surprisePercent"))
        fd["eps_actual"], fd["eps_estimate"] = _num(last.get("epsActual")), _num(last.get("epsEstimate"))
        fd["beats_last4"] = int((eh["surprisePercent"].tail(4) > 0).sum())

    q = _safe(lambda: tk.quarterly_income_stmt)
    rev = _row(q, ["Total Revenue", "Operating Revenue"])
    eps = _row(q, ["Diluted EPS", "Basic EPS"])
    if rev is not None and len(rev) >= 5:
        yoy = rev.iloc[-1] / rev.iloc[-5] - 1
        fd["rev_yoy"] = _num(yoy)
        if len(rev) >= 6:
            fd["rev_accel"] = _num(yoy - (rev.iloc[-2] / rev.iloc[-6] - 1))
    if eps is not None and len(eps) >= 5 and eps.iloc[-5] > 0:
        fd["eps_yoy"] = _num(eps.iloc[-1] / eps.iloc[-5] - 1)
    if rev is not None:
        fd["revenue_quarters"] = {str(k.date()): float(v) for k, v in rev.tail(6).items()}

    tr = _safe(lambda: tk.eps_trend)
    if tr is not None and "+1y" in tr.index and tr.loc["+1y", "30daysAgo"]:
        fd["revision_30d"] = _num(tr.loc["+1y", "current"] / tr.loc["+1y", "30daysAgo"] - 1)

    it = _safe(lambda: tk.insider_transactions)
    fd.update(_insiders(it))
    return fd


def _insiders(it, days=90):
    out = {"insider_buyers": 0, "insider_buy_value": 0.0, "insider_net_sell_value": 0.0, "insider_recent": []}
    if it is None or it.empty:
        return out
    dcol = "Start Date" if "Start Date" in it else "Transaction Start Date"
    it = it[pd.to_datetime(it[dcol]) >= pd.Timestamp(datetime.now()) - pd.Timedelta(days=days)]
    text = it["Text"].fillna("").str.lower()
    buys = it[text.str.contains("purchase")]
    sells = it[text.str.contains("sale")]
    out["insider_buyers"] = int(buys["Insider"].nunique())
    out["insider_buy_value"] = float(buys["Value"].fillna(0).sum())
    out["insider_net_sell_value"] = float(sells["Value"].fillna(0).sum() - out["insider_buy_value"])
    out["insider_recent"] = [
        {"who": r["Insider"], "role": r["Position"], "what": r["Text"], "value": _num(r["Value"]), "date": str(r[dcol])[:10]}
        for _, r in it.head(8).iterrows()]
    return out


def _num(x):
    try:
        x = float(x)
        return None if np.isnan(x) or np.isinf(x) else x
    except (TypeError, ValueError):
        return None
