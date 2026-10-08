"""Buy-the-dip strategy report: how every qualifying dip played out over the past year (and since 2015).

Rules (Setup D): close > rising SMA200, SMA50 > SMA200, RSI(2) <= 10, close >= 5% below its 20-day high.
Entry: next open. Two exit styles are reported side by side:
  target  stop 2 ATR, take profit at the pre-dip 20-day high, exit after 20 days
  revert  stop 2 ATR, exit at the first close above the 5-day SMA, exit after 10 days
Benchmarks: SPY buy-and-hold, and random days in the same uptrend stocks with the same exits."""
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C, plan as P, universe
from .setups import features

REPORT = C.DATA_DIR / "dips_report.json"
COST = C.SPREAD + 2 * C.SLIPPAGE


def _revert_exit(o, l, c, sma5, i, stop, max_hold=10):
    entry = o[i + 1]
    last = min(len(c) - 1, i + max_hold)
    if last < i + max_hold:
        return None
    for j in range(i + 1, last + 1):
        if l[j] <= stop:
            px = min(o[j], stop) if j > i + 1 else stop
            return px, j, "loss"
        if c[j] > sma5[j]:
            return c[j], j, "win" if c[j] > entry else "loss"
    return c[last], last, "timeout"


def _trade(sym, df, f, i, style):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    if i + 1 >= len(df):
        return None
    entry = o[i + 1]
    if entry > f["D_max_entry"].iloc[i]:
        return None   # gapped up past the no-chase limit: skipped, same as the live scan
    p, why = P.build(f, i, "D", h, entry=entry)
    if p is None:
        return None
    if style == "target":
        r = P.simulate(o, h, l, c, i, p)
        if r is None:
            return None
        exit_px = entry + (r["r"] + p["cost_r"]) * p["R"]
        j, outcome = i + r["days"], r["outcome"]
    else:
        res = _revert_exit(o, l, c, f["sma5"].to_numpy(), i, p["stop"])
        if res is None:
            return None
        exit_px, j, outcome = res
    ret = exit_px / entry - 1 - COST
    return {"symbol": sym, "signal": str(f.index[i].date()), "entry_date": str(f.index[i + 1].date()),
            "exit_date": str(f.index[j].date()), "entry": round(float(entry), 2), "exit": round(float(exit_px), 2),
            "stop": round(p["stop"], 2), "target": round(p["t1"], 2), "ret": float(ret),
            "r": float((exit_px - entry) / p["R"] - p["cost_r"]), "days": int(j - i), "outcome": outcome,
            "quality": float(f["D_quality"].iloc[i])}


def _one_at_a_time(trades):
    """A trader holds one position per stock: skip signals while the previous dip trade is still open."""
    kept, busy_until = [], ""
    for t in trades:
        if t and t["entry_date"] > busy_until:
            kept.append(t)
            busy_until = t["exit_date"]
    return kept


def _summary(t):
    if not len(t):
        return {"trades": 0}
    wins, losses = t[t.ret > 0], t[t.ret <= 0]
    pf = wins.ret.sum() / -losses.ret.sum() if len(losses) and losses.ret.sum() < 0 else None
    return {"trades": int(len(t)), "win_rate": float((t.ret > 0).mean()), "avg_ret": float(t.ret.mean()),
            "median_ret": float(t.ret.median()), "avg_win": float(wins.ret.mean()) if len(wins) else None,
            "avg_loss": float(losses.ret.mean()) if len(losses) else None, "profit_factor": pf,
            "avg_r": float(t.r.mean()), "avg_days": float(t.days.mean()),
            "hit_target": float((t.outcome == "win").mean()), "stopped": float((t.outcome == "loss").mean()),
            "timed_out": float((t.outcome == "timeout").mean())}


def portfolio(trades, start, end, slots=10):
    """Equal-weight book with `slots` positions; each day the best-quality new signals fill free slots
    (one position per symbol). Returns the closed-trade equity curve (starts at 1.0)."""
    t = trades[(trades.entry_date >= start) & (trades.entry_date <= end)].sort_values(
        ["entry_date", "quality"], ascending=[True, False])
    free_at, held, taken = [""] * slots, {}, []
    for _, tr in t.iterrows():
        k = next((k for k in range(slots) if free_at[k] < tr.entry_date), None)
        if k is None or held.get(tr.symbol, "") >= tr.entry_date:
            continue
        free_at[k], held[tr.symbol] = tr.exit_date, tr.exit_date
        taken.append((tr.exit_date, k, tr.ret))
    value = [1.0 / slots] * slots
    curve, peak, mdd = [], 1.0, 0.0
    for d, k, ret in sorted(taken):
        value[k] *= 1 + ret
        eq = sum(value)
        peak, mdd = max(peak, eq), min(mdd, eq / peak - 1)
        curve.append({"date": d, "equity": round(eq, 4)})
    return {"slots": slots, "total_return": sum(value) - 1, "max_drawdown_closed": mdd,
            "trades_taken": len(taken), "curve": curve}


def run(progress=print, seed=11):
    syms = universe.symbols()
    progress(f"🛒 dip backtest: {len(syms)} symbols since 2015")
    bars = universe.history(syms, "2015-01-01", "bt")
    spy = bars["SPY"]
    end = spy.index[-1]
    year_start = end - timedelta(days=365)
    rng = np.random.default_rng(seed)
    rows = {"target": [], "revert": []}
    rand = {"target": [], "revert": []}
    for n, s in enumerate(syms, 1):
        df = bars.get(s)
        if df is None or len(df) < C.MIN_BARS:
            continue
        f = features(df, spy["close"])
        liquid = ((f["dollar_vol"] >= C.MIN_DOLLAR_VOL) & (f["close"] >= C.MIN_PRICE)).to_numpy()
        sig = np.where(f["D"].to_numpy() & liquid)[0]
        rnd = np.where(f["D_uptrend"].to_numpy() & liquid & (rng.random(len(f)) < 0.01))[0]
        for style in rows:
            rows[style] += _one_at_a_time([_trade(s, df, f, i, style) for i in sig])
            rand[style] += _one_at_a_time([_trade(s, df, f, i, style) for i in rnd])
        if n % 250 == 0:
            progress(f"dips {n}/{len(syms)}")

    ys, ye = str(year_start.date()), str(end.date())
    spy_1y = float(spy["close"].iloc[-1] / spy["close"][spy.index >= year_start].iloc[0] - 1)
    report = {"generated": str(pd.Timestamp.now()), "asof": ye, "window": [ys, ye],
              "rules": {"uptrend": "close > SMA200, SMA200 rising over 20 days, SMA50 > SMA200",
                        "dip": f"RSI(2) <= {C.D_RSI2_MAX} and close >= {C.D_MIN_DIP:.0%} below its 20-day high",
                        "entry": "next day's open", "stop": f"{C.D_STOP_ATR} x ATR(14) below entry (max 8% risk)",
                        "exits": {"target": "pre-dip 20-day high, else exit after 20 trading days",
                                  "revert": "first close above the 5-day SMA, else exit after 10 trading days"},
                        "costs": f"{COST:.2%} round trip"},
              "spy_1y": spy_1y, "styles": {}}
    for style in rows:
        t, r = pd.DataFrame(rows[style]), pd.DataFrame(rand[style])
        t1, r1 = t[t.entry_date >= ys], r[r.entry_date >= ys] if len(r) else r
        per = (t1.groupby("symbol").agg(trades=("ret", "size"), win_rate=("ret", lambda x: (x > 0).mean()),
                                        avg_ret=("ret", "mean"), total=("ret", lambda x: (1 + x).prod() - 1))
               .reset_index().to_dict("records")) if len(t1) else []
        report["styles"][style] = {
            "past_year": _summary(t1), "random_uptrend_past_year": _summary(r1),
            "since_2015": _summary(t), "random_uptrend_since_2015": _summary(r),
            "by_year": {int(y): _summary(g) for y, g in t.groupby(t.entry_date.str[:4])},
            "portfolio_past_year": portfolio(t, ys, ye),
            "per_ticker": per,
            "recent": t1.sort_values("entry_date", ascending=False).head(60).to_dict("records"),
        }
        progress(f"{style}: past year {report['styles'][style]['past_year']}")
    REPORT.write_text(json.dumps(report, default=str))
    return report
