"""Published, heavily researched strategies backtested side by side against SPY.

Candidates (rules as published, not tuned here):
  clenow    Andreas Clenow, "Stocks on the Move" (2015): rank large caps by 90-day exponential-regression
            slope x R^2; buy only when SPY > 200-day SMA; stock must be above its 100-day SMA with no
            >15% one-day move in 90 days; ATR risk-parity sizing (10 bps/day per position); weekly trades,
            position sizes reset every 2nd week; sell when out of the top 20%, below the 100-day, or gappy.
  jt_mom    Jegadeesh & Titman (1993) 12-1 month momentum: monthly, top 50 of the large-cap universe, equal weight.
  gem       Antonacci (2014) Global Equities Momentum: SPY vs EFA vs bonds on 12-month returns, monthly.
  faber     Faber (2007) 10-month SMA timing: SPY when above its 10-month average, else T-bills, monthly.
Benchmarks: SPY buy-and-hold, and an equal-weight portfolio of the same large-cap universe
(it carries the same survivorship bias, so beating it is the fair test for the stock strategies).

Mechanics: signals at the close, trades at the next open, 0.10% cost per unit of turnover,
idle cash earns T-bills (BIL). Large-cap universe = point-in-time top 500 by 60-day dollar volume, drawn from
every listed and delisted ticker Alpaca has (reduces, but cannot fully remove, survivorship bias).

Pre-registered rule for "apply it" (written before the results were seen):
  CAGR > SPY, Sharpe > SPY, CAGR > equal-weight universe, wins >= 60% of rolling 36-month windows
  vs SPY, and the neighbouring parameter settings must also beat SPY."""
import json
from datetime import date

import numpy as np
import pandas as pd

from . import alpaca, config as C, universe

REPORT = C.DATA_DIR / "momentum_report.json"
COST = 0.0010
ETFS = ["SPY", "EFA", "BIL", "AGG"]
START = "2016-01-04"


# ---------------------------------------------------------------- data

def load():
    syms = universe.all_symbols_pit()
    bars = universe.history(syms, "2015-01-01", "pit")
    extra = [e for e in ETFS if e not in bars]
    if extra:
        bars.update(alpaca.bars(extra, "2015-01-01"))
    wide = {k: pd.DataFrame({s: d[k] for s, d in bars.items()}).sort_index() for k in ("open", "high", "low", "close", "volume")}
    return wide


def regression_score(close, n=90):
    """Clenow: annualized exponential-regression slope x R^2 (vectorized over all columns)."""
    y = np.log(close)
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    my, mt = y.rolling(n).mean(), t.rolling(n).mean()
    cov = y.mul(t, axis=0).rolling(n).mean() - my.mul(mt, axis=0)
    vy = (y * y).rolling(n).mean() - my * my
    vt = (n * n - 1) / 12
    b = cov / vt
    r2 = (cov * cov) / (vt * vy)
    return (np.exp(b * 252) - 1) * r2


def atr(w, n=20):
    pc = w["close"].shift()
    tr = np.maximum(w["high"] - w["low"], np.maximum((w["high"] - pc).abs(), (w["low"] - pc).abs()))
    return tr.rolling(n).mean()


def large_caps(w, k=500):
    dv = (w["close"] * w["volume"]).rolling(60).mean()
    ok = w["close"].notna().rolling(252).count() >= 252
    funds = set(ETFS) | set(universe.BENCH)        # benchmark/sector ETFs are not stocks
    dv = dv.where(ok & (w["close"] >= 5)).drop(columns=[e for e in funds if e in dv], errors="ignore")
    return dv.rank(axis=1, ascending=False) <= k


# ---------------------------------------------------------------- engine

def run_portfolio(w, signal_days, decide, start=START):
    """decide(i, current_weights) -> target weights (sum <= 1), using data up to close of day i.
    Executes at the open of day i+1. Returns daily equity, weights history and turnover."""
    O = w["open"]
    idx = O.index
    rets = (O.shift(-1) / O - 1)
    cash = rets["BIL"].fillna(0).to_numpy()
    R = rets.to_numpy()
    col = {s: j for j, s in enumerate(O.columns)}
    sig = set(signal_days)
    k0 = idx.get_indexer([pd.Timestamp(start)], method="bfill")[0]
    V, cur, eq, hist, turn = 1.0, {}, [], [], 0.0
    for k in range(k0, len(idx) - 1):
        if (k - 1) in sig:
            tgt = {s: x for s, x in decide(k - 1, dict(cur)).items() if x > 1e-6}
            tov = sum(abs(tgt.get(s, 0) - cur.get(s, 0)) for s in set(tgt) | set(cur))
            V *= 1 - COST * tov
            turn += tov
            cur = tgt
            hist.append((idx[k], dict(cur)))
        invested = sum(cur.values())
        r = sum(x * np.nan_to_num(R[k, col[s]]) for s, x in cur.items()) + (1 - invested) * cash[k]
        V *= 1 + r
        if r > -1:   # drift weights with prices
            cur = {s: x * (1 + np.nan_to_num(R[k, col[s]])) / (1 + r) for s, x in cur.items()}
        eq.append((idx[k + 1], V))
    s = pd.Series(dict(eq))
    years = (s.index[-1] - s.index[0]).days / 365.25
    return {"equity": s, "history": hist, "turnover_per_year": turn / years, "final": cur}


def month_ends(idx, k0):
    m = pd.Series(idx.month, index=idx)
    return [i for i in range(k0, len(idx)) if i + 1 == len(idx) or m.iloc[i + 1] != m.iloc[i]]


def weekly(idx, k0, weekday=2):
    return [i for i in range(k0, len(idx)) if idx[i].weekday() == weekday]


# ---------------------------------------------------------------- strategies

def make_clenow(w, lookback=90, top_frac=0.20, market_filter=True):
    C_, univ = w["close"], large_caps(w)
    score = regression_score(C_, lookback)
    sma100, spy200 = C_.rolling(100).mean(), C_["SPY"].rolling(200).mean()
    gappy = (C_ / C_.shift() - 1).abs().rolling(90).max() > 0.15
    size = (0.001 * C_ / atr(w)).clip(upper=0.25)
    state = {"n": 0}

    def decide(i, cur):
        u = univ.iloc[i]
        sc = score.iloc[i].where(u).dropna().sort_values(ascending=False)
        top = set(sc.index[:max(1, int(len(sc) * top_frac))])
        ok = lambda s: (s in top and C_[s].iloc[i] > sma100[s].iloc[i] and not gappy[s].iloc[i])
        keep = {s: x for s, x in cur.items() if ok(s)}
        state["n"] += 1
        if state["n"] % 2 == 0:   # every second week: reset position sizes to risk parity
            keep = {s: float(size[s].iloc[i]) for s in keep}
        total = sum(keep.values())
        if total > 1:
            keep = {s: x / total for s, x in keep.items()}
            total = 1
        if not market_filter or C_["SPY"].iloc[i] > spy200.iloc[i]:
            for s in sc.index:
                if total >= 0.999:
                    break
                if s in keep or not ok(s):
                    continue
                x = min(float(size[s].iloc[i]), 1 - total)
                if x > 0.002:
                    keep[s], total = x, total + x
        return keep
    return decide


def make_jt(w, top=50, market_filter=False):
    C_, univ = w["close"], large_caps(w)
    mom = C_.shift(21) / C_.shift(252) - 1
    spy200 = C_["SPY"].rolling(200).mean()

    def decide(i, cur):
        if market_filter and C_["SPY"].iloc[i] < spy200.iloc[i]:
            return {}
        sc = mom.iloc[i].where(univ.iloc[i]).dropna().sort_values(ascending=False)
        return {s: 1 / top for s in sc.index[:top]}
    return decide


def make_gem(w):
    C_ = w["close"]
    r12 = C_[["SPY", "EFA", "BIL", "AGG"]] / C_[["SPY", "EFA", "BIL", "AGG"]].shift(252) - 1

    def decide(i, cur):
        r = r12.iloc[i]
        if r["SPY"] > r["BIL"]:
            return {"SPY" if r["SPY"] >= r["EFA"] else "EFA": 1.0}
        return {"AGG": 1.0}
    return decide


def make_faber(w):
    C_ = w["close"]
    sma = C_["SPY"].rolling(210).mean()
    return lambda i, cur: {"SPY": 1.0} if C_["SPY"].iloc[i] > sma.iloc[i] else {}


def make_equal_weight(w):
    univ = large_caps(w)
    def decide(i, cur):
        names = univ.columns[univ.iloc[i].to_numpy()]
        return {s: 1 / len(names) for s in names}
    return decide


# ---------------------------------------------------------------- metrics

def metrics(eq, spy_eq, cash_eq):
    r, b, c = eq.pct_change().dropna(), spy_eq.pct_change().dropna(), cash_eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    ex = r - c.reindex(r.index).fillna(0)
    dd = (eq / eq.cummax() - 1).min()
    m, mb = eq.resample("ME").last(), spy_eq.resample("ME").last()
    roll = lambda n: float(((m / m.shift(n)) > (mb / mb.shift(n))).iloc[n:].mean())
    yr = eq.resample("YE").last().pct_change()
    yr.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    ybr = spy_eq.resample("YE").last().pct_change()
    ybr.iloc[0] = spy_eq.resample("YE").last().iloc[0] / spy_eq.iloc[0] - 1
    bb = b.reindex(r.index).fillna(0)
    beta = float(np.cov(r, bb)[0, 1] / bb.var())
    exb = bb - c.reindex(r.index).fillna(0)
    alpha = float((ex.mean() - beta * exb.mean()) * 252)     # Jensen's alpha vs SPY, annualized
    return {"cagr": float(cagr), "vol": float(r.std() * np.sqrt(252)),
            "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)), "max_dd": float(dd),
            "beta": beta, "alpha": alpha, "win_12m": roll(12), "win_36m": roll(36),
            "years": {int(k.year): float(v) for k, v in yr.items()},
            "years_beat_spy": int(sum(yr.to_numpy() > ybr.to_numpy())), "n_years": int(len(yr))}


def run(progress=print):
    progress("📚 loading 10 years of daily bars")
    w = load()
    idx = w["open"].index
    k0 = idx.get_indexer([pd.Timestamp(START)], method="bfill")[0] - 1
    me, wk = month_ends(idx, k0), weekly(idx, k0)
    hold = lambda name: (lambda i, cur: {name: 1.0})
    runs = {
        "spy": (hold("SPY"), [k0]),
        "equal_weight": (make_equal_weight(w), me),
        "clenow": (make_clenow(w), wk),
        "jt_mom": (make_jt(w), me),
        "gem": (make_gem(w), me),
        "faber": (make_faber(w), me),
        # robustness neighbours
        "clenow_60d": (make_clenow(w, lookback=60), wk),
        "clenow_125d": (make_clenow(w, lookback=125), wk),
        "clenow_top10": (make_clenow(w, top_frac=0.10), wk),
        "clenow_top30": (make_clenow(w, top_frac=0.30), wk),
        "clenow_nofilter": (make_clenow(w, market_filter=False), wk),
        "jt_top25": (make_jt(w, top=25), me),
        "jt_top100": (make_jt(w, top=100), me),
        "jt_filter": (make_jt(w, market_filter=True), me),
    }
    res = {}
    for name, (fn, days) in runs.items():
        progress(f"⚙️ {name}")
        res[name] = run_portfolio(w, days, fn)
    cash = run_portfolio(w, [k0], hold("BIL"))["equity"]
    spy = res["spy"]["equity"]
    out = {"generated": str(pd.Timestamp.now()), "start": str(spy.index[0].date()), "end": str(spy.index[-1].date()),
           "doc": __doc__, "strategies": {}}
    for name, r in res.items():
        m = metrics(r["equity"], spy, cash)
        m["turnover_per_year"] = r["turnover_per_year"]
        m["avg_positions"] = float(np.mean([len(h) for _, h in r["history"]])) if r["history"] else 0
        m["curve"] = [{"d": str(d.date()), "v": round(float(v), 4)} for d, v in r["equity"].resample("W").last().items()]
        out["strategies"][name] = m
    s, spym, ew = out["strategies"], out["strategies"]["spy"], out["strategies"]["equal_weight"]
    for base, neighbours in (("clenow", ["clenow_60d", "clenow_125d", "clenow_top10", "clenow_top30"]),
                             ("jt_mom", ["jt_top25", "jt_top100"]), ("gem", []), ("faber", [])):
        m = s[base]
        checks = {"cagr_beats_spy": m["cagr"] > spym["cagr"], "sharpe_beats_spy": m["sharpe"] > spym["sharpe"],
                  "beats_equal_weight": base in ("gem", "faber") or m["cagr"] > ew["cagr"],
                  "wins_60pct_rolling_36m": m["win_36m"] >= 0.60,
                  "neighbours_beat_spy": all(s[n]["cagr"] > spym["cagr"] for n in neighbours)}
        m["checks"], m["passes"] = checks, all(checks.values())
    passing = [n for n in ("clenow", "jt_mom", "gem", "faber") if s[n]["passes"]]
    out["applied"] = max(passing, key=lambda n: s[n]["sharpe"]) if passing else None
    # live portfolio = final state of the applied (or best-Sharpe) strategy
    live = out["applied"] or max(("clenow", "jt_mom", "gem", "faber"), key=lambda n: s[n]["sharpe"])
    last = res[live]["history"][-1] if res[live]["history"] else (None, {})
    out["live"] = {"strategy": live, "applied": out["applied"] is not None,
                   "last_rebalance": str(last[0].date()) if last[0] is not None else None,
                   "holdings": sorted(({"symbol": k, "weight": round(v, 4)} for k, v in res[live]["final"].items()),
                                      key=lambda x: -x["weight"]),
                   "spy_above_200d": bool(w["close"]["SPY"].iloc[-1] > w["close"]["SPY"].rolling(200).mean().iloc[-1])}
    REPORT.write_text(json.dumps(out, default=str))
    return out
