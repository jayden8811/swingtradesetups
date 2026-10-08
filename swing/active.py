"""Active mega-cap trading: buy short-term weakness, sell into the bounce.

Universe: point-in-time 30 most-traded stocks (60-day dollar volume, includes delisted names) = MSFT, NVDA, AAPL...
Candidates (published rules):
  rsi2      Connors & Alvarez (2008): close > SMA200 and RSI(2) < 10 -> buy; sell on first close > SMA5.
  ibs       Pagonidis / IBS research: IBS = (C-L)/(H-L) < 0.2 and close > SMA200 -> buy; sell on close > prior high.
  reversal  Short-term reversal (Lehmann 1990; de Groot et al. 2010): each Friday buy the 5 worst 5-day
            performers, sell 5 days later.
Controls: SPY and QQQ buy-and-hold, equal-weight the same 30 stocks (monthly), and random entries with the
same exits and slots (does the timing add anything?).
Execution: market-on-close at the signal close (default) or next open; 0.05% cost per side; at most 10 days held;
capital in equal slots; idle cash earns T-bills. In-sample 2016-2020, out-of-sample 2021-today."""
import json

import numpy as np
import pandas as pd

from . import config as C, momentum as M

REPORT = C.DATA_DIR / "active_report.json"
SIDE_COST = 0.0005
SPLIT = "2021-01-01"


def indicators(w, top=30):
    Cl, H, L, O = w["close"], w["high"], w["low"], w["open"]
    d = Cl.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return {
        "C": Cl, "O": O, "H": H, "univ": M.large_caps(w, top).reindex(columns=Cl.columns, fill_value=False).fillna(False).astype(bool),
        "sma200": Cl.rolling(200).mean(), "sma5": Cl.rolling(5).mean(),
        "rsi2": 100 - 100 / (1 + up / dn.replace(0, np.nan)),
        "ibs": ((Cl - L) / (H - L).replace(0, np.nan)),
        "ret5": Cl / Cl.shift(5) - 1,
    }


def simulate(ind, entry, exit_fn, priority, slots=10, max_hold=10, entry_days=None, at_open=False, start=M.START,
             rng=None, random_rate=None):
    """Slot-based daily simulation. entry: bool DataFrame (signal at close). exit_fn(sym, i, held_days) -> bool.
    Returns dict(equity Series, trades list, exposure)."""
    Cl, O = ind["C"], ind["O"]
    idx, cols = Cl.index, Cl.columns
    Cn, On = Cl.to_numpy(), O.to_numpy()
    E = (entry.reindex(columns=cols).fillna(False).astype(bool) & ind["univ"]).to_numpy() if entry is not None else None
    U = ind["univ"].to_numpy()
    P = priority.to_numpy() if priority is not None else None
    bil = (Cl["BIL"] / Cl["BIL"].shift() - 1).fillna(0).to_numpy()
    col = {s: j for j, s in enumerate(cols)}
    k0 = idx.get_indexer([pd.Timestamp(start)], method="bfill")[0]
    cash, pos, eq, trades, expo = 1.0, {}, [], [], []
    pending_buy, pending_sell = [], []
    for i in range(k0, len(idx)):
        cash *= 1 + bil[i]
        if at_open:   # orders decided at yesterday's close fill at today's open
            for s in pending_sell:
                p = pos.pop(s)
                px = On[i, col[s]]
                cash += p["units"] * px * (1 - SIDE_COST)
                trades.append({"symbol": s, "entry": p["date"], "exit": str(idx[i].date()), "ret": px / p["px"] * (1 - SIDE_COST) ** 2 - 1, "days": i - p["i"]})
            equity = cash + sum(p["units"] * On[i, col[s]] for s, p in pos.items())
            for s in pending_buy:
                if len(pos) >= slots or not np.isfinite(On[i, col[s]]):
                    break
                alloc = min(cash, equity / slots)
                pos[s] = {"units": alloc * (1 - SIDE_COST) / On[i, col[s]], "px": On[i, col[s]], "i": i, "date": str(idx[i].date())}
                cash -= alloc
            pending_buy, pending_sell = [], []
        # exits at the close
        for s in list(pos):
            p = pos[s]
            if s in pending_sell:
                continue
            held = i - p["i"]
            px = Cn[i, col[s]]
            if not np.isfinite(px):
                continue
            if held >= 1 and (held >= max_hold or exit_fn(s, i, held)):
                if at_open:
                    pending_sell.append(s)
                else:
                    pos.pop(s)
                    cash += p["units"] * px * (1 - SIDE_COST)
                    trades.append({"symbol": s, "entry": p["date"], "exit": str(idx[i].date()), "ret": px / p["px"] * (1 - SIDE_COST) ** 2 - 1, "days": held})
        equity = cash + sum(p["units"] * np.nan_to_num(Cn[i, col[s]], nan=p["px"]) for s, p in pos.items())
        # entries at the close
        free = slots - len(pos) + len(pending_sell) * 0
        if free > 0 and (entry_days is None or i in entry_days):
            if rng is not None:
                cand = [j for j in np.where(U[i] & (rng.random(len(cols)) < random_rate))[0]]
                rng.shuffle(cand)
            else:
                cand = np.where(E[i])[0]
                cand = sorted(cand, key=lambda j: P[i, j])
            for j in cand:
                s = cols[j]
                if s in pos or free <= 0 or not np.isfinite(Cn[i, j]):
                    continue
                if at_open:
                    pending_buy.append(s)
                else:
                    alloc = min(cash, equity / slots)
                    pos[s] = {"units": alloc * (1 - SIDE_COST) / Cn[i, j], "px": Cn[i, j], "i": i, "date": str(idx[i].date())}
                    cash -= alloc
                free -= 1
        equity = cash + sum(p["units"] * np.nan_to_num(Cn[i, col[s]], nan=p["px"]) for s, p in pos.items())
        expo.append(1 - cash / equity)
        eq.append((idx[i], equity))
    return {"equity": pd.Series(dict(eq)), "trades": trades, "exposure": float(np.mean(expo)),
            "open": {s: {"since": p["date"], "entry": round(float(p["px"]), 2)} for s, p in pos.items()}}


def strategies(ind):
    Cl, H = ind["C"], ind["H"]
    sma5, ph = ind["sma5"].to_numpy(), H.shift().to_numpy()
    Cn = Cl.to_numpy()
    col = {s: j for j, s in enumerate(Cl.columns)}
    up = Cl > ind["sma200"]
    fridays = {i for i in range(len(Cl.index)) if i + 1 == len(Cl.index) or Cl.index[i + 1].weekday() < Cl.index[i].weekday()}
    rsi_exit = lambda s, i, h: Cn[i, col[s]] > sma5[i, col[s]]
    ibs_exit = lambda s, i, h: Cn[i, col[s]] > ph[i, col[s]]
    return {
        "rsi2": dict(entry=up & (ind["rsi2"] < 10), exit_fn=rsi_exit, priority=ind["rsi2"]),
        "ibs": dict(entry=up & (ind["ibs"] < 0.2), exit_fn=ibs_exit, priority=ind["ibs"]),
        "reversal": dict(entry=ind["ret5"].notna(), exit_fn=lambda s, i, h: h >= 5, priority=ind["ret5"],
                         slots=5, entry_days=fridays),
        # robustness neighbours
        "rsi2_5": dict(entry=up & (ind["rsi2"] < 5), exit_fn=rsi_exit, priority=ind["rsi2"]),
        "rsi2_15": dict(entry=up & (ind["rsi2"] < 15), exit_fn=rsi_exit, priority=ind["rsi2"]),
        "rsi2_nofilter": dict(entry=ind["rsi2"] < 10, exit_fn=rsi_exit, priority=ind["rsi2"]),
        "rsi2_5slots": dict(entry=up & (ind["rsi2"] < 10), exit_fn=rsi_exit, priority=ind["rsi2"], slots=5),
        "ibs_15": dict(entry=up & (ind["ibs"] < 0.15), exit_fn=ibs_exit, priority=ind["ibs"]),
        "ibs_25": dict(entry=up & (ind["ibs"] < 0.25), exit_fn=ibs_exit, priority=ind["ibs"]),
        "reversal_3": dict(entry=ind["ret5"].notna(), exit_fn=lambda s, i, h: h >= 5, priority=ind["ret5"], slots=3, entry_days=fridays),
        "reversal_10": dict(entry=ind["ret5"].notna(), exit_fn=lambda s, i, h: h >= 5, priority=ind["ret5"], slots=10, entry_days=fridays),
    }


def _period(eq, a, b):
    e = eq[(eq.index >= a) & (eq.index < b)]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    r = e.pct_change().dropna()
    return {"cagr": float((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "max_dd": float((e / e.cummax() - 1).min())}


def run(progress=print, top=30):
    progress("📚 loading point-in-time data")
    w = M.load()
    ind = indicators(w, top)
    k0 = ind["C"].index.get_indexer([pd.Timestamp(M.START)], method="bfill")[0] - 1
    idx = ind["C"].index
    res = {}
    progress("⚙️ benchmarks")
    for b in ("SPY", "QQQ"):
        res[b.lower()] = {"equity": M.run_portfolio(w, [k0], lambda i, cur, b=b: {b: 1.0})["equity"], "trades": [], "exposure": 1.0}
    ew = M.run_portfolio(w, M.month_ends(idx, k0), lambda i, cur: {s: 1 / top for s in ind["univ"].columns[ind["univ"].iloc[i].to_numpy()]})
    res["equal_weight"] = {"equity": ew["equity"], "trades": [], "exposure": 1.0}
    for name, kw in strategies(ind).items():
        progress(f"⚙️ {name}")
        res[name] = simulate(ind, **kw)
    res["rsi2_next_open"] = simulate(ind, **strategies(ind)["rsi2"], at_open=True)
    res["ibs_next_open"] = simulate(ind, **strategies(ind)["ibs"], at_open=True)
    global SIDE_COST
    base_cost, SIDE_COST = SIDE_COST, 0.0015
    res["ibs_cost_15bp"] = simulate(ind, **strategies(ind)["ibs"])
    SIDE_COST = base_cost
    # random-timing control: same exits/slots as rsi2, entries at the same average rate
    sig_rate = float((strategies(ind)["rsi2"]["entry"] & ind["univ"]).to_numpy()[k0:].mean() / ind["univ"].to_numpy()[k0:].mean())
    rnd = [simulate(ind, None, strategies(ind)["rsi2"]["exit_fn"], None, rng=np.random.default_rng(sd), random_rate=sig_rate)
           for sd in range(5)]
    res["random_same_exits"] = {"equity": pd.concat([r["equity"] for r in rnd], axis=1).mean(axis=1),
                                "trades": sum((r["trades"] for r in rnd), []), "exposure": float(np.mean([r["exposure"] for r in rnd]))}

    spy, ewq = res["spy"]["equity"], res["equal_weight"]["equity"]
    end = str(idx[-1].date())
    out = {"generated": str(pd.Timestamp.now()), "start": M.START, "split": SPLIT, "end": end, "doc": __doc__,
           "universe_top": top, "strategies": {}}
    for name, r in res.items():
        eq = r["equity"]
        t = pd.DataFrame(r["trades"])
        yr = eq.resample("YE").last()
        yr = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]), yr]).pct_change().dropna()
        out["strategies"][name] = {
            "full": _period(eq, M.START, "2100"), "in_sample": _period(eq, M.START, SPLIT), "out_sample": _period(eq, SPLIT, "2100"),
            "years": {int(k.year): float(v) for k, v in yr.items()}, "exposure": r["exposure"],
            "trades_per_year": len(t) / ((eq.index[-1] - eq.index[0]).days / 365.25) if len(t) else 0,
            "win_rate": float((t.ret > 0).mean()) if len(t) else None, "avg_trade": float(t.ret.mean()) if len(t) else None,
            "avg_days": float(t.days.mean()) if len(t) else None,
            "curve": [{"d": str(d.date()), "v": round(float(v), 4)} for d, v in eq.resample("W").last().items()],
            "recent": t.tail(30).to_dict("records") if len(t) else [], "open": r.get("open", {}),
        }
    S = out["strategies"]
    neighbours = {"rsi2": ["rsi2_5", "rsi2_15", "rsi2_next_open"], "ibs": ["ibs_15", "ibs_25", "ibs_next_open"], "reversal": ["reversal_3", "reversal_10"]}
    for k, nb in neighbours.items():
        o, os_, ew_o, spy_o = S[k], S[k]["out_sample"], S["equal_weight"]["out_sample"], S["spy"]["out_sample"]
        checks = {"oos_cagr_beats_spy": os_["cagr"] > spy_o["cagr"], "oos_sharpe_beats_spy": os_["sharpe"] > spy_o["sharpe"],
                  "oos_beats_holding_same_stocks": os_["cagr"] > ew_o["cagr"],
                  "beats_random_timing": o["full"]["cagr"] > S["random_same_exits"]["full"]["cagr"] if k.startswith("rsi2") else True,
                  "in_sample_also_beats_spy": o["in_sample"]["cagr"] > S["spy"]["in_sample"]["cagr"],
                  "neighbours_beat_spy_oos": all(S[n]["out_sample"]["cagr"] > spy_o["cagr"] for n in nb)}
        o["checks"], o["passes"] = checks, all(checks.values())
    passing = [k for k in neighbours if S[k]["passes"]]
    out["applied"] = max(passing, key=lambda k: S[k]["out_sample"]["sharpe"]) if passing else None
    out["candidate"] = "ibs"
    out["live"] = live_ibs(ind, res["ibs"])
    REPORT.write_text(json.dumps(out, default=str))
    return out


def live_ibs(ind, sim):
    """Today's IBS signals in the mega-cap universe and the backtest book's open positions with their exit levels."""
    i = len(ind["C"]) - 1
    row = lambda k: ind[k].iloc[i]
    u = ind["univ"].iloc[i]
    names = list(u[u].index)
    sigs = []
    for s in names:
        c, ibs, sma = float(row("C")[s]), float(row("ibs")[s]), float(row("sma200")[s])
        sigs.append({"symbol": s, "close": round(c, 2), "ibs": round(ibs, 3), "above_200d": bool(c > sma),
                     "buy": bool(ibs < 0.2 and c > sma)})
    sigs.sort(key=lambda x: x["ibs"])
    held = []
    for s, p in sim["open"].items():
        c = float(row("C")[s])
        held.append({"symbol": s, "since": p["since"], "entry": p["entry"], "close": round(c, 2),
                     "pnl": round(c / p["entry"] - 1, 4), "sell_if_close_above": round(float(ind["H"][s].iloc[i]), 2)})
    return {"asof": str(ind["C"].index[i].date()), "signals": sigs, "positions": held, "slots": 10}
