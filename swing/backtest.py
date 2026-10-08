"""Walk the full history, detect setups, simulate trades, and store per-cell statistics.

Only price-based parts are backtestable (fundamental history and LLM judgments would leak the future).
Cells are keyed by setup x TechScore bucket (x regime), which is exactly what the live scan looks up."""
import json

import numpy as np
import pandas as pd

from . import config as C, plan as P, regime, scoring, universe, yahoo
from .setups import features

START = "2015-01-01"


def run(progress=print, random_frac=0.01, seed=7):
    syms = universe.symbols()
    progress(f"universe: {len(syms)} symbols; loading history since {START}")
    bars = universe.history(syms, START, "bt")
    spy = bars["SPY"]
    feats, raw = {}, {k: {} for k in scoring.RANKED}
    above50 = {}
    for n, s in enumerate(syms, 1):
        df = bars.get(s)
        if df is None or len(df) < C.MIN_BARS:
            continue
        f = features(df, spy["close"])
        liquid = (f["dollar_vol"] >= C.MIN_DOLLAR_VOL) & (f["close"] >= C.MIN_PRICE)
        for k in scoring.RANKED:
            raw[k][s] = f[k].where(liquid)
        above50[s] = (f["close"] > f["sma50"]).where(liquid & f["sma50"].notna())
        feats[s] = (df, f, liquid)
        if n % 250 == 0:
            progress(f"features {n}/{len(syms)}")

    progress("ranking cross-sections")
    ranks = {k: pd.DataFrame(v).rank(axis=1, pct=True) * 100 for k, v in raw.items()}
    breadth = pd.DataFrame(above50).mean(axis=1) * 100
    vix, vix3m = yahoo.vix_history()
    reg = regime.regime_frame(spy, bars["QQQ"], breadth, vix, vix3m)

    rng = np.random.default_rng(seed)
    trades = []
    for n, (s, (df, f, liquid)) in enumerate(feats.items(), 1):
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        rk = {k: ranks[k][s].reindex(f.index).to_numpy() for k in scoring.RANKED}
        sig = {"A": f["A"].to_numpy(), "C": f["C"].to_numpy(),
               "B": f["B_raw"].to_numpy() & (np.nan_to_num(rk["rs_raw"]) >= C.B_RS_MIN)}
        rand = liquid.to_numpy() & (rng.random(len(f)) < random_frac) & f["atr"].notna().to_numpy()
        liq = liquid.to_numpy()
        for setup, mask in list(sig.items()) + [("RANDOM", rand)]:
            for i in np.where(mask & liq)[0]:
                if i + 1 >= len(df):
                    continue
                row = f.iloc[i]
                use = "B" if setup == "RANDOM" else setup
                if setup != "RANDOM" and o[i + 1] > row[f"{use}_max_entry"]:
                    continue  # gapped past the no-chase limit: skipped, like live
                p, why = P.build(f, i, use, h, entry=o[i + 1])
                if p is None:
                    continue
                res = P.simulate(o, h, l, c, i, p)
                if res is None:
                    continue
                r_i = {k: rk[k][i] for k in scoring.RANKED}
                q = scoring.setup_quality(use, row, r_i["rs_raw"]) if setup != "RANDOM" else 50
                tech = scoring.weighted(scoring.tech_buckets(r_i, q), C.TECH_BUCKETS)
                d = f.index[i]
                trades.append({"symbol": s, "date": d, "setup": setup, "tech": tech,
                               "bucket": scoring.tech_bucket_label(tech),
                               "regime": reg["state"].get(d, "Neutral"), "rr": p["rr"],
                               "baseline": 1 / (1 + p["rr"]), **res})
        if n % 250 == 0:
            progress(f"simulated {n}/{len(feats)} symbols, {len(trades)} trades")

    t = pd.DataFrame(trades)
    t.to_parquet(C.DATA_DIR / "backtest_trades.parquet")
    stats = summarize(t)
    scoring.STATS_PATH.write_text(json.dumps(stats, indent=1, default=str))
    progress(report(stats))
    return stats


def _cell(g):
    wins = g[g.outcome == "win"]
    to = g[g.outcome == "timeout"]
    d = wins["days"]
    return {"n": int(len(g)), "wins": int(len(wins)), "timeouts": int(len(to)),
            "win_rate": float(len(wins) / len(g)), "baseline": float(g["baseline"].mean()),
            "avg_r": float(g["r"].mean()), "timeout_r": float(to["r"].mean()) if len(to) else 0.0,
            "days_median": float(d.median()) if len(d) else None,
            "days_iqr": [float(d.quantile(.25)), float(d.quantile(.75))] if len(d) else None}


def summarize(t):
    cells = {}
    for keys in (["setup"], ["setup", "bucket"], ["setup", "bucket", "regime"], ["setup", "regime"]):
        for k, g in t.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            cells["|".join(map(str, k))] = _cell(g)
    years = {}
    for (s, y), g in t.groupby([t.setup, pd.to_datetime(t.date).dt.year]):
        years.setdefault(s, {})[int(y)] = round(float(g["r"].mean()), 3)
    return {"generated": str(pd.Timestamp.now()), "start": START, "trades": int(len(t)),
            "cells": cells, "avg_r_by_year": years,
            "caveats": ["survivorship bias: universe = stocks listed today",
                        "fills at next open; stop assumed first when stop and target touch the same day",
                        "costs: spread + slippage per side included in R"]}


def report(stats):
    lines = ["", "setup   trades  win%   baseline  avgR    medianDays"]
    for s in ("A", "B", "C", "RANDOM"):
        c = stats["cells"].get(s)
        if c:
            lines.append(f"{s:7} {c['n']:6}  {c['win_rate']:.1%}  {c['baseline']:.1%}     "
                         f"{c['avg_r']:+.3f}  {c['days_median']}")
    for s in ("A", "B", "C"):
        for b in ("0-50", "50-60", "60-70", "70-80", "80-100"):
            c = stats["cells"].get(f"{s}|{b}")
            if c:
                lines.append(f"  {s} tech {b:6} n={c['n']:5} win {c['win_rate']:.1%} avgR {c['avg_r']:+.3f}")
    return "\n".join(lines)
