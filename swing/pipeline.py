"""Daily scan: Stage 0-5 rules, then LLM layers 1-3, then the setup card(s)."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import alpaca, config as C, llm, plan as P, regime as R, scoring as S, universe, yahoo
from .setups import features

SETUP_NAMES = {"A": "Base Breakout", "B": "Pullback in Uptrend", "C": "Post-Earnings Drift"}
LATEST = C.DATA_DIR / "latest.json"
PICKS_LOG = C.DATA_DIR / "picks.jsonl"


def _r(x, n=2):
    try:
        x = float(x)
        return None if not np.isfinite(x) else round(x, n)
    except (TypeError, ValueError):
        return None


def scan(progress=print):
    stats = S.load_stats()
    progress("🌎 building universe")
    syms = universe.symbols()
    progress(f"📥 loading ~15 months of bars for {len(syms)} symbols")
    start = (date.today() - timedelta(days=460)).isoformat()
    bars = universe.history(syms, start, "live")
    spy = bars["SPY"]

    progress("🧮 computing setups and factors")
    feats, last_vals, above = {}, {}, {}
    for s in syms:
        df = bars.get(s)
        if df is None or len(df) < C.MIN_BARS or df.index[-1] != spy.index[-1]:
            continue
        f = features(df, spy["close"])
        row = f.iloc[-1]
        if row["dollar_vol"] < C.MIN_DOLLAR_VOL or row["close"] < C.MIN_PRICE:
            continue
        feats[s] = (df, f)
        last_vals[s] = row[S.RANKED]
        above[s] = row["close"] > row["sma50"]
    ranks = S.cross_ranks(pd.DataFrame(last_vals).T.astype(float))
    breadth = pd.Series({spy.index[-1]: 100 * np.mean(list(above.values()))})
    vix, vix3m = yahoo.vix_history(start)
    reg_df = R.regime_frame(spy, bars["QQQ"], breadth, vix, vix3m)
    reg_row = reg_df.iloc[-1]
    regime = {"state": reg_row["state"], "score": int(reg_row["score"]), "breadth_pct": _r(reg_row["breadth_pct"], 1),
              "dist_days": int(reg_row["dist_days"]), "vix": _r(vix.iloc[-1]) if vix is not None else None,
              "components": {k: int(reg_row[k]) for k in ("spy_above_200", "spy_50_over_200", "qqq_above_50",
                                                           "breadth", "vix", "vix_term", "distribution_ok")}}
    threshold, max_picks = C.REGIME[regime["state"]]
    asof = str(spy.index[-1].date())

    # sector ETF 63-day return ranks (top 3 -> 100, bottom 3 -> 0)
    etf_ret = {e: bars[e]["close"].iloc[-1] / bars[e]["close"].iloc[-64] - 1 for e in C.SECTOR_ETF.values() if e in bars}
    order = sorted(etf_ret, key=etf_ret.get, reverse=True)
    sector_rs = {e: float(np.interp(i, [2, len(order) - 3], [100, 0])) for i, e in enumerate(order)}

    progress("🔎 detecting setups")
    cands, rejected = [], []
    for s, (df, f) in feats.items():
        row, rk = f.iloc[-1], ranks.loc[s].to_dict()
        hits = [k for k in ("A", "C") if row[k]] + (["B"] if row["B_raw"] and rk["rs_raw"] >= C.B_RS_MIN else [])
        for setup in hits:
            p, why = P.build(f, len(f) - 1, setup, df["high"].to_numpy())
            if p is None:
                rejected.append({"symbol": s, "setup": setup, "reason": why})
                continue
            q = S.setup_quality(setup, row, rk["rs_raw"])
            tb = S.tech_buckets(rk, q)
            tech = S.weighted(tb, C.TECH_BUCKETS)
            prob = S.probability(stats, setup, S.tech_bucket_label(tech), regime["state"], p["rr"])
            ev = S.expected_value(prob, p)
            cands.append({"symbol": s, "setup": setup, "plan": p, "prob": prob, "ev_r": ev,
                          "buckets": tb, "tech": tech, "ranks": rk, "row": row, "df": df})
    progress(f"✅ {len(cands)} setups passed trade-plan rules ({len(rejected)} rejected)")

    # Stage 0 (late) + fundamentals for survivors
    def enrich(c):
        c["fund"] = yahoo.fundamentals(c["symbol"])
        return c
    with ThreadPoolExecutor(4) as ex:
        cands = list(ex.map(enrich, cands))
    kept = []
    for c in cands:
        fd, why = c["fund"], None
        if fd.get("quote_type") not in (None, "EQUITY"):
            why = f"not common stock ({fd.get('quote_type')})"
        elif fd.get("market_cap") and fd["market_cap"] < C.MIN_MARKET_CAP:
            why = f"market cap ${fd['market_cap'] / 1e9:.2f}B < $1B"
        elif c["setup"] != "C" and fd.get("days_to_earnings") is not None and 0 <= fd["days_to_earnings"] <= C.EARNINGS_BLACKOUT_DAYS:
            why = f"earnings in {fd['days_to_earnings']} days"
        elif c["setup"] == "C" and not _earnings_gap(c):
            why = "gap not confirmed as earnings"
        elif c["ev_r"] < C.MIN_EV_R:
            why = f"EV {c['ev_r']:+.2f}R < {C.MIN_EV_R}R"
        if why:
            rejected.append({"symbol": c["symbol"], "setup": c["setup"], "reason": why})
        else:
            kept.append(c)

    for c in kept:
        fs = S.fundamental_scores(c["fund"])
        etf = C.SECTOR_ETF.get(c["fund"].get("sector"))
        srs = sector_rs.get(etf, 50)
        b = dict(c["buckets"], rs=(c["ranks"]["rs_raw"] + srs) / 2, fundamental=fs["bucket"],
                 positioning=S.positioning_scores(c["fund"], c["ranks"]["rs_raw"])["bucket"])
        c["buckets"], c["fund_scores"], c["sector_etf"], c["sector_rs"] = b, fs, etf, srs
        c["quant"] = S.weighted(b, C.WEIGHTS)
    if kept:
        ev_rank = pd.Series([c["ev_r"] for c in kept]).rank(pct=True) * 100
        for c, er in zip(kept, ev_rank):
            c["rank_score"] = 0.6 * c["quant"] + 0.4 * er
    kept.sort(key=lambda c: c["rank_score"], reverse=True)
    top = kept[:C.LLM_TOP_N]
    for c in top:
        c["digest"] = digest(c, regime)

    judge = None
    llm_on = llm.enabled() and bool(top)
    for c in top:
        c.update(l1_adj=0.0, l2_penalty=0.0, l3_adj=0.0, veto=None, l1={}, l2=None, l3_pick=True, card=None)
    if llm_on:
        progress(f"🧠 L1: 4 specialists x {len(top)} candidates")
        for c in top:
            c["digest"]["news"] = alpaca.news(c["symbol"])
            c["chart"] = llm.chart_png(c["df"], c["plan"])
        llm.layer1(top, progress)
        l2 = sorted(top, key=lambda c: c["quant"] + c["l1_adj"], reverse=True)[:C.L2_TOP_N]
        progress(f"⚔️ L2: bull vs bear on {len(l2)}")
        llm.layer2(l2, progress)
        for c in top:
            c["score_pre_l3"] = c["quant"] + c["l1_adj"] + c["l2_penalty"]
        l3 = [c for c in sorted(top, key=lambda c: c["score_pre_l3"], reverse=True) if not c["veto"]][:C.L3_TOP_N]
        if l3:
            progress(f"⚖️ L3: chief judge x{C.L3_RUNS} on {len(l3)}")
            judge = llm.layer3(l3, regime, max_picks, progress)
        l3_syms = {c["symbol"] for c in l3}
        for c in top:
            if c["symbol"] not in l3_syms:
                c["l3_pick"] = False

    for c in top:
        c["final"] = c["quant"] + c["l1_adj"] + c["l2_penalty"] + c["l3_adj"]
    picks, sectors = [], set()
    for c in sorted(top, key=lambda c: c["final"], reverse=True):
        ok = (c["final"] >= threshold and c["ev_r"] >= C.MIN_EV_R and not c["veto"] and c["l3_pick"]
              and (regime["state"] != "Risk-Off" or c["setup"] in ("A", "C"))
              and c.get("sector_etf") not in sectors and len(picks) < max_picks)
        if ok:
            picks.append(c)
            sectors.add(c.get("sector_etf"))
        c["published"] = ok

    result = {
        "generated": datetime.now().isoformat(timespec="seconds"), "asof": asof, "regime": regime,
        "threshold": threshold, "max_picks": max_picks, "llm": llm_on,
        "backtest": {"available": stats is not None, "trades": stats["trades"] if stats else 0},
        "universe": len(feats), "n_setups": len(cands), "judge": judge,
        "picks": [card(c) for c in picks],
        "runners_up": [card(c) for c in top if not c["published"]][:8],
        "rejected": rejected[:40],
    }
    LATEST.write_text(json.dumps(result, default=str, indent=1))
    with PICKS_LOG.open("a") as fh:
        for p in result["picks"]:
            fh.write(json.dumps({"asof": asof, **{k: p[k] for k in ("symbol", "setup", "entry", "stop", "t1", "t2", "p_win", "ev_r", "final")}}) + "\n")
    progress("🏁 done")
    return result


def _earnings_gap(c):
    """Gap day must be 10-80 days after the last fiscal quarter end, or have an earnings headline."""
    gd = c["row"].get("C_gap_date")
    qe = c["fund"].get("last_quarter_end")
    if gd and qe and 10 <= (pd.Timestamp(gd) - pd.Timestamp(qe)).days <= 80:
        return True
    try:
        days = (date.today() - pd.Timestamp(gd).date()).days + 2
        heads = " ".join(n["headline"].lower() for n in alpaca.news(c["symbol"], days=days))
        return any(w in heads for w in ("earnings", "results", "quarter", "eps", "revenue", "guidance"))
    except Exception:
        return False


def digest(c, regime):
    row, p, df = c["row"], c["plan"], c["df"]
    sm = {"A": {"pivot": row["A_pivot"], "base_depth": row["A_depth"], "base_len_days": row["A_base_len"],
                "contraction_ratio": row["A_contraction"], "bb_width_pctile": row["A_bbw_pct"],
                "volume_dryup_ratio": row["A_dryup"], "breakout_rvol": row["rvol"]},
          "B": {"pullback_from_20d_high": row["B_pullback"], "pullback_volume_ratio": row["B_pb_vol"],
                "dist_to_ma_atr": row["B_dist_ma"], "trigger_rvol": row["rvol"]},
          "C": {"gap_date": row["C_gap_date"], "gap_size": row["C_gap_size"], "gap_atr": row["C_gap_atr"],
                "gap_rvol": row["C_gap_rvol"], "days_since_gap": row["C_gap_age"]}}[c["setup"]]
    c20 = df["close"].tail(21)
    fund = {k: v for k, v in c["fund"].items() if k not in ("symbol",)}
    return {
        "symbol": c["symbol"], "setup": SETUP_NAMES[c["setup"]],
        "setup_metrics": {k: _r(v, 3) if not isinstance(v, str) else v for k, v in sm.items()},
        "plan": {k: _r(v, 3) for k, v in p.items() if k != "setup"},
        "probability": {k: (_r(v, 3) if isinstance(v, (int, float)) else v) for k, v in c["prob"].items()},
        "ev_r": _r(c["ev_r"], 3),
        "factors": {"ranks_0_100": {k: _r(v, 0) for k, v in c["ranks"].items()},
                    "buckets": {k: _r(v, 0) for k, v in c["buckets"].items()},
                    "sector_etf": c.get("sector_etf"), "sector_rs": c.get("sector_rs")},
        "price_context": {"close": _r(row["close"]), "atr": _r(row["atr"]), "sma50": _r(row["sma50"]),
                          "sma200": _r(row["sma200"]), "ret_20d": _r(c20.iloc[-1] / c20.iloc[0] - 1, 3),
                          "high52_ratio": _r(row["high52"], 3), "rvol_today": _r(row["rvol"])},
        "fundamentals": fund, "fundamental_scores": c.get("fund_scores"), "regime": regime,
    }


def card(c):
    p, prob = c["plan"], c["prob"]
    days, iqr = S.expected_days(prob, p)
    e = p["entry"]
    p_w, p_t = prob["p_win"], min(prob["p_timeout"], 1 - prob["p_win"])
    return {
        "symbol": c["symbol"], "name": c["fund"].get("name"), "sector": c["fund"].get("sector"),
        "setup": c["setup"], "setup_name": SETUP_NAMES[c["setup"]],
        "entry": _r(e), "max_entry": _r(p["max_entry"]), "stop": _r(p["stop"]), "t1": _r(p["t1"]), "t2": _r(p["t2"]),
        "risk_pct": _r(p["risk_pct"] * 100, 1), "t1_pct": _r((p["t1"] / e - 1) * 100, 1),
        "t2_pct": _r((p["t2"] / e - 1) * 100, 1), "rr": _r(p["rr"], 1),
        "p_win": _r(p_w, 3), "p_loss": _r(1 - p_w - p_t, 3), "p_timeout": _r(p_t, 3),
        "baseline": _r(prob["baseline"], 3), "prob_source": prob["source"], "n": prob["n"],
        "ev_r": _r(c["ev_r"], 2), "ev_pct": _r(c["ev_r"] * p["risk_pct"] * 100, 2),
        "days": days, "days_range": iqr, "time_stop": int(min(C.MAX_HOLD, 2 * (days or 10))),
        "quant": _r(c["quant"], 1), "l1_adj": _r(c["l1_adj"], 1), "l2_penalty": _r(c["l2_penalty"], 1),
        "l3_adj": _r(c["l3_adj"], 1), "final": _r(c["final"], 1),
        "buckets": {k: _r(v, 0) for k, v in c["buckets"].items()},
        "veto": c.get("veto"), "l3_votes": c.get("l3_votes"),
        "l1": {a: {"stance": r["stance"], "confidence": r["confidence"], "summary": r.get("summary"),
                   "red_flags": r.get("red_flags")} for a, r in (c.get("l1") or {}).items() if r},
        "l2": c.get("l2"), "llm_card": c.get("card"), "issues": c.get("l3_issues"),
        "digest": c.get("digest"),
    }
