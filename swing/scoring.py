"""Stage 3 factor scores and Stage 4 probability / expected value from backtest statistics."""
import json

import numpy as np
import pandas as pd

from . import config as C
from .indicators import lin

RANKED = ["mom_12_1", "mom_6_1", "rs_raw", "high52", "trend_q", "vol_accum"]
STATS_PATH = C.DATA_DIR / "backtest_stats.json"


def cross_ranks(values: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank (0-100) of each column across the universe (rows = symbols)."""
    return values.rank(pct=True) * 100


def setup_quality(setup, row, rs_rank):
    if setup == "A":
        return row["A_quality"]
    if setup == "B":
        return (2 * row["B_quality_part"] + rs_rank) / 3
    if setup == "D":
        return row["D_quality"]
    return row["C_quality_part"]


def tech_buckets(rk, quality):
    """Price-only bucket scores (0-100). `rk` maps factor -> rank."""
    return {
        "momentum": _mean([rk["mom_12_1"], rk["mom_6_1"], rk["high52"], rk["trend_q"]]),
        "rs": rk["rs_raw"],
        "setup": quality,
        "volume": rk["vol_accum"],
    }


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else 50.0


def weighted(buckets, keys):
    w = sum(C.WEIGHTS[k] for k in keys)
    return sum(C.WEIGHTS[k] * (50 if pd.isna(buckets.get(k)) else buckets[k]) for k in keys) / w


def tech_bucket_label(score):
    edges = [0, 50, 60, 70, 80, 101]
    for lo, hi in zip(edges, edges[1:]):
        if lo <= score < hi:
            return f"{lo}-{min(hi, 100)}"
    return "0-50"


# ---------- fundamentals & positioning (absolute scales: stable without a full-universe pull) ----------

def fundamental_scores(fd):
    out = {}
    s = fd.get("surprise_pct")
    out["SUE"] = None if s is None else float(lin(s, -0.05, 0.15))
    beats = fd.get("beats_last4")
    if out["SUE"] is not None and beats is not None:
        out["SUE"] = 0.7 * out["SUE"] + 0.3 * beats / 4 * 100
    g = [x for x in (fd.get("rev_yoy"), fd.get("eps_yoy")) if x is not None]
    parts = [float(lin(x, -0.10, 0.40)) for x in g]
    if fd.get("rev_accel") is not None:
        parts.append(float(lin(fd["rev_accel"], -0.10, 0.10)))
    out["GROWTH"] = float(np.mean(parts)) if parts else None
    rv = fd.get("revision_30d")
    out["REVISIONS"] = None if rv is None else float(lin(rv, -0.05, 0.05))
    vals = [v for v in out.values() if v is not None]
    out["bucket"] = float(np.mean(vals)) if vals else 50.0
    return out


def positioning_scores(fd, rs_rank):
    buyers, buy_value, net_sell = fd.get("insider_buyers", 0), fd.get("insider_buy_value", 0), fd.get("insider_net_sell_value", 0)
    if buyers >= 2 and buy_value >= 100_000:
        ins = 100
    elif buyers >= 1:
        ins = 75
    elif net_sell > 5_000_000:
        ins = 35
    else:
        ins = 50
    si = fd.get("short_pct_float")
    if si is None:
        short = 50
    elif si > 0.15 and rs_rank < 50:
        short = 25
    elif si > 0.10 and rs_rank >= 70:
        short = 65
    else:
        short = 50
    return {"INSIDER": ins, "SHORT": short, "bucket": (ins + short) / 2}


# ---------- probability & EV from backtest cells ----------

def load_stats():
    if STATS_PATH.exists():
        return json.loads(STATS_PATH.read_text())
    return None


def _shrink(wins, n, prior_p, strength=C.PRIOR_STRENGTH):
    return (wins + strength * prior_p) / (n + strength)


def probability(stats, setup, bucket, regime, rr):
    """Hierarchical Beta-Binomial shrinkage:
    random entry (same stop/target/time rules) -> setup -> setup x bucket -> setup x bucket x regime,
    then the edge over random entry is haircut for survivorship bias."""
    base = 1 / (1 + rr)
    out = {"baseline": base, "n": 0, "p_win": base, "p_timeout": 0.0, "timeout_r": 0.0,
           "days_median": None, "days_iqr": None, "source": "no backtest yet: random-walk baseline only"}
    if not stats:
        return out
    cells = stats["cells"]
    rnd = cells.get("RANDOM")
    if rnd and rnd["n"]:
        # what a random entry with this R:R wins under the same rules (time stop included)
        base = float(np.clip(base * rnd["win_rate"] / rnd["baseline"], 0.01, 0.95))
        p_to, to_r = rnd["timeouts"] / rnd["n"], rnd["timeout_r"]
    else:
        p_to, to_r = 0.0, 0.0
    p, n, used, days, iqr = base, 0, "random entry", None, None
    for key in (setup, f"{setup}|{bucket}", f"{setup}|{bucket}|{regime}"):
        c = cells.get(key)
        if not c or not c["n"]:
            continue
        p = _shrink(c["wins"], c["n"], p)
        p_to = _shrink(c["timeouts"], c["n"], p_to)
        to_r = (c["timeouts"] * c["timeout_r"] + C.PRIOR_STRENGTH * to_r) / (c["timeouts"] + C.PRIOR_STRENGTH)
        n, used = c["n"], key
        days, iqr = c["days_median"] or days, c["days_iqr"] or iqr
    p = base + C.LIVE_EDGE_HAIRCUT * (p - base)
    out.update(baseline=base, p_win=float(np.clip(p, 0.01, 0.95)), n=n, p_timeout=float(p_to),
               timeout_r=float(to_r), days_median=days, days_iqr=iqr,
               source=f"backtest {used.replace('|', ' / tech ', 1).replace('|', ' / ')} (n={n}), "
                      "shrunk toward random entry, edge haircut 30%")
    return out


def expected_value(prob, plan):
    p_w, p_t = prob["p_win"], min(prob["p_timeout"], 1 - prob["p_win"])
    p_l = 1 - p_w - p_t
    return p_w * plan["rr"] - p_l * 1 + p_t * prob["timeout_r"] - plan["cost_r"]


def expected_days(prob, plan, drift_r_per_day=None):
    if prob.get("days_median"):
        return prob["days_median"], prob["days_iqr"]
    d, sigma = plan["t1"] - plan["entry"], plan["atr"]
    est = (d / sigma) ** 2
    if drift_r_per_day:
        est = min(est, d / (drift_r_per_day * plan["R"]))
    est = float(np.clip(est, 2, C.MAX_HOLD))
    return round(est), [round(est * 0.6), round(min(C.MAX_HOLD, est * 1.6))]
