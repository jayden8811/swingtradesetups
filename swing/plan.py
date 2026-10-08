"""Stage 4: trade plan math (entry/stop/targets) and trade simulation."""
import numpy as np

from . import config as C


def resistance_levels(high, i, lookback=252, k=5):
    """Confirmed swing highs (k bars each side) in the lookback window, plus the 52-week high."""
    lo = max(0, i - lookback)
    h = high[lo:i + 1]
    levels = [h.max()]
    for j in range(k, len(h) - k):
        if h[j] == h[j - k:j + k + 1].max():
            levels.append(h[j])
    return np.unique(levels)


def build(f, i, setup, high, entry=None):
    """Trade plan for signal row i. `f` is the features frame, `high` the raw high array.
    Returns (plan, None) or (None, rejection reason)."""
    row = f.iloc[i]
    entry = float(row["close"] if entry is None else entry)
    atr = float(row["atr"])
    s_struct = float(row[f"{setup}_stop"])
    stop = max(s_struct, entry - C.STOP_ATR * atr)
    R = entry - stop
    if not np.isfinite(R) or R <= 0:
        return None, "entry below structural stop"
    if R / entry > C.MAX_RISK_PCT:
        return None, f"risk {R / entry:.1%} > {C.MAX_RISK_PCT:.0%}"
    if R < C.MIN_RISK_ATR * atr:
        return None, "stop inside normal noise (<0.75 ATR)"
    max_hold = C.MAX_HOLD
    if setup == "D":   # dip buy: the pre-dip 20-day high is both the target and the resistance
        t1, max_hold = float(row["D_t1"]), C.D_MAX_HOLD
        rr = (t1 - entry) / R
        if rr < C.D_MIN_RR - 1e-9:
            return None, f"20-day high {t1:.2f} only {rr:.1f}R away"
        t2 = max(float(row["D_t2"]), t1 + 0.5 * R)
    else:
        lv = resistance_levels(high, i)
        above = lv[lv > entry + R]
        t1 = min(above.min(), entry + C.MAX_RR_T1 * R) if len(above) else entry + C.MAX_RR_T1 * R
        rr = (t1 - entry) / R
        if rr < C.MIN_RR - 1e-9:
            return None, f"resistance at {above.min():.2f} caps reward at {rr:.1f}R"
        if setup == "A":
            t2 = entry + float(row["A_t2_add"])
        elif setup == "B":
            t2 = float(row["B_t2"])
        else:
            t2 = entry + float(row["C_gap_size"])
        t2 = max(t2, t1 + 0.5 * R)
    return {"setup": setup, "entry": entry, "stop": stop, "R": R, "t1": t1, "t2": t2, "rr": rr,
            "atr": atr, "max_entry": float(row[f"{setup}_max_entry"]),
            "risk_pct": R / entry, "max_hold": max_hold, "cost_r": (C.SPREAD + 2 * C.SLIPPAGE) * entry / R}, None


def simulate(o, h, l, c, i, plan, max_hold=None):
    """Enter at the next open; stop first if both levels touch on one bar (conservative).
    Returns dict(outcome in {win, loss, timeout}, r, days)."""
    entry, stop, t1, R = plan["entry"], plan["stop"], plan["t1"], plan["R"]
    max_hold = max_hold or plan.get("max_hold", C.MAX_HOLD)
    last = min(len(c) - 1, i + max_hold)
    if last <= i:
        return None
    for j in range(i + 1, last + 1):
        if l[j] <= stop:
            px = min(o[j], stop) if j > i + 1 else stop
            return {"outcome": "loss", "r": (px - entry) / R - plan["cost_r"], "days": j - i}
        if h[j] >= t1:
            px = max(o[j], t1) if j > i + 1 else t1
            return {"outcome": "win", "r": (px - entry) / R - plan["cost_r"], "days": j - i}
    if last < i + max_hold:
        return None  # still open at end of data
    return {"outcome": "timeout", "r": (c[last] - entry) / R - plan["cost_r"], "days": last - i}
