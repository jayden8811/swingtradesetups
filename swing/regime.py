"""Stage 1: market regime score (0-10) from SPY/QQQ trend, breadth, VIX and distribution days."""
import pandas as pd

from .indicators import sma


def regime_frame(spy: pd.DataFrame, qqq: pd.DataFrame, breadth: pd.Series,
                 vix: pd.Series | None, vix3m: pd.Series | None) -> pd.DataFrame:
    idx = spy.index
    c, v = spy["close"], spy["volume"]
    q = qqq["close"].reindex(idx).ffill()
    r = pd.DataFrame(index=idx)
    r["spy_above_200"] = (c > sma(c, 200)) * 2
    r["spy_50_over_200"] = (sma(c, 50) > sma(c, 200)) * 1
    r["qqq_above_50"] = (q > sma(q, 50)) * 1
    b = breadth.reindex(idx)
    r["breadth"] = (b >= 50) * 2 + ((b >= 35) & (b < 50)) * 1
    dist = ((c / c.shift() - 1 <= -0.002) & (v > v.shift())).astype(int).rolling(25).sum()
    r["distribution_ok"] = (dist <= 4) * 1
    if vix is not None and len(vix):
        vx = vix.reindex(idx).ffill()
        r["vix"] = (vx < 20) * 2 + ((vx >= 20) & (vx <= 28)) * 1
        if vix3m is not None and len(vix3m):
            r["vix_term"] = ((vx / vix3m.reindex(idx).ffill()) < 1.0) * 1
        else:
            r["vix_term"] = 0
    else:  # VIX unavailable: score the rest and scale to 10
        r["vix"], r["vix_term"] = 0, 0
    pts = r.sum(axis=1)
    if vix is None or not len(vix):
        pts = pts * 10 / 7
    r["score"] = pts.round().clip(0, 10)
    r["breadth_pct"] = b
    r["dist_days"] = dist
    r["state"] = pd.cut(r["score"], [-1, 3, 6, 10], labels=["Risk-Off", "Neutral", "Risk-On"]).astype(str)
    return r
