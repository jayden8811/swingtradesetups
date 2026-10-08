"""Vectorized indicators on a single OHLCV DataFrame or numpy arrays."""
import numpy as np
import pandas as pd


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return wilder(tr, n)


def rsi(close, n=14):
    d = close.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def clv(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return ((df["close"] - df["low"]) / rng).fillna(0.5)


def bb_width(close, n=20, k=2):
    m, sd = sma(close, n), close.rolling(n).std()
    return 2 * k * sd / m


def trend_r2(close, n=63):
    """Signed R^2 of log price vs time over a rolling window (smoothness of trend)."""
    y = np.log(close)
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    my, mt = y.rolling(n).mean(), t.rolling(n).mean()
    cov = (y * t).rolling(n).mean() - my * mt
    vy = (y * y).rolling(n).mean() - my * my
    vt = (n * n - 1) / 12
    r = cov / np.sqrt(vy * vt)
    return np.sign(r) * r * r


def swing_highs(high, k=5):
    """Boolean array: bar is a local max with k bars on each side (only confirmed k bars later)."""
    h = np.asarray(high, float)
    out = np.zeros(len(h), bool)
    for i in range(k, len(h) - k):
        w = h[i - k:i + k + 1]
        out[i] = h[i] == w.max()
    return out


def windows(a, n):
    """Trailing windows: row t holds a[t-n .. t-1] (excludes t). Rows < n are NaN."""
    a = np.asarray(a, float)
    v = np.full((len(a), n), np.nan)
    if len(a) > n:
        v[n:] = np.lib.stride_tricks.sliding_window_view(a, n)[:-1]
    return v


def lin(x, x0, x1):
    """Map x linearly so x0 -> 0 and x1 -> 100, clipped. Works for x0 > x1 too."""
    return np.clip((np.asarray(x, float) - x0) / (x1 - x0) * 100, 0, 100)
