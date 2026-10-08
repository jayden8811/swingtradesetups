"""All tunable thresholds and weights live here (see docs/STRATEGY.md)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SWING_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
for _d in (DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Stage 0: universe filters
MIN_PRICE = 10.0
MIN_DOLLAR_VOL = 20e6          # 50-day average
MIN_MARKET_CAP = 1e9
MIN_BARS = 260
EARNINGS_BLACKOUT_DAYS = 21    # calendar days (~15 trading days)
EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
NAME_EXCLUDE = ("etf", "fund", "trust", "warrant", "right", "unit", "preferred",
                "notes", "depositary shares", "proshares", "ishares", "direxion",
                "leveraged", "2x", "3x", "bull", "bear", "acquisition corp")

# Stage 1: regime thresholds -> (min final score, max picks)
REGIME = {"Risk-On": (70, 3), "Neutral": (78, 1), "Risk-Off": (88, 1)}

# Stage 2: setups
BASE_MIN, BASE_MAX = 15, 65
BASE_MAX_DEPTH = 0.30
A_RVOL, A_CLV, A_MAX_EXT_ATR = 1.5, 0.6, 0.75
B_RS_MIN = 70
C_GAP_MIN, C_GAP_ATR, C_RVOL = 0.04, 1.5, 3.0
# Setup D (buy the dip): uptrend (close > rising SMA200, SMA50 > SMA200) + RSI(2) <= 10 + >= 5% off 20-day high
D_RSI2_MAX, D_MIN_DIP, D_STOP_ATR, D_MIN_RR, D_MAX_HOLD = 10, 0.05, 2.0, 1.0, 20
ACTIVE_SETUPS = tuple(os.environ.get("SWING_SETUPS", "D").split(","))

# Stage 4: trade plan
STOP_ATR = 2.0
STOP_BUFFER_ATR = 0.25
MAX_RISK_PCT = 0.08
MIN_RISK_ATR = 0.75
MIN_RR = 2.0
MAX_RR_T1 = 3.0
MAX_HOLD = 30
SLIPPAGE = 0.0005              # per side
SPREAD = 0.0005                # assumed round-trip half-spread for liquid names
MIN_EV_R = 0.15
PRIOR_STRENGTH = 20            # Beta prior worth N trades
LIVE_EDGE_HAIRCUT = 0.7        # survivorship-bias haircut on backtested edge

# Stage 3: factor bucket weights (QuantScore)
WEIGHTS = {"momentum": 25, "rs": 20, "setup": 20, "volume": 10, "fundamental": 15, "positioning": 10}
TECH_BUCKETS = ("momentum", "rs", "setup", "volume")   # backtestable, price-only

HEATMAP_N = 120               # "main tickers": most-traded names by 50-day dollar volume

# LLM layers
LLM_MODEL = os.environ.get("SWING_LLM_MODEL", "claude-opus-5-5")
LLM_TOP_N, L2_TOP_N, L3_TOP_N, L3_RUNS = 15, 8, 5, 3
LLM_WORKERS = int(os.environ.get("SWING_LLM_WORKERS", "8"))

SECTOR_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Basic Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
