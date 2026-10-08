# 📈 Swing Setups

This tool finds long-stock swing trades, holding for about 3 to 30 days.
Deterministic rules find and price each setup. Three LLM layers then judge it.

- [`docs/DATA_POINTS.md`](docs/DATA_POINTS.md): which data is used, and the evidence behind it
- [`docs/STRATEGY.md`](docs/STRATEGY.md): the full rule set, the math, and the LLM layers

## Quick start
```bash
pip install -r requirements.txt
export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...   # free Alpaca account (paper keys work)
export ANTHROPIC_API_KEY=...                              # optional: enables the 3 LLM layers

python -m swing backtest   # one-time, about 15-30 min: calibrates the probabilities and holding times
python -m swing dips       # about 5 min: how every qualifying dip played out (past year + since 2015)
python -m swing            # opens http://localhost:8000 → 🔍 Run scan (best after 4:20pm ET)
python -m swing scan       # same scan, printed in the terminal
```

## 🛒 Active strategy: buy the dip
- **Qualifies**: close above a rising 200-day SMA, 50-day SMA above the 200-day, RSI(2) ≤ 10, and close at least 5% below the 20-day high.
- **Entry**: next open. A gap up of more than 0.5 ATR is skipped.
- **Stop**: 2 × ATR below entry (trades risking more than 8% are skipped).
- **Exit**: at the pre-dip 20-day high, or after 20 days.
- **Other setups**: A, B and C are still in the code. Turn them back on with `SWING_SETUPS=A,B,C,D`.

## 🗺️ Heat map
The 120 most-traded stocks are grouped by sector and colored from red (avoid) to green (best dip buy).
`rating = 0.40 × dip/oversold + 0.35 × trend + 0.25 × relative strength`
- A stock not in an uptrend is capped at 45.
- A stock that meets every dip rule (🛒) gets at least 75.
- Tap a tile to see its rating breakdown, the scan's verdict, and its past-year dip trades.

## How a pick is made
1. 🌎 **Universe**: US stocks ≥ $10 with ≥ $20M average daily dollar volume, and market cap ≥ $1B.
2. 🚦 **Regime**: a 0–10 score built from SPY/QQQ trend, breadth, VIX, VIX term structure and distribution days. It sets the score bar a pick must clear and the maximum number of picks.
3. 🔎 **Setups**: 📦 base breakout, 🪜 pullback in an uptrend, and 📣 post-earnings drift.
4. 🧮 **Quant score**: percentile ranks for momentum, relative strength, setup quality and volume, plus fundamentals and positioning.
5. 📐 **Trade plan**: a structural or ATR stop, a resistance-aware target (at least 2R), a backtested P(target before stop) shrunk toward the random-walk baseline, the expected value in R, and the expected days to target.
6. 🧠 **L1**: four specialist agents (technical with a chart image, fundamental, news, positioning). Adjustment is capped at ±10.
7. ⚔️ **L2**: a bear raises objections, a bull answers, and a referee rules. Penalty ranges from 0 to −15.
8. ⚖️ **L3**: a chief judge runs 3 times, and a pick needs at least 2 of 3 votes. Adjustment is capped at ±5, or the judge returns "no trade".

## Data sources (all free)
| Source | Used for |
|---|---|
| Alpaca market data | Daily bars (SIP, adjusted), news, paper trading |
| Yahoo Finance via `yfinance` | Earnings dates and surprises, quarterly financials, estimate revisions, insiders, short interest, sector, VIX |

`yfinance` is unofficial and can break or rate-limit. SEC EDGAR and FRED are planned as more robust sources.

## Honest limits
- The backtest only covers stocks listed today (survivorship bias), so the edge over baseline is cut by 30% before display.
- LLM layers can't be backtested, because the models may know what happened later. Judge them only on forward paper trades. Each pick is logged to `data/picks.jsonl`.
- Fundamentals use fixed scales rather than universe-wide ranks, to avoid pulling data for 2,000 tickers.
- This is a research tool, not financial advice.

## Tests
```bash
python -m pytest -q
```
