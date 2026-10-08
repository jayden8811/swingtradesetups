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
python -m swing            # opens http://localhost:8000 → 🔍 Run scan (best after 4:20pm ET)
python -m swing scan       # same scan, printed in the terminal
```

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
