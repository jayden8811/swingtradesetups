# Published Strategies vs SPY (2016-01 → 2026-10)

Run it: `python -m swing research` (about 8 minutes). The results appear on the site under 📚.

## Why these four
Each candidate is among the most-cited, publicly documented rules-based equity strategies. All four can be built from free daily price data:

| Strategy | Source | Rules (as published) |
|---|---|---|
| 12-1 momentum | Jegadeesh & Titman (1993), *Journal of Finance* | Monthly. Hold the top 50 large caps by return from 12 months ago to 1 month ago, equal weight |
| Stocks on the Move | Clenow (2015) | Weekly. 90-day exponential regression slope × R². Buy only when SPY is above its 200-day; stock above its 100-day; no 15% gap; ATR risk-parity sizing |
| Global Equities Momentum | Antonacci (2014) | Monthly. Hold SPY or EFA, whichever has the higher 12-month return, if it beats T-bills; otherwise hold bonds (AGG) |
| 10-month trend | Faber (2007) | Monthly. Hold SPY when above its 10-month SMA; otherwise hold T-bills |

Context from the literature:
- About 79% of active large-cap funds trailed the S&P 500 in 2025 (SPIVA).
- Published anomalies lose roughly half their excess return after publication (McLean & Pontiff 2016; replicated by Chen & Zimmermann).
- The momentum premium fell from about 10.9% a year (1927-93) to about 5.5% (1994-2015).

## Method
- **Execution**: signals at the close, trades at the next open, 0.10% cost per unit of turnover. Idle cash earns BIL.
- **Universe**: the point-in-time top 500 stocks by 60-day dollar volume. It is drawn from every active and delisted ticker Alpaca has, about 6,100 names.
  - **Bias check**: with today's survivors only, 12-1 momentum showed +25.4% a year. Adding delisted names cut it to +20.0%, and its max drawdown grew from −38% to −50%.
  - **What's left**: coverage of delisted tickers is incomplete, so some bias remains.
- **Pass rule** (written before seeing results): return above SPY, Sharpe above SPY, beats an equal-weight portfolio of the same universe, wins at least 60% of rolling 3-year windows, and nearby parameter settings also beat SPY.

## Results
| Strategy | CAGR | Sharpe | Max DD | Beta | Alpha/yr | Beats SPY (rolling 3-yr) |
|---|---|---|---|---|---|---|
| SPY | +15.1% | 0.79 | −32% | 1.00 | – | – |
| 12-1 momentum | +20.0% | 0.62 | −50% | 1.47 | +3.3% | 71% |
| Stocks on the Move | +9.9% | 0.48 | −26% | 0.74 | −0.7% | 16% |
| Dual momentum | +7.1% | 0.39 | −32% | 0.75 | −4.2% | 0% |
| 10-month trend | +7.9% | 0.50 | −26% | 0.55 | −1.1% | 2% |
| Equal-weight universe | +10.7% | 0.52 | −36% | 1.05 | −4.1% | 10% |

**Verdict: none passed, so nothing is applied.**
- **12-1 momentum** came closest. It returned more than SPY, but only by taking about 1.5× the market's risk. Its Sharpe is lower than SPY's, and it lost 50% at its worst.
- **Momentum settings**: the 100-stock version matched SPY's return almost exactly (+15.0% vs +15.1%). The 25-stock version had a −63% drawdown.
- **Trend and dual momentum**: these cut drawdowns in theory but lagged badly in a decade where SPY rarely stayed down for long.
- **The site**: it shows 12-1 momentum's current portfolio, clearly labeled as not passing, for anyone who wants to accept that risk.

---

# Active Mega-Cap Trading: Buy Weakness, Sell the Bounce

Run it: `python -m swing active` (about 1 minute). The results appear on the site under ⚡.

**Setup**
- **Universe**: the point-in-time 30 most-traded stocks (MSFT, NVDA, AAPL, …), delisted names included.
- **Periods**: 2016–2020 in-sample, 2021–2026 out-of-sample.
- **Costs**: 0.05% per side.
- **Sizing**: 10 equal slots.
- **Controls**: SPY, QQQ, holding the same 30 stocks, and random entries with the same exits.

| Strategy (published source) | 2016–26 | 2016–20 | 2021–26 | Sharpe | Max DD | Time in market |
|---|---|---|---|---|---|---|
| **IBS** (Pagonidis; arXiv 2306.12434): buy close in bottom 20% of day's range above 200-day; sell close > prior high | **+19.7%** | +20.6% | +19.1% | 0.97 | −29% | 62% |
| RSI(2) (Connors & Alvarez 2008) | +13.6% | +7.4% | +19.4% | 0.92 | −13% | 28% |
| Weekly reversal (Lehmann 1990; de Groot et al. 2010) | +5.9% | +10.2% | +2.1% | 0.34 | −57% | 78% |
| SPY | +15.1% | +15.1% | +14.9% | 0.92 | −32% | 100% |
| QQQ | +20.5% | +24.4% | +17.1% | 0.97 | −37% | 100% |
| Hold the same 30 stocks | +15.9% | +16.6% | +15.2% | 0.70 | −50% | 100% |
| Random entries, same exits | +7.3% | +7.8% | +7.0% | 0.44 | −42% | 64% |

**Strongest candidate: IBS.**
- It beat SPY and the same stocks held passively, in both halves, and timing clearly matters (random entries returned 7%).
- Nearby settings also held up: thresholds of 0.15 and 0.25, and next-open entry at +18.4%.

**What it didn't pass:**
- Its 2021–26 Sharpe (0.85) was below SPY's (0.92).
- It beat SPY in only 5 of 11 calendar years; most of the lead came from volatile years (2020, 2025, 2026).
- It roughly matched QQQ over the whole period.

**Very cost-sensitive**: at 0.15% per side it returns +10.7%/yr. Use commission-free, liquid names with market-on-close orders.

**RSI(2)** was excellent out-of-sample (Sharpe 1.10, −13% max DD while in the market only 28% of the time) but weak in 2016–20.
