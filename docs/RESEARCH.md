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
