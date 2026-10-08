# Swing Trade Data Points (Free Sources)

Scope: long stock swing trades, holding period ~3 days to ~6 weeks.
Each section maps to one specialist agent. "Evidence" notes how well the signal
is supported by published research vs. mostly practitioner convention.

## 1. Price & Trend Agent
| Data point | What it tells you | Free source | Evidence |
|---|---|---|---|
| Price vs. 20/50/200-day moving averages, MA slope | Trend direction & stage | Alpaca bars (connected), yfinance | Practitioner standard; trend-following has moderate academic support |
| 3/6/12-month price momentum (skip last month) | Persistent winners tend to keep winning | Alpaca / yfinance | Strong (Jegadeesh & Titman, many replications) |
| Relative strength vs. SPY and vs. sector ETF | Leadership vs. market | Alpaca / yfinance (SPY, XLK, XLF, etc.) | Moderate–strong (momentum family) |
| Distance from 52-week high | Stocks near highs tend to break higher | Alpaca / yfinance | Strong (George & Hwang 2004) |
| ATR (14) & ATR % of price | Volatility → stop distance, position size, target range | Computed from bars | Standard for risk sizing |
| Base / consolidation detection (range contraction, Bollinger width) | Setup quality before breakout (VCP, flags, pullbacks) | Computed from bars | Practitioner (Minervini, O'Neil); limited formal evidence |
| Support/resistance levels, prior swing highs/lows | Entry, stop, and price targets | Computed from bars | Practitioner |
| RSI(14), MACD | Pullback depth / momentum shift | Computed from bars | Weak–mixed alone; useful as filter |

## 2. Volume & Liquidity Agent
| Data point | What it tells you | Free source | Evidence |
|---|---|---|---|
| Relative volume (today vs. 50-day avg) | Institutional participation on breakouts | Alpaca / yfinance | Practitioner; moderate |
| Up-volume vs. down-volume (accumulation/distribution), OBV | Buying vs. selling pressure | Computed from bars | Mixed |
| Average dollar volume | Tradability / slippage filter | Computed | Essential filter |
| Bid-ask spread | Execution cost | Alpaca quotes (IEX feed on free tier) | Essential filter |

## 3. Fundamentals & Earnings Agent
| Data point | What it tells you | Free source | Evidence |
|---|---|---|---|
| Earnings surprise & post-earnings gap | Post-earnings announcement drift | yfinance, Finnhub free tier, SEC EDGAR | Strong (PEAD, Bernard & Thomas) |
| EPS & revenue growth (QoQ, YoY), acceleration | Fundamental tailwind | SEC EDGAR XBRL "companyfacts" API | Moderate |
| Analyst estimate revisions | Rising expectations | Finnhub free tier (limited), Yahoo | Moderate–strong |
| Next earnings date | Event risk inside the holding window | Nasdaq calendar, yfinance, Finnhub | Essential risk filter |
| Margins, debt/equity, float, market cap | Quality & how explosive a move can be | SEC EDGAR, yfinance | Moderate (quality factor) |

## 4. Smart Money / Positioning Agent
| Data point | What it tells you | Free source | Evidence |
|---|---|---|---|
| Insider open-market purchases (Form 4, especially clusters) | Insiders buying with own money | SEC EDGAR Form 4 (free, near real-time) | Moderate–strong for buys; sells are weak signal |
| Institutional holdings changes (13F) | Fund accumulation | SEC EDGAR 13F (45-day lag) | Weak–moderate due to lag |
| Short interest & days-to-cover | Squeeze fuel or bearish conviction | FINRA (twice monthly), yfinance | Mixed (high SI predicts lower returns on average, but fuels squeezes) |
| Daily short-sale volume ratio | Short activity trend | FINRA Reg SHO daily files | Weak–mixed |
| Put/call ratio (stock & market) | Sentiment extreme | CBOE (market-level), yfinance option chains | Weak–mixed, contrarian at extremes |

## 5. News & Sentiment Agent
| Data point | What it tells you | Free source | Evidence |
|---|---|---|---|
| Company news headlines & catalysts (FDA, contracts, guidance) | Why price is moving | Alpaca News API (connected), Finnhub, SEC 8-K filings | Event-dependent |
| Headline sentiment score (LLM-scored) | Tone of coverage | Alpaca News + LLM | Weak–moderate; noisy |
| 8-K material events | Official catalysts | SEC EDGAR | Strong as fact source |

## 6. Market Regime / Macro Agent
| Data point | What it tells you | Free source | Evidence |
|---|---|---|---|
| SPY/QQQ vs. 50/200-day MA | Is the market in an uptrend (most swing longs fail in downtrends) | Alpaca / yfinance | Moderate–strong |
| Market breadth (% of stocks above 50-day MA, advance/decline) | Health of the rally | Computed from universe bars | Moderate |
| VIX level & trend | Risk appetite / volatility regime | FRED (VIXCLS), yfinance | Moderate |
| Sector ETF rankings (relative strength) | Where money is rotating | Alpaca / yfinance | Moderate (industry momentum) |
| 10Y yield, 2s10s curve, credit spreads, dollar | Macro headwinds | FRED | Weak–moderate for short horizons |
| Economic calendar (CPI, FOMC, jobs) | Event risk in the window | FRED release calendar, Fed site | Risk filter |

## Output Derivations (for the final setup card)
- **Entry**: breakout level or pullback-to-support level.
- **Stop**: below recent swing low or 1.5–2× ATR.
- **Target(s)**: next resistance, measured move of the base, or 2–3R (R = entry − stop).
- **Expected time to target**: (target − entry) ÷ typical daily move (ATR), checked against historical backtest of similar setups.
- **Projected outcome / probability**: from backtesting each setup type on historical data — not stated without a backtest.

## Data Access Notes
- Alpaca (connected here): free tier gives historical bars, IEX real-time quotes, and news. SIP (full consolidated) data requires paid plan.
- yfinance is unofficial scraping of Yahoo; can break or rate-limit.
- SEC EDGAR requires a descriptive User-Agent header and ≤10 requests/sec.
- Finnhub free tier: ~60 calls/min.
