# Swing Setup Engine: Rules, Calculations, and LLM Judgment Layers

Long-only stock swing trades held for about 3 to 30 trading days.
Data inputs are listed in [DATA_POINTS.md](DATA_POINTS.md).

Design principles:
1. **Rules find the candidates and the LLMs judge them.** Every number (entry, stop, target, probability) comes from deterministic code. The LLM layers can adjust scores only within fixed bounds, and they can veto.
2. **Rank by expected value, not by how good a chart looks.** The final ranking uses the expected R-multiple. That depends on the hit rate *and* the reward:risk.
3. **"No trade" is a valid output.** On many days nothing meets the bar.
4. **Calibrate before trusting.** All weights below are starting priors. They get replaced by values fitted on walk-forward backtests.

```
Universe (~1,500 names)
  └─ Stage 0  Hard filters (liquidity, price, events)
  └─ Stage 1  Market regime gate            → sets threshold & max picks
  └─ Stage 2  Setup detection (A/B/C)       → ~20–80 candidates
  └─ Stage 3  Factor scoring (percentile ranks) → Quant Score 0–100
  └─ Stage 4  Trade plan math (entry/stop/targets/R:R/time)
  └─ Stage 5  Expected value & ranking      → top 15 go to the LLMs
  └─ LLM L1   Specialist analysts (per data domain)
  └─ LLM L2   Bull vs. Bear adversarial review
  └─ LLM L3   Chief Judge (final pick or "no trade")
  └─ Output   Setup card + paper trade on Alpaca + outcome logging
```

---

## Stage 0: Universe & Hard Filters

A stock is dropped if any of these fail:

| Filter | Rule | Why |
|---|---|---|
| Listing | US common stock on NYSE/Nasdaq/AMEX; no ETFs, ADRs ok | Data consistency |
| Price | Close ≥ $10 | Low-priced stocks have higher spreads and manipulation risk |
| Liquidity | 50-day avg dollar volume ≥ $20M | Lets you exit at the stop with little slippage |
| Market cap | ≥ $1B | Fewer gap/dilution blowups |
| Spread | Median quoted spread ≤ 0.15% of price | Execution cost |
| Earnings | Next earnings date **not** within the next 15 trading days (setup C is exempt because it trades *after* earnings) | Avoids binary gap risk inside the holding window |
| Corporate events | No pending merger target, no S-1/S-3/424B offering filed in the last 10 days | Prices are pinned, or dilution overhang |
| Data quality | ≥ 260 daily bars, no unadjusted splits | Indicators need history |

---

## Stage 1: Market Regime Gate

Most swing longs fail in a falling market, so the regime decides how selective to be.

| Component | Points |
|---|---|
| SPY close > 200-day SMA | +2 |
| SPY 50-day SMA > 200-day SMA | +1 |
| QQQ close > 50-day SMA | +1 |
| Breadth: % of universe above its 50-day SMA ≥ 50% | +2 (≥ 35% → +1) |
| VIX < 20 → +2; 20–28 → +1; > 28 → 0 | 0–2 |
| VIX / VIX3M < 1.0 (term structure in contango) | +1 |
| Distribution days in SPY over the last 25 sessions ≤ 4 (a day down ≥ 0.2% on higher volume) | +1 |

**Regime score 0–10:**

| Regime | Score | Min final score to publish | Max picks |
|---|---|---|---|
| Risk-On | 7–10 | 70 | 3 |
| Neutral | 4–6 | 78 | 1 |
| Risk-Off | 0–3 | 88 (and setup must be A or C) | 1 (often 0) |

---

## Stage 2: Setup Detection (deterministic)

The universe stays only if a stock matches at least one setup. Each setup's statistics are tracked separately because their hit rates and holding times differ.

Common definitions:
- `ATR` = Wilder's 14-day Average True Range. `ATR%` = ATR / close.
- `RVOL` = today's volume / 50-day average volume.
- `CLV` (close location) = (close − low) / (high − low).
- **Trend Template** (all must be true): close > SMA50 > SMA150 > SMA200; SMA200 higher than it was 20 days ago; close ≥ 1.30 × 52-week low; close ≥ 0.75 × 52-week high.

### Setup A: Base Breakout (volatility contraction)
1. Trend Template passes.
2. Base: a range of 15–65 trading days whose high-to-low depth is ≤ 30% (≤ 20% preferred).
3. Contraction: at least 2 successive pullbacks inside the base, each shallower than the last (e.g., 18% → 9% → 4%), **or** 20-day Bollinger Band width in the lowest 20th percentile of its own 1-year history.
4. Volume dry-up: the 10-day average volume over the last part of the base is < 0.75 × the 50-day average.
5. **Trigger:** close > pivot (base high) **and** RVOL ≥ 1.5 **and** CLV ≥ 0.6.
6. Not extended: close ≤ pivot + 0.75 × ATR (avoids chasing).

### Setup B: Pullback in an Uptrend
1. Trend Template passes. The stock's RS rank (Stage 3) is ≥ 70.
2. Pullback: price is down 3–12% from its 20-day high and has touched within 0.5 ATR of the rising 21-day EMA or the 50-day SMA.
3. Orderly: average volume during the pullback < 0.9 × the 50-day average. No pullback day with a close below the 50-day SMA on RVOL > 1.5.
4. Oversold short-term: RSI(2) ≤ 15 during the pullback, **or** RSI(14) between 38 and 52.
5. **Trigger:** close > the prior day's high, with CLV ≥ 0.5.

### Setup C: Post-Earnings Drift (PEAD)
1. Earnings released 1–10 trading days ago.
2. Earnings-day gap ≥ max(4%, 1.5 × ATR) **and** earnings-day RVOL ≥ 3.
3. Standardized surprise `SUE` in the top 20% of the universe (formula below), **or** revenue beat with raised guidance (verified from the 8-K by LLM L1).
4. Price has held above the gap day's low since, with no close below the gap-day midpoint.
5. **Trigger:** close above the high of a 3- to 8-day tight flag after the gap, **or** an intraday reclaim of the gap-day high.

---

## Stage 3: Factor Scoring

Every factor is converted to a **cross-sectional percentile rank (0–100)** across the full filtered universe that day. Ranks are scale-free, robust to outliers, and comparable across factors.

### Factor formulas
| Factor | Formula |
|---|---|
| `MOM_12_1` | P(t−21) / P(t−252) − 1. The most recent month is skipped because returns over 1 month tend to reverse |
| `MOM_6_1` | P(t−21) / P(t−126) − 1 |
| `RS` | 0.4·r63 + 0.2·r126 + 0.2·r189 + 0.2·r252, where rN = stock N-day return − SPY N-day return |
| `HIGH52` | close / 52-week high |
| `SECTOR_RS` | 63-day return rank of the stock's sector ETF among the 11 SPDR sectors (top 3 → 100, bottom 3 → 0, linear between) |
| `TREND_QUAL` | R² of a linear regression of log(price) over 63 days × sign(slope). Smooth trends beat choppy ones |
| `SETUP_QUAL` | Setup A: average of ranks for (−base depth), (−BB-width percentile), (−dry-up ratio), RVOL on trigger. B: (−pullback volume ratio), (−distance to MA in ATR), RS. C: SUE, gap/ATR, earnings-day RVOL |
| `VOL_ACCUM` | Up/Down volume ratio over 50 days = Σ volume on up days / Σ volume on down days |
| `SUE` | (EPS actual − EPS consensus) / σ(last 8 surprises). If consensus is unavailable, use (EPS_q − EPS_q−4) / σ of that difference over 8 quarters |
| `GROWTH` | Average rank of YoY revenue growth, YoY EPS growth, and revenue growth acceleration (this quarter's YoY − last quarter's YoY) |
| `REVISIONS` | Net % change in the next-fiscal-year EPS estimate over 30 days (when available, otherwise neutral = 50) |
| `INSIDER` | 100 if ≥ 2 distinct officers/directors made open-market buys (Form 4 code **P** only, excluding 10b5-1 plans) totaling ≥ $100k in 90 days; 75 for one such buyer; 50 none; 35 for heavy net selling outside 10b5-1 |
| `SHORT` | Non-linear: Short interest % of float 0–10% → 50. Above 10% with RS ≥ 70 → 65 (squeeze fuel). Above 15% with RS < 50 → 25 (informed shorts) |
| `NEWS` | 50 by default. Set by LLM L1 (bounded, see below), not here |

### Composite Quant Score (starting weights)

| Bucket | Factors (equal within bucket) | Weight |
|---|---|---|
| Momentum & trend | MOM_12_1, MOM_6_1, HIGH52, TREND_QUAL | 25% |
| Relative strength | RS, SECTOR_RS | 20% |
| Setup quality | SETUP_QUAL | 20% |
| Volume/accumulation | VOL_ACCUM | 10% |
| Fundamentals/earnings | SUE, GROWTH, REVISIONS | 15% |
| Positioning | INSIDER, SHORT | 10% |

`QuantScore = Σ weight × bucket average`, range 0–100.

**Where the weights come from:** price momentum, 52-week-high proximity, industry momentum, and post-earnings drift have the most replicated academic evidence, so they carry the most weight. Insider buying is well documented but rare, so it sits in a smaller bucket. After backtesting, these weights are replaced by coefficients from a regularized logistic regression of *P(hit target before stop)* on the factor ranks. It is refit quarterly, walk-forward.

---

## Stage 4: Trade Plan Math

### Entry
- A: pivot + 0.1% (buy-stop), or the next open if it is already triggered and still within 0.75 ATR of the pivot.
- B: prior day's high + 0.1%.
- C: flag high + 0.1%.

### Stop (structural, then capped)
```
S_struct = (lowest low of last 5 bars for B/C, or last contraction low for A) − 0.25 × ATR
S_atr    = Entry − 2.0 × ATR
Stop     = max(S_struct, S_atr)            # use the tighter of the two
R        = Entry − Stop                    # risk per share
```
**Reject** if R / Entry > 8% or R < 0.75 × ATR (a stop that tight gets hit by normal noise).

### Targets
```
T1 = min( nearest overhead resistance above Entry + 1R,     # prior swing high / 52-wk high / gap fill
          Entry + 3R )
T2 = Setup A: Entry + base depth (measured move)
     Setup B: prior 20-day high + 1 × ATR
     Setup C: Entry + 1 × earnings gap size
RR1 = (T1 − Entry) / R
```
**Reject** if RR1 < 2.0.
Exit plan: sell half at T1 and move the stop to breakeven; trail the rest under the 10-day EMA (close basis) toward T2.
**Time stop:** exit if neither the target nor the stop is hit within `2 × expected days`, capped at 30 trading days.

### Probability of hitting the target (the statistically honest part)
A no-edge random walk gives a baseline probability that the target is hit before the stop:

```
P_base = R / (T1 − Stop) = 1 / (1 + RR1)        # e.g. RR 2 → 33.3%, RR 3 → 25%
```

Any edge must show up as a hit rate above this baseline. From the backtest of each **setup × score-bucket** cell (for example, Setup A with QuantScore 80–90):

```
p_hat  = (hits + α) / (n + α + β)    # Beta-Binomial shrinkage toward baseline
         with α = 20·P_base, β = 20·(1 − P_base)   (prior worth 20 trades)
```
This keeps a cell with only 12 historical trades from claiming a 70% win rate.

### Expected value (in R units, net of costs)
```
EV_R = p_hat × RR1  −  (1 − p_hat) × 1  −  cost_R
cost_R = (2 × spread/2 + 2 × 0.05% slippage) × Entry / R
```
Example: RR 2.5, p_hat 0.42 → EV = 1.05 − 0.58 − 0.04 = **+0.43R** per trade.
**Reject** if EV_R ≤ 0.15.

### Expected time to target
Primary: **median days-to-T1 among winning trades** in the same setup × score cell of the backtest, with the interquartile range shown.

Fallback (before enough history): a diffusion estimate. With daily volatility σ ≈ ATR and the distance d = T1 − Entry, a random walk needs roughly (d/σ)² days. With a positive drift μ per day (from the backtest), the estimate becomes:
```
days_est ≈ d / μ          if μ > 0 and d/μ < (d/σ)²
           (d / σ)²       otherwise
```
Displayed as a range, for example "6–14 trading days (median 9)".

### Projected outcome (what the card shows)
- Win case: T1 hit with probability p_hat, +RR1 R.
- Loss case: stop hit with probability ≈ (1 − p_hat − p_timeout), −1R.
- Time-stop case: p_timeout, average outcome from the backtest (usually around 0R).
- Expected % return on the position = EV_R × (R / Entry).

---

## Stage 5: Pre-LLM Ranking
```
RankScore = 0.6 × QuantScore  +  0.4 × percentile_rank(EV_R)
```
The top **15** by RankScore that pass every rejection rule go to the LLM layers. If fewer than 3 pass, the LLMs still run on what is there, but the bar is unchanged.

---

## LLM Judgment Layers

General rules for all three layers:
- **Inputs are structured JSON** of computed values plus raw text (news, 8-K excerpts, earnings call highlights). The LLM must **cite field names/values** for each claim. A claim with no citation is discarded by a validator.
- **Outputs are strict JSON schemas.** Temperature 0. A malformed output gets one retry, then that layer counts as neutral.
- **Bounded authority.** LLMs never change entry/stop/target numbers. They adjust a score inside fixed limits or veto.
- **Look-ahead leakage:** an LLM may "know" what happened after a historical date, so LLM layers **cannot be backtested honestly**. Their value is measured only **forward**, through Alpaca paper trades. Each pick is logged with and without the LLM adjustments, to test whether the layers add value.

### Layer 1: Specialist Analysts (parallel, one per domain)
Each candidate gets 4 specialist reviews:

| Agent | Reads | Judges what the rules can't |
|---|---|---|
| **Technical** | OHLCV summary, computed levels, plus a rendered chart image (daily, 1 year, with MAs and volume) | Does the base look like a real setup or a broken one (e.g., wide and loose, failed breakouts, overhead supply)? Is the stop below a logical level? |
| **Fundamental/Earnings** | EDGAR XBRL facts, latest 8-K/10-Q text, earnings-release text | Quality of the beat (one-time items? guidance raised or cut?), margin trend, dilution |
| **News & Catalyst** | Last 30 days of Alpaca/Finnhub headlines, 8-K items | Is there a real catalyst, or is the move news-less? Upcoming binary events (FDA date, trial, lawsuit ruling, index inclusion)? Is the news already priced in? Sets `NEWS` |
| **Positioning & Macro** | Insider, short interest, sector RS, regime data | Does the sector or macro backdrop help or fight this trade? Are insider buys meaningful relative to compensation? |

Output schema (each agent):
```json
{ "stance": -2..+2, "confidence": 0..1,
  "evidence": [{"claim": "...", "source_field": "..."}],
  "red_flags": [{"flag": "...", "severity": "low|med|high"}],
  "hard_veto": false, "veto_reason": null }
```
Score effect: `L1_adj = Σ_agents clip(stance × confidence × 2.5, −5, +5)`, then the total is clipped to **±10**.
A `hard_veto` is allowed only for a specific list of reasons: undisclosed binary event inside the window, active offering/dilution, fraud/delisting/going-concern language, or a merger arb situation.

### Layer 2: Adversarial Review (Bull vs. Bear)
Applied to the top 8 after Layer 1.

1. **Bear agent** (prompted to argue the trade will fail) gets all data plus the L1 outputs. It must produce its **3 strongest falsifiable objections**, each tied to data. Examples: "price is 6% above the pivot (1.4 ATR), so the entry is extended"; "sector ETF is below its 50-day"; "breakout volume RVOL 1.6 is borderline"; "similar setup failed in this name 2 months ago".
2. **Bull agent** answers each objection, again only with data.
3. **Referee** (a separate call) labels each objection: `refuted`, `partially_refuted`, or `stands`, with a severity of 1–3.

```
L2_penalty = −Σ (severity × {stands: 3, partially_refuted: 1.5, refuted: 0})   clipped to [−15, 0]
```
Layer 2 can only lower a score. Its job is to catch what the optimism of the rules and L1 misses.

### Layer 3: Chief Judge (Portfolio Manager)
Gets the top 5 after L2: all numbers, all L1/L2 outputs, the regime, and current open positions.

Tasks:
1. **Consistency check:** do the trade plan numbers make sense (stop below support, target below obvious resistance, time estimate realistic for the stock's ATR)? Flag any inconsistency, which sends the candidate back to the rules for a recompute or rejects it.
2. **Comparative ranking:** comparing all 5 side by side, which has the best mix of evidence strength, EV, and clean risk? Adjustment of **±5** per candidate.
3. **Diversification:** at most 1 pick per sector, and no pick with 60-day return correlation > 0.7 to an existing open position.
4. **Decision:** pick the best setup(s) up to the regime's max, **or return "NO TRADE"** with a reason.
5. **Write the setup card reasoning:** plain-language thesis, the 3 strongest supporting data points, the top risks (from L2), and what would invalidate the trade.

**Self-consistency:** L3 runs 3 times independently. A pick is published only if it wins **≥ 2 of 3** runs. Otherwise the output is "low agreement, no trade".

### Final Score
```
FinalScore = QuantScore + L1_adj (±10) + L2_penalty (−15..0) + L3_adj (±5)
Publish if FinalScore ≥ regime threshold  AND  EV_R ≥ 0.15  AND  no veto  AND  L3 majority pick
```
The quant engine always contributes at least 70% of the possible score range. The LLM layers can promote or demote a candidate but cannot create a trade the rules didn't find.

---

## Output: The Setup Card
```
TICKER — Setup A: Base Breakout            FinalScore 84 (Quant 81, L1 +6, L2 −4, L3 +1)
Regime: Risk-On (8/10)
Entry  $52.40 (buy-stop)      Stop $49.10 (−6.3%, 1R = $3.30)
Target 1  $59.00 (+12.6%, 2.0R)   Target 2  $63.80 (+21.8%)
P(T1 before stop): 44% (backtest n=312 for this cell, baseline 33%)
Expected value: +0.29R ≈ +1.8% per trade after costs
Expected time to T1: 7–15 trading days (median 10); time stop at 20 days
Thesis / 3 key data points / Top risks / Invalidation: ...
Position size at 1% account risk: shares = (0.01 × equity) / 3.30
```

---

## Validation Plan (before trusting any number)
1. **Backtest data:** 10+ years of daily bars. Free data has **survivorship bias** (delisted stocks are missing), so backtest results will be optimistic. Note this on the card, and reduce live expectations accordingly (for example, multiply p_hat edge over baseline by 0.7).
2. **Execution realism:** enter on the next bar's stop trigger (never the signal bar's close), include spread and slippage, and treat a gap below the stop as a fill at the open.
3. **Walk-forward:** fit on years 1–5, test year 6, roll forward. Report only out-of-sample results.
4. **Overfitting control:** limit the parameters that are tuned (pivot buffer, ATR multiples, thresholds) to coarse grids. Report the number of configurations tried, and use a deflated Sharpe ratio / multiple-testing haircut.
5. **Benchmarks:** compare against (a) buying SPY, (b) random entries in the same universe with the same stop/target rules, and (c) the quant-only ranking without LLM layers.
6. **Forward test:** paper trade every published pick on Alpaca's paper account for at least 3 months / 50 trades before relying on the probabilities.
7. **Monitoring:** track calibration with a reliability chart of predicted p vs. realized hit rate. If the gap exceeds 10 points over 50 trades, refit.

*This is a research tool, not financial advice. Historical edges decay and can disappear.*
