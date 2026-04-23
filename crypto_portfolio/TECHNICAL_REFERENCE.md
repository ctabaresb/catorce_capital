# Catorce Capital — Technical Reference
## Mathematical Foundations, Model Assumptions, and Methodology

*For the analyst maintaining and extending this platform. This document covers the "why" and "how" behind every quantitative decision. The operational README covers the "where" and "what to run".*

*Last updated: April 22, 2026*

---

## Table of Contents

1. [Investment Thesis](#1-investment-thesis)
2. [Portfolio Profile Design](#2-portfolio-profile-design)
3. [Portfolio Construction Strategies](#3-portfolio-construction-strategies)
4. [Rebalancing Mechanics and Fee Model](#4-rebalancing-mechanics-and-fee-model)
5. [Performance Metrics: Definitions and Assumptions](#5-performance-metrics-definitions-and-assumptions)
6. [Backtesting Methodology](#6-backtesting-methodology)
7. [Stochastic Simulation via Geometric Brownian Motion](#7-stochastic-simulation-via-geometric-brownian-motion)
8. [Model Assumptions and Limitations](#8-model-assumptions-and-limitations)
9. [Statistical Considerations](#9-statistical-considerations)
10. [Future Extensions](#10-future-extensions)

---

## 1. Investment Thesis

The core hypothesis is that a diversified, systematically rebalanced portfolio of crypto assets can deliver superior risk-adjusted returns compared to holding a single asset (typically BTC).

This rests on three observations:

**Observation 1 — Diversification reduces idiosyncratic risk.** Holding N assets instead of 1 lowers portfolio variance as long as assets are not perfectly correlated. Crypto correlations are high (typically 0.4 to 0.8 across major assets) but not perfect, so diversification still provides a measurable benefit.

**Observation 2 — Systematic rebalancing harvests volatility.** When one asset outperforms, rebalancing sells some of the winner and buys the laggard. In a mean-reverting environment this is equivalent to a disciplined "buy low, sell high" rule.

**Observation 3 — Different risk profiles need different allocations.** Not every investor wants maximum return. A conservative investor wants capital preservation with low drawdown; an aggressive investor tolerates large drawdowns in pursuit of higher CAGR. By parameterizing profiles and running the full backtest grid, we let the data reveal the optimal strategy per risk appetite rather than picking one by gut feeling.

The platform validates these hypotheses through two complementary methods: backtesting (did the strategy work in the past?) and Monte Carlo simulation (does it hold under alternative market scenarios?).

---

## 2. Portfolio Profile Design

### How profiles are defined

Each asset in the universe is assigned a `RiskTier` based on qualitative and quantitative criteria:

| RiskTier | Criteria | Typical assets |
|---|---|---|
| LOW | Top-10 market cap, 5+ years of trading history, deep liquidity across multiple exchanges, established network effects | BTC, ETH, SOL, BNB |
| MEDIUM | Top-50 market cap, 1+ years of history, meaningful liquidity, clear use case or protocol revenue | LINK, UNI, AAVE, XRP, HYPE, SUI |
| HIGH | Emerging or volatile, sub-top-50, high beta, thesis-driven | TAO, FET, AGIX, WLD, TIA |
| EXCLUDED | Stablecoins or assets that would distort return calculations | USDT, USDC, DAI |

Profiles are then defined by which risk tiers are eligible:

- **Conservative** = LOW only (4 assets currently)
- **Balanced** = LOW + MEDIUM (18 assets currently)
- **Aggressive** = LOW + MEDIUM + HIGH (27 assets currently)

This is a strict superset relationship: every Conservative asset is in Balanced, every Balanced asset is in Aggressive. This ensures that the optimizer has a weakly larger feasible set as risk tolerance increases, which guarantees (in theory) weakly better optimal portfolios for higher risk profiles.

### Why not use quantitative thresholds (e.g. volatility buckets)?

We considered tiering by realized volatility or market cap quantiles. The problem is regime dependence: BTC's annualized volatility ranged from 30% to 120% over the last 5 years. A token classified as "low volatility" in Q1 could be "high volatility" in Q3. Manual curation with qualitative criteria is more stable over time and avoids turnover-induced reclassification churn.

### Stablecoin treatment

Stablecoins are excluded from all portfolio profiles because including them as investable assets would distort both the covariance matrix (near-zero volatility creates ill-conditioned matrices) and performance metrics (a portfolio holding 50% stablecoins would have artificially low drawdown but also suppressed returns). 

---

## 3. Portfolio Construction Strategies

All strategies solve the same problem: given a universe of N eligible assets on a rebalance date, compute a vector of portfolio weights w = (w_1, w_2, ..., w_N) subject to:

```
w_i >= 0.01        (minimum weight: 1% per asset)
w_i <= 0.40        (maximum weight: 40% per asset)
sum(w_i) = 1.0     (fully invested, no cash)
w_i >= 0           (long-only, no shorting)
```

These constraints are enforced in `src/backtest/config.py` via `PortfolioConstraints`.


### 3.1 Equal Weight (1/N)

**Formula:**

```
w_i = 1/N   for all eligible assets i = 1, ..., N
```

**Rationale:** The DeMiguel, Garlappi, and Uppal (2009) result showed that 1/N often outperforms mean-variance optimization on out-of-sample Sharpe ratio when estimation error in expected returns is high. In crypto, return estimation is extremely noisy (short histories, regime changes), making 1/N a strong baseline.

**When it works well:** When the covariance matrix is hard to estimate reliably (few data points, unstable correlations) and expected returns are nearly unpredictable.

**When it fails:** When some assets are clearly dominated (negative expected return, extreme drawdowns). Equal weight allocates the same capital to winners and losers.


### 3.2 Market Capitalization Weighted

**Formula:**

```
w_i = MarketCap_i / sum(MarketCap_j)   for all eligible assets j
```

**Rationale:** This mirrors how broad market indices are constructed (e.g., S&P 500 is cap-weighted). The CAPM implies that the market portfolio is mean-variance efficient, so cap-weighting should approximate the efficient frontier if CAPM assumptions hold.

**In practice for crypto:** BTC dominance is typically 50-60% of total crypto market cap. A cap-weighted portfolio is heavily concentrated in BTC and ETH. This makes it low-variance but also limits the diversification benefit. For Conservative profiles with only 4 assets, BTC alone may receive 70%+ weight.

**Edge case handling:** If market cap data is missing for an asset on a given date, the strategy falls back to equal-weight for that asset.


### 3.3 Momentum (Volatility-Adjusted)

**Signal construction:**

```
raw_momentum_i = sum(r_i,t for t in [T-30, T]) / 30
vol_i = std(r_i,t for t in [T-30, T]) * sqrt(365)
signal_i = raw_momentum_i / vol_i
```

Where r_i,t is the daily log return of asset i on day t, and the lookback window is 30 calendar days.

**Weight computation:**

```
score_i = max(signal_i, 0)              # zero out negative momentum
w_i = score_i / sum(score_j)            # normalize to sum to 1
w_i = clip(w_i, min=0.01, max=0.40)     # enforce constraints
w_i = w_i / sum(w_j)                    # re-normalize after clipping
```

**Rationale:** Momentum is one of the most robust return factors in traditional finance (Jegadeesh and Titman, 1993). In crypto, momentum is even stronger due to retail-driven herding and narrative cycles. Dividing by volatility penalizes assets whose recent gains came with extreme variance (risk-adjusted signal rather than raw signal).

**When it works well:** Trending markets with persistent winners. Bull runs.

**When it fails spectacularly:** Mean-reverting or crash regimes. The strategy chases yesterday's winners, which in a reversal become today's biggest losers. In the April 2025 to April 2026 bear market, momentum aggressive was the worst-performing strategy at -69% CAGR.

**Lookback sensitivity:** 30 days is a short lookback. Longer windows (60, 90 days) are smoother but slower to react. The choice of 30 days is a compromise that favors responsiveness. This could be extended to a parameterized lookback in the grid.


### 3.4 Mean-Variance Optimization: Maximum Sharpe Ratio

This solves the classic Markowitz (1952) portfolio selection problem.

**Objective:**

```
maximize    w' * mu / sqrt(w' * Sigma * w)
subject to  sum(w_i) = 1
            0.01 <= w_i <= 0.40
```

Where:
- mu = vector of expected (mean) returns, estimated as the sample mean of daily log returns over the lookback window, annualized
- Sigma = covariance matrix of daily log returns, annualized
- w = weight vector

**Implementation:** The optimization is solved using `cvxpy` with the ECOS solver. Because the Sharpe ratio objective is a ratio (non-convex), it is reformulated as a convex quadratic program using the Cornuejols and Tutuncu (2006) transformation:

```
minimize    w' * Sigma * w
subject to  w' * mu = 1    (fix return, minimize variance)
            w_i >= 0
```

Then the solution is rescaled so weights sum to 1.

**Parameter estimation:**

| Parameter | Estimator | Window |
|---|---|---|
| mu (expected returns) | Sample mean of daily log returns, annualized (*365) | Full history available in Silver |
| Sigma (covariance) | Sample covariance of daily log returns, annualized (*365) | Full history available in Silver |

**Critical assumption:** MVO assumes that historical mean returns are a reasonable estimate of future expected returns. This is the weakest assumption in the entire platform. In crypto, mean returns are extremely noisy and regime-dependent. A token with +200% annualized return over 6 months of bull market has no guarantee of repeating. MVO will aggressively overweight that token, which is dangerous if the regime shifts.

**Regularization:** No shrinkage estimator is currently applied to the covariance matrix. For N > T (more assets than observations), the sample covariance is singular. In practice, with 27 assets and 300+ days of data, we have T > N so the matrix is invertible. However, applying Ledoit-Wolf shrinkage would improve out-of-sample stability and is a recommended extension.


### 3.5 Mean-Variance Optimization: Minimum Variance

**Objective:**

```
minimize    w' * Sigma * w
subject to  sum(w_i) = 1
            0.01 <= w_i <= 0.40
```

This ignores expected returns entirely and focuses only on minimizing portfolio variance.

**Rationale:** If expected returns are essentially unpredictable (a reasonable position in crypto), then the only reliable input is the covariance matrix. The minimum variance portfolio is the leftmost point on the efficient frontier and has the smallest estimation error of any mean-variance portfolio (Jagannathan and Ma, 2003).

**Behavior:** Concentrates in the lowest-volatility assets (typically BTC and stablecoins, but stablecoins are excluded). In bear markets, this strategy tends to outperform because it minimizes exposure to high-beta altcoins that crash hardest. In bull markets, it underperforms because it underweights the assets with the highest returns.


### 3.6 Risk Parity (Equal Risk Contribution)

**Objective:** Allocate weights so that each asset contributes equally to total portfolio risk.

The risk contribution of asset i is:

```
RC_i = w_i * (Sigma * w)_i / sqrt(w' * Sigma * w)
```

Where (Sigma * w)_i is the i-th element of the matrix-vector product. The optimization finds w such that:

```
RC_1 = RC_2 = ... = RC_N = (1/N) * sqrt(w' * Sigma * w)
```

**Implementation:** Solved via iterative optimization (scipy minimize with SLSQP solver). The objective function minimizes the sum of squared differences between each RC_i and the target (equal contribution).

**Rationale:** Risk parity was popularized by Bridgewater's All Weather fund. The idea is that equal-weight gives equal dollar allocation but unequal risk allocation (a 10% position in BTC contributes far more risk than a 10% position in LINK simply because BTC has higher absolute volatility in dollar terms). Risk parity equalizes the risk budget.

**Behavior in crypto:** Because crypto volatilities are all high (40-150% annualized), risk parity weights tend to cluster closer to equal weight than in traditional multi-asset portfolios (where bonds at 5% vol vs equities at 15% vol create large weight differentials). The strategy still provides a meaningful tilt toward lower-vol assets.

---

## 4. Rebalancing Mechanics and Fee Model

### Rebalancing frequencies

The grid tests 6 frequencies:

| Frequency | Calendar rule | Trades per year |
|---|---|---|
| Daily | Every calendar day | 365 |
| Weekly | Every Monday (or first trading day) | 52 |
| Biweekly | Every other Monday | 26 |
| Monthly | First day of each month | 12 |
| Quarterly | First day of each quarter | 4 |
| Annually | First day of each year | 1 |

### What happens on a rebalance date

1. The strategy computes target weights w_target for today.
2. The engine computes current weights w_current from the portfolio's mark-to-market values.
3. The delta for each asset is: delta_i = |w_target_i - w_current_i|.
4. The total turnover is: turnover = sum(delta_i) / 2 (divided by 2 because every dollar sold is a dollar bought).
5. The fee is applied to the turnover: fee_cost = turnover * fee_rate.
6. The portfolio NAV is reduced by fee_cost before computing the next day's return.

### Fee model detail

The fee is modeled as a **proportional cost on the traded notional**. This is how exchange fees actually work: you pay a percentage of each trade, not a fixed cost per rebalance event.

| Fee level | Maps to |
|---|---|
| 0.0% | Theoretical zero-cost baseline |
| 0.1% | Discount exchange |
| 0.2% | Standard exchange |
| 0.5% | High-friction venue or stressed scenario |

**Why delta-based, not notional-based:** If the portfolio already holds near-target weights (e.g., BTC drifted from 25% to 26%), only the 1% delta is traded, not the full 25% position. This correctly models real trading behavior and makes daily rebalancing viable even at non-zero fees.

### Rebalancing frequency tradeoff

Higher frequency captures more of the "rebalancing premium" (buying dips, selling rips more often) but incurs more cumulative fees. The optimal frequency depends on the fee level.

---

## 5. Performance Metrics: Definitions and Assumptions

### 5.1 Sharpe Ratio

**Definition:**

```
Sharpe = (R_p - R_f) / sigma_p
```

Where:
- R_p = annualized portfolio return
- R_f = risk-free rate (set to 0 in our implementation)
- sigma_p = annualized standard deviation of portfolio returns

**Annualization:** Daily returns are annualized by multiplying by 365 (crypto trades every day, no weekends). Daily standard deviation is annualized by multiplying by sqrt(365).

**Does the Sharpe ratio assume normally distributed returns?** No. The Sharpe ratio is defined for any return distribution. However, its *interpretation* depends on the distribution:

- Under normality, Sharpe = 1.0 means there is approximately a 16% probability of negative returns over a 1-year horizon (one standard deviation below the mean).
- Under fat-tailed distributions (which crypto returns follow), a Sharpe of 1.0 understates the true probability of extreme losses because the tails contain more mass than a normal distribution predicts.

In other words, the Sharpe ratio is always a valid *ranking* metric (higher is better, ceteris paribus), but its translation to "probability of loss" or "confidence of positive returns" is only accurate under normality. Since crypto returns are leptokurtic (fat-tailed) and often negatively skewed, the Sharpe ratio is somewhat optimistic about downside risk. This is why we also compute the Sortino ratio, VaR, and Expected Shortfall.

**Why risk-free rate = 0:** There is no universally accepted crypto risk-free rate. In traditional finance, the 3-month T-bill rate is used. Some practitioners use USDT staking yield (~2-5%) as a crypto risk-free proxy. We set it to 0 for simplicity and because the primary use case is comparing strategies against each other (relative ranking), not computing absolute excess returns.


### 5.2 Sortino Ratio

**Definition:**

```
Sortino = (R_p - R_f) / sigma_downside
```

Where sigma_downside is the standard deviation computed only on negative returns:

```
sigma_downside = sqrt(mean(min(r_t, 0)^2)) * sqrt(365)
```

**Key difference from Sharpe:** The Sharpe ratio penalizes upside volatility and downside volatility equally. The Sortino ratio only penalizes downside volatility. A strategy that has occasional large positive days (good) but steady small positive days otherwise gets a better Sortino than Sharpe.

**When Sortino diverges from Sharpe:** When the return distribution is highly skewed. A momentum strategy in a bull market may have Sharpe = 0.5 but Sortino = 1.2 because most of the volatility comes from positive jumps.


### 5.3 CAGR (Compound Annual Growth Rate)

**Definition:**

```
CAGR = (NAV_final / NAV_initial)^(365 / T_days) - 1
```

Where T_days is the number of calendar days in the backtest window.

**Sensitivity to start/end dates:** CAGR is heavily path-dependent. The same strategy can show +50% CAGR if the backtest starts at a local bottom and -30% CAGR if it starts at a local top. This is one of the key weaknesses the Sharpe ratio addresses (by normalizing by volatility). However, CAGR remains the most intuitive metric for investors ("how much would my money have grown?").

**Why we show both:** CAGR tells you absolute wealth creation. Sharpe tells you risk-adjusted efficiency. A strategy with CAGR = 50% and Sharpe = 0.3 is a wild ride. A strategy with CAGR = 15% and Sharpe = 1.2 is a much smoother path. Different investors care about different things.


### 5.4 Maximum Drawdown

**Definition:**

```
Drawdown_t = (NAV_t - NAV_peak_t) / NAV_peak_t
MaxDrawdown = min(Drawdown_t)   over all t
```

Where NAV_peak_t = max(NAV_s for s <= t) is the running maximum of NAV up to time t.

**Interpretation:** The worst peak-to-trough loss an investor would have experienced if they entered at the worst possible time and exited at the worst possible time. In crypto, max drawdowns of -50% to -80% are common.

**No distributional assumptions.** Max drawdown is purely empirical, computed directly from the NAV path.


### 5.5 Value at Risk (VaR)

**Definition (historical, 95th percentile):**

```
VaR_95 = -percentile(r_t, 5)
```

This is the 5th percentile of daily returns (the loss that is exceeded only 5% of the time).

**Interpretation:** "On 95% of days, your daily loss will be smaller than VaR." For example, VaR_95 = 5.58% means that on a typical day there is a 5% chance of losing more than 5.58% of portfolio value.

**Method:** We use historical simulation (the empirical quantile of observed returns), not parametric VaR. Parametric VaR assumes normality and uses mu - 1.645*sigma, which underestimates tail risk in crypto. Historical VaR captures the actual distribution including fat tails.


### 5.6 Expected Shortfall (ES / Conditional VaR)

**Definition (95th percentile):**

```
ES_95 = -mean(r_t | r_t <= -VaR_95)
```

The average loss on days when the loss exceeds VaR. This answers: "when things go wrong, how wrong do they go on average?"

**Why ES is better than VaR:** VaR tells you the threshold, ES tells you the expected damage beyond that threshold. VaR is not subadditive (the VaR of a combined portfolio can be larger than the sum of individual VaRs), which violates coherent risk measure axioms. ES is subadditive and is now preferred by regulators (Basel III uses ES instead of VaR for market risk capital requirements).


### 5.7 Beta

**Definition:**

```
Beta = Cov(r_portfolio, r_BTC) / Var(r_BTC)
```

**Interpretation:** Measures sensitivity to BTC as the "market" factor. Beta = 1.0 means the portfolio moves 1:1 with BTC. Beta = 0.5 means half the sensitivity. Beta > 1.0 means the portfolio amplifies BTC moves (common for altcoin-heavy portfolios).

**Why BTC as market:** BTC is the dominant systematic factor in crypto. When BTC crashes, almost everything else crashes harder (beta > 1 for most altcoins). When BTC rallies, altcoins often rally more (same mechanism). BTC serves the same role as the S&P 500 in equity factor models.


### 5.8 Calmar Ratio

**Definition:**

```
Calmar = CAGR / |MaxDrawdown|
```

**Interpretation:** Return per unit of worst-case drawdown. Combines the growth metric (CAGR) with the pain metric (MaxDrawdown) into a single number. Higher is better. A Calmar of 1.0 means every percent of max drawdown was "rewarded" with a percent of annualized growth.


### 5.9 Win Rate

**Definition:**

```
WinRate = count(r_t > 0) / count(r_t)
```

**Interpretation:** Percentage of days with positive returns. In crypto, win rates around 48-52% are typical. A strategy with a 55% win rate is exceptionally good. Note that win rate says nothing about the magnitude of wins vs losses. A 45% win rate can still be profitable if the average win is much larger than the average loss.

---

## 6. Backtesting Methodology

### 6.1 The backtest loop

For each combination in the grid (strategy x profile x frequency x fee_level):

```
1. Load Silver prices for the full backtest date range
2. Filter to assets eligible for this profile (in_conservative/balanced/aggressive)
3. Set NAV_0 = 1.0 (normalized)
4. For each calendar day t:
     a. If t is a rebalance date:
          - Call strategy.compute_weights(prices_up_to_t) -> w_target
          - Compute delta from w_current
          - Deduct fee = turnover * fee_rate from NAV
          - Set w_current = w_target
     b. Compute today's portfolio return: r_t = sum(w_current_i * r_asset_i_t)
     c. Update NAV: NAV_t = NAV_{t-1} * (1 + r_t)
5. Compute all metrics from the NAV series
6. Return one row of results
```

### 6.2 Look-ahead bias prevention

The backtest engine only passes data up to and including day t to the strategy. Prices on day t+1 are never visible to the weight computation. This is enforced by the `prices_up_to_t` slicing in `rebalancing.py`.

### 6.3 Survivorship bias

All 27 assets in the current universe are included in the backtest from their first available date in Silver. Assets that were delisted or lost significant value are not retroactively removed. However, because the universe is manually curated (not dynamically selected based on performance), there is inherent survivorship bias: we chose these 27 assets because they are "good" projects today. Tokens that failed and were delisted are not in the backtest at all.

This is a known limitation. Mitigations: (a) the universe includes tokens that have lost 50-70% of value (TAO, FET, AGIX in the current bear market) so it is not a pure "winners only" universe, and (b) the Monte Carlo simulation stress-tests strategies under alternative return paths that include scenarios worse than historical.

### 6.4 Transaction cost realism

The fee model applies proportional costs on each rebalance. It does not model: slippage (price impact of large trades), spread (bid-ask), minimum trade sizes, or partial fills. For a retail-scale portfolio ($10K-$1M), these second-order effects are small relative to the 0.1-0.5% fee levels tested. For institutional scale, a more detailed market impact model would be needed.

---

## 7. Stochastic Simulation via Geometric Brownian Motion

### 7.1 Why simulate?

Backtesting answers: "did this work in the past?" But there is only one historical path. The market could have evolved differently. Maybe the bull run lasted 2 months longer. Maybe the crash was 20% deeper. Maybe correlations broke down during a liquidity crisis.

Monte Carlo simulation generates thousands of alternative market histories that are statistically consistent with observed behavior. If a strategy performs well across most of these paths, we have stronger evidence that its backtest success was structural (driven by the strategy design) rather than lucky (driven by a fortunate sequence of market moves).

### 7.2 The price dynamics model

Each asset's price follows a Geometric Brownian Motion (GBM):

```
dS_t = mu * S_t * dt + sigma * S_t * dW_t
```

Where:
- S_t = price of the asset at time t
- mu = drift (annualized expected return)
- sigma = volatility (annualized standard deviation of returns)
- W_t = Wiener process (standard Brownian motion)

**Analytical solution (Ito's lemma):**

```
S_{t+dt} = S_t * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
```

Where Z ~ N(0,1) is a standard normal random variable.

The term (mu - sigma^2/2) is called the drift adjustment or Ito correction. It arises because the log of a geometric Brownian motion has drift mu - sigma^2/2, not mu. Without this correction, simulated prices would systematically overshoot.

### 7.3 Parameter estimation

| Parameter | Estimator |
|---|---|
| mu_i | Sample mean of daily log returns for asset i, annualized: mu_i = mean(log(S_{t+1}/S_t)) * 365 |
| sigma_i | Sample standard deviation of daily log returns, annualized: sigma_i = std(log(S_{t+1}/S_t)) * sqrt(365) |
| rho_ij | Pearson correlation between daily log returns of assets i and j |

**Stationarity assumption:** These estimators assume that mu and sigma are constant over time. This is a strong assumption. In reality, crypto volatility is highly time-varying (volatility clustering, GARCH effects). The impact is that simulated paths underestimate the frequency of volatility spikes and calm periods relative to reality. See Section 8 for discussion.

### 7.4 Multi-asset correlated simulation (Cholesky decomposition)

Crypto assets are highly correlated. Simulating each asset independently would break the correlation structure and produce unrealistic scenarios (e.g., BTC crashing while ETH rallies strongly, which rarely happens).

**Step 1 — Compute the correlation matrix R:**

```
R_ij = Corr(r_i, r_j)    for all eligible assets i, j
```

This is an N x N symmetric positive semi-definite matrix.

**Step 2 — Cholesky decomposition:**

```
R = L * L'
```

Where L is a lower-triangular matrix. This decomposition exists and is unique when R is positive definite.

**Step 3 — Generate correlated shocks:**

For each simulation step:
```
Z_independent = (Z_1, Z_2, ..., Z_N)    each Z_i ~ N(0,1) iid
Z_correlated = L * Z_independent
```

The vector Z_correlated has the property that Corr(Z_correlated_i, Z_correlated_j) = R_ij.

**Step 4 — Simulate prices:**

For each asset i and each time step:

```
S_i,{t+1} = S_i,t * exp((mu_i - sigma_i^2/2) * dt + sigma_i * sqrt(dt) * Z_correlated_i)
```

This produces N correlated price paths that respect the historical correlation structure.

### 7.5 Simulation grid

The simulation runs for each of the 18 strategy-profile combinations (6 strategies x 3 profiles):

- N_paths = 1,000 simulated market scenarios
- T = 365 days forward
- dt = 1 day
- Initial prices = last observed Silver prices
- Portfolio weights = from the latest backtest results (note: currently using equal weights as a proxy, see limitations)

### 7.6 Output statistics

For each strategy-profile combination, across 1,000 paths:

| Statistic | Definition |
|---|---|
| prob_positive_cagr | Fraction of paths where 1-year CAGR > 0 |
| cagr_p5 | 5th percentile of CAGR across paths (pessimistic scenario) |
| cagr_p50 | Median CAGR (central tendency) |
| cagr_p95 | 95th percentile of CAGR (optimistic scenario) |
| sharpe_p50 | Median Sharpe ratio across paths |
| n_assets | Number of assets used in this simulation |

### 7.7 Coverage threshold

An asset is included in the simulation only if it has data for at least 50% of the available return dates. This prevents assets with very short histories from introducing noise into the parameter estimates.

With the current 50% threshold and ~393 days of Silver data, an asset needs at least ~196 days of returns. All 27 universe assets clear this threshold after backfill.

The original threshold was 80%, which was too strict and left only BTC and ETH eligible. 50% is the current compromise. See Section 9 for the statistical justification.

---

## 8. Model Assumptions and Limitations

### 8.1 GBM assumptions that do NOT hold in crypto

| Assumption | Reality | Impact |
|---|---|---|
| Returns are normally distributed | Crypto returns are leptokurtic (excess kurtosis 3-10) with negative skew | Simulation underestimates tail events (extreme crashes and rallies) |
| Volatility is constant | Crypto volatility clusters (GARCH effects, regime switches) | Simulated paths are "too smooth" compared to reality |
| Drift is constant | Crypto has strong bull/bear regimes with very different mean returns | Simulation averages over regimes, missing the bimodal nature of crypto returns |
| Correlations are stable | Crypto correlations spike toward 1.0 during crashes ("correlation breakdown") | Simulation underestimates contagion risk during stress events |
| Continuous trading, no jumps | Crypto has exchange outages, flash crashes, black swan events | Simulation does not produce the discrete jumps seen in real markets |

### 8.2 Why use GBM despite these flaws?

GBM is a first-order approximation. It captures the two dominant features of price dynamics (trend + random volatility) in a mathematically tractable framework. More sophisticated models exist but add complexity without necessarily improving decision quality:

- **GGBM (Generalized GBM):** Replaces the normal distribution with the empirical distribution. Captures fat tails but requires much more data for reliable estimation of the tail shape.
- **Stochastic volatility (Heston, SABR):** Models volatility as a separate random process. Captures volatility clustering but doubles the parameter estimation burden.
- **Jump-diffusion (Merton):** Adds Poisson jumps to the diffusion process. Captures flash crashes but requires estimation of jump intensity and jump size distribution.
- **VAR on volatility:** Models the covariance matrix as a time-varying process. Captures correlation dynamics but is computationally expensive for 27 assets.

For an MVP platform with 1 year of data and 27 assets, GBM provides the right tradeoff between model complexity and estimation reliability. Assuming normality completely shatters altcoins, since they don't have even by the slightest normal returns. The simulation results should be interpreted as a central-tendency guide, not a precise risk forecast.

### 8.3 Estimation risk

With ~365 days of daily data per asset, the standard error of the mean daily return estimate is:

```
SE(mu_hat) = sigma / sqrt(T) ≈ 0.04 / sqrt(365) ≈ 0.002 daily
```

Annualized, this is an SE of ~0.7 percentage points. For a true annualized return of 20%, the 95% CI is roughly [18.6%, 21.4%]. This seems precise, but the real uncertainty is much larger because the stationarity assumption is violated (mu is not constant over the year).

For the covariance matrix, the estimation error is even more problematic. The sample covariance with T observations and N assets has (N*(N+1)/2) free parameters. For N=27, that is 378 parameters from 365 observations. The ratio T/N = 13.5 is acceptable (rule of thumb: T/N > 10 for a well-conditioned covariance estimate) but not generous. Ledoit-Wolf shrinkage would reduce this estimation error and is recommended.

### 8.4 Backtest limitations

- **Only 1 year of history.** The April 2025 to April 2026 window was one of the worst for altcoins. Results may not generalize to bull markets.
- **No execution modeling.** Assumes trades execute at the exact closing price with no slippage or market impact.
- **Universe survivorship bias.** The 27 assets were selected because they are currently "good" projects, not randomly sampled from all historical crypto assets.
- **Weights used as proxy in simulation.** GBM simulation currently uses equal weights rather than actual strategy-specific weights.

---

## 9. Statistical Considerations

### 9.1 Why Sharpe is more robust than CAGR

CAGR depends on the exact start and end dates. Shift the window by 30 days and the number can change dramatically. The Sharpe ratio, because it normalizes by standard deviation, converges under the Law of Large Numbers:

```
Sharpe_hat = mean(r_t) / std(r_t)
```

As T grows, both the numerator and denominator converge to their population values. The rate of convergence is O(1/sqrt(T)), meaning the standard error of the Sharpe estimate is approximately:

```
SE(Sharpe_hat) ≈ sqrt((1 + Sharpe^2/2) / T)
```

For Sharpe = 0.5 and T = 365 daily observations:

```
SE ≈ sqrt((1 + 0.125) / 365) ≈ 0.055
```

So a measured Sharpe of 0.62 has a 95% CI of roughly [0.51, 0.73]. This is relatively tight but still wide enough that the difference between the #1 and #5 strategy may not be statistically significant.

### 9.2 Sample size for stable Cholesky decomposition

The Cholesky decomposition requires the correlation matrix to be positive definite. With N assets and T observations, the sample correlation matrix is guaranteed positive definite when T > N (we have 393 > 27). However, some eigenvalues may be very small, leading to numerical instability.

The 50% coverage threshold ensures each asset pair has at least ~196 joint observations for correlation estimation. The rule of thumb for stable pairwise correlation estimates is T > 30 (textbook) or T > 100 (conservative). 196 is well above both.

### 9.3 Interpreting simulation results

The 1,000-path simulation produces a distribution of outcomes. Useful interpretations:

- **prob_positive_cagr > 0.5:** The strategy is more likely to make money than lose money under the modeled distribution. This is a necessary but not sufficient condition for recommending the strategy.
- **cagr_p5 > -30%:** Even in the pessimistic 5th percentile scenario, the drawdown is manageable.
- **cagr_p95 / cagr_p5 ratio:** Measures the width of the outcome distribution. A ratio of 5 means there is 10x uncertainty between optimistic and pessimistic scenarios. High ratios indicate the strategy is "high conviction" (either great or terrible), low ratios indicate robustness.

**What the simulation does NOT tell you:** It does not account for regime changes not present in the estimation window. If the next year brings a regulatory crackdown or a stablecoin depeg that has no precedent in the training data, the simulation will not have generated paths resembling that event.

---

## 10. Future Extensions

### Near-term (improve current model)

- **Ledoit-Wolf shrinkage** on the covariance matrix. Reduces estimation error, improves MVO stability. ~2 hours of implementation.
- **Actual strategy weights in simulation** instead of equal-weight proxy. Requires persisting the latest weight vector from each backtest combination. ~1 day.
- **Rolling-window backtests** (e.g., 6-month rolling Sharpe) to assess temporal stability. A strategy with stable rolling Sharpe is more trustworthy than one with high average Sharpe driven by a single good quarter.
- **Multiple lookback windows for momentum** (30, 60, 90 days) as additional grid dimensions.

### Medium-term (extend the model)

- **GGBM simulation.** Replace the normal distribution with the empirical distribution of returns (bootstrap resampling). Captures fat tails without additional parameters.
- **Block bootstrap.** Resample blocks of consecutive returns (e.g., 5-day blocks) to preserve autocorrelation and volatility clustering.
- **CoinGecko Analyst upgrade.** Unlocks 5 years of history, enabling backtests across the 2021 bull, 2022 crash, 2023 recovery, 2024 rally, and 2025 correction. This would dramatically improve estimation reliability and strategy validation.
- **DCA overlay.** Combine systematic rebalancing with dollar-cost-averaging inflows. 
- **Stablecoin cash allocation.** Allow a configurable percentage of the portfolio in USDT/USDC as a volatility dampener. 

### Long-term (new capabilities)

- **Factor models.** Decompose crypto returns into systematic factors (market, size, momentum, liquidity) and construct factor-tilted portfolios.
- **Regime detection.** Use hidden Markov models to classify bull/bear/sideways regimes and switch strategy allocations dynamically.
- **On-chain signals.** Incorporate exchange flows, active addresses, and fee revenue as alpha signals.
- **Live trading via Bitso/Hyperliquid API.** Execute the optimal strategy's rebalancing trades automatically.

---

## Glossary

| Term | Definition |
|---|---|
| Alpha | Return in excess of a benchmark (typically BTC in this platform) |
| Beta | Sensitivity of portfolio returns to BTC returns |
| Cholesky decomposition | Factorization of a positive definite matrix R = L*L' into a lower-triangular matrix L |
| CAGR | Compound Annual Growth Rate |
| Convex optimization | Optimization where the objective and constraints are convex functions, guaranteeing a global optimum |
| Covariance matrix | Matrix where element (i,j) is Cov(r_i, r_j), the covariance of returns between assets i and j |
| Drawdown | Percentage decline from a previous peak in portfolio value |
| Efficient frontier | Set of portfolios that offer the highest expected return for each level of risk |
| ES (Expected Shortfall) | Average loss in the worst q% of scenarios (also called CVaR) |
| GBM | Geometric Brownian Motion: stochastic process where log returns are normally distributed |
| Ito's lemma | Calculus rule for functions of stochastic processes, used to derive the GBM closed-form solution |
| Leptokurtic | Distribution with fatter tails than normal (excess kurtosis > 0) |
| Log return | ln(S_{t+1} / S_t), the continuously compounded return. Preferred over simple returns because they are additive across time |
| MVO | Mean-Variance Optimization (Markowitz framework) |
| NAV | Net Asset Value of the portfolio, normalized to 1.0 at inception |
| Risk parity (ERC) | Equal Risk Contribution: portfolio where each asset contributes equally to total portfolio variance |
| Sharpe ratio | Excess return per unit of total risk (standard deviation) |
| SDE | Stochastic Differential Equation: differential equation involving random processes |
| Sortino ratio | Excess return per unit of downside risk (downside standard deviation) |
| Turnover | Fraction of the portfolio traded during a rebalance event |
| VaR | Value at Risk: the loss threshold exceeded with probability q (typically 5%) |
| Wiener process | Continuous-time stochastic process with independent, normally distributed increments (Brownian motion) |

---

*This document is for internal use by the Catorce Capital team. It is not investment advice. Past performance does not guarantee future results. Crypto assets are volatile and you can lose your entire investment.*
