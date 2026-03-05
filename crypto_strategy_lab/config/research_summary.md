# Crypto Strategy Research — Summary of Prior Work

## Exchange & Constraints
- **Exchange:** Bitso (Mexico)
- **Direction:** Long-only spot (no shorting)
- **Fees:** Zero maker and taker
- **Assets:** BTC/USD, ETH/USD, SOL/USD
- **Data:** Minute-level BBO + DOM, no trade prints
- **Avg BTC spread:** ~4.5–5.0 bps median — this is the friction floor

---

## Infrastructure Built
| Component | File | Status |
|-----------|------|--------|
| Feature engineering | build_features.py | ✅ Production |
| 15m decision bars | features_decision_15m_{asset}_{days}d.parquet | ✅ Available |
| Daily bars | build_daily_bars.py | ✅ Production |
| Holding period scanner | holding_period_scanner.py | ✅ Production |
| Sanity checks | sanity_check_features.py | ✅ Production |

### Key Features Available (no new engineering needed)
- Microprice delta, WIMB, depth imbalance, notional imbalance
- Spread metrics (BBO + DOM), tradability score, regime score
- EMA 30m/120m + slopes, Bollinger, Donchian, Ichimoku
- RSI, realized volatility (30m, 120m), vol-of-vol
- Cross-asset returns: ETH and SOL ret/rv at 15m
- Forward returns: H60m, H120m, H240m (all with validity flags)

---

## Strategies Tested and Eliminated

| Strategy | Horizon | Long Gross | Verdict |
|----------|---------|------------|---------|
| Microprice Imbalance | H60m | 0.022 bps | ❌ Dead |
| Volatility Compression Breakout | H120m | -14.1 bps | ❌ Dead |
| Time-of-Day Seasonality (hours 0,10,20,23) | H120m | +10.1 bps gross | ❌ Dead on 180d |
| Pullback-in-Trend | H120m | -8.5 bps, n=13 | ❌ Dead (sample) |
| Holding Period Scanner (daily, 1-15d) | 1-15d | All negative | ❌ Bear market window |

### Root Causes of Failure
1. BTC declined ~27% over 180-day test window — long-only strategies face structural headwind
2. Intraday gross edge consistently below 4.5 bps friction floor
3. Time-of-day edge was 60-day bear market artifact, not structural
4. Pullback-in-trend has too few uptrend periods in bearish window

---

## Key Lessons
1. **Regime filter is mandatory** — long-only signals must require confirmed uptrend
2. **180-day minimum window** — 60 days is insufficient for temporal validation
3. **Temporal stability rule** — edge must be positive in ≥2 of 3 independent segments
4. **Friction floor** — gross edge must exceed 2× avg spread (~9 bps) to be reliable
5. **Short-side edge exists** but is not exploitable on Bitso

---

## New Research Direction: Modular Strategy Scanner

### 5 Orthogonal Strategies to Test
| # | Strategy | Signal Source | Key Features |
|---|----------|--------------|--------------|
| 1 | Spread Compression | Execution quality | spread_bps_bbo_p50, tradability_score |
| 2 | Cross-Asset Divergence | Relative returns | eth/sol ret vs btc ret |
| 3 | Book Depth Exhaustion | DOM dynamics | bid_depth_k, depth_imb_k, wimb_last |
| 4 | Volatility Reversion | Vol regime change | rv_bps_30m, rv_bps_120m, vol_of_vol |
| 5 | Ichimoku Cloud Breakout | Trend initiation | ichi_above_cloud, ema_120m_slope |

### Timeframes
- 15m bars (existing)
- 30m bars (build_features.py update needed)
- Daily bars (existing)

### Evaluation (Identical for All Strategies)
- Net mean > 0
- Positive in ≥2/3 temporal segments  
- n ≥ 30 trades
- Gross > 2× avg spread
- Regime filter (uptrend gate) applied before signal

---

## Next Steps
1. Update build_features.py to accept `--bar_size` parameter (15m, 30m)
2. Build base_strategy.py abstract class
3. Build evaluator.py single evaluation engine
4. Implement 5 strategy modules
5. Build run_scanner.py entry point
6. Run scanner: BTC → ETH → SOL
7. Select best strategy per asset from ranked output