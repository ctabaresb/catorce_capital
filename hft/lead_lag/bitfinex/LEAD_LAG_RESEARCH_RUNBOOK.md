# Lead-Lag Arbitrage Research Runbook
## Bitfinex (Follower) vs BinanceUS + Coinbase (Leaders)

**Last updated:** April 7, 2026  
**Author:** Carlos  
**Strategy:** Cross-exchange lead-lag arbitrage on crypto spot markets

---

## Overview

This strategy exploits the delay between leader exchanges (BinanceUS, Coinbase) and Bitfinex's order book. When both leaders agree on a price move, we enter on Bitfinex before it catches up. Bitfinex has **zero trading fees**, so even small edges are profitable.

### Architecture

```
BinanceUS (leader, ~4 ticks/sec)  ──┐
                                     ├──→  Signal: combined divergence > 7 bps
Coinbase  (leader, ~8 ticks/sec)  ──┘           ↓
                                          Entry on Bitfinex at stale ask
Bitfinex  (follower, book channel)  ──→  Exit after 30-60s (time stop)
```

### Key Parameters

| Parameter | BTC | SOL | Notes |
|-----------|-----|-----|-------|
| Threshold | 7.0 bps | 7.0 bps | Both leaders must agree |
| Spread max | 2.0 bps | 4.0 bps | Tighter = less slippage |
| Hold time | 60s | 60s | Longer hold = more follow-through |
| Stop loss | 15 bps | 15 bps | Rarely fires |
| Cooldown | 120s | 120s | Between trades |
| Fees | 0 bps | 0 bps | Bitfinex zero-fee |

### Data Sources

| Source | S3 Bucket | Description |
|--------|-----------|-------------|
| Bitfinex book BBO | `s3://bitfinex-orderbook/data/*_bitfinex_book_*` | Real-time order book best bid/ask |
| BinanceUS + Coinbase | `s3://bitso-orderbook/data/lead_lag/` | Leader exchange tick data |

---

## Step 1: Download Fresh Data

Run from your Mac (or any machine with AWS CLI configured):

```bash
#!/bin/bash
# run_research_download.sh
# Downloads the latest data from S3 for lead-lag research

set -e

echo "============================================"
echo "Lead-Lag Research: Downloading Data"
echo "============================================"

# Clean slate — avoid mixing old and new data
rm -rf ~/bitfinex_research/bitfinex_book_data
mkdir -p ~/bitfinex_research/bitfinex_book_data
mkdir -p ~/bitfinex_research/leader_data

echo ""
echo ">>> Downloading Bitfinex book-channel data..."
aws s3 sync s3://bitfinex-orderbook/data/ ~/bitfinex_research/bitfinex_book_data/ \
  --exclude "*" --include "*_bitfinex_book_*"

echo ""
echo ">>> Downloading leader data (BinanceUS + Coinbase)..."
aws s3 sync s3://bitso-orderbook/data/lead_lag/ ~/bitfinex_research/leader_data/ \
  --exclude "*" \
  --include "btc_binance_*" --include "btc_coinbase_*" \
  --include "eth_binance_*" --include "eth_coinbase_*" \
  --include "sol_binance_*" --include "sol_coinbase_*" \
  --include "xrp_binance_*" --include "xrp_coinbase_*"

echo ""
echo "============================================"
echo "Verification"
echo "============================================"

echo ""
echo "=== BITFINEX BOOK FILES ==="
echo "Total: $(ls ~/bitfinex_research/bitfinex_book_data/ | wc -l | tr -d ' ') files"

echo ""
echo "=== PER ASSET ==="
echo "BTC: $(ls ~/bitfinex_research/bitfinex_book_data/btc_bitfinex_book_* 2>/dev/null | wc -l | tr -d ' ') files"
echo "ETH: $(ls ~/bitfinex_research/bitfinex_book_data/eth_bitfinex_book_* 2>/dev/null | wc -l | tr -d ' ') files"
echo "SOL: $(ls ~/bitfinex_research/bitfinex_book_data/sol_bitfinex_book_* 2>/dev/null | wc -l | tr -d ' ') files"
echo "XRP: $(ls ~/bitfinex_research/bitfinex_book_data/xrp_bitfinex_book_* 2>/dev/null | wc -l | tr -d ' ') files"

echo ""
echo "=== LEADERS ==="
echo "BTC Binance:  $(ls ~/bitfinex_research/leader_data/btc_binance_* 2>/dev/null | wc -l | tr -d ' ') files"
echo "BTC Coinbase: $(ls ~/bitfinex_research/leader_data/btc_coinbase_* 2>/dev/null | wc -l | tr -d ' ') files"
echo "ETH Binance:  $(ls ~/bitfinex_research/leader_data/eth_binance_* 2>/dev/null | wc -l | tr -d ' ') files"
echo "SOL Binance:  $(ls ~/bitfinex_research/leader_data/sol_binance_* 2>/dev/null | wc -l | tr -d ' ') files"
echo "XRP Binance:  $(ls ~/bitfinex_research/leader_data/xrp_binance_* 2>/dev/null | wc -l | tr -d ' ') files"

echo ""
echo ">>> IMPORTANT: Each asset needs 50+ files for meaningful research."
echo ">>> If any asset shows <50 files, let the recorder run longer."
echo ""
echo "Download complete. Proceed to Step 2."
```

---

## Step 2: Run Research

Run from the `~/bitfinex_research` directory. The script `master_leadlag_bitfinex.py` must be in that directory.

```bash
#!/bin/bash
# run_research_all.sh
# Runs lead-lag research for all 4 assets

set -e

cd ~/bitfinex_research

LEADER_DIR=~/bitfinex_research/leader_data
FOLLOWER_DIR=~/bitfinex_research/bitfinex_book_data
POS_USD=292

echo "============================================"
echo "Lead-Lag Research: Running All Assets"
echo "============================================"
echo "Leader dir:   $LEADER_DIR"
echo "Follower dir: $FOLLOWER_DIR"
echo "Position:     \$$POS_USD"
echo "============================================"

for ASSET in btc eth sol xrp; do
  echo ""
  echo ">>> Running $ASSET research..."
  echo ""
  python3 master_leadlag_bitfinex.py \
    --asset $ASSET \
    --data-dir $LEADER_DIR \
    --follower-dir $FOLLOWER_DIR \
    --pos-usd $POS_USD \
    2>&1 | tee research_${ASSET}_book.log
  echo ""
  echo ">>> $ASSET complete. Log saved to research_${ASSET}_book.log"
  echo "--------------------------------------------"
done

echo ""
echo "============================================"
echo "All research complete. Review logs:"
echo "  research_btc_book.log"
echo "  research_eth_book.log"
echo "  research_sol_book.log"
echo "  research_xrp_book.log"
echo "============================================"
```

---

## Step 3: Interpret Results

### What to look for in each output

1. **IC (Information Coefficient):** > 0.15 means signal has predictive power
2. **Lag median:** > 0.5s means exploitable delay exists
3. **Follow rate:** > 30% means the follower catches up often enough
4. **Realistic avg P&L:** > +1.5 bps = profitable after slippage

### Decision matrix

| Realistic Avg P&L | Win Rate | Action |
|--------------------|----------|--------|
| > +3.0 bps | > 55% | **DEPLOY** — strong edge |
| +1.5 to +3.0 bps | > 45% | **PAPER TRADE** — validate with live book prices |
| 0 to +1.5 bps | any | **MARGINAL** — collect more data or try different params |
| < 0 bps | any | **KILL** — no edge on this asset/config |

### Best configurations (as of April 7, 2026)

| Asset | Config | Avg P&L | Status |
|-------|--------|---------|--------|
| **SOL** | spread<4, 60s hold | +4.24 bps | **Live** |
| BTC | spread<2, 60s hold | +1.67 bps | **Live (tight filter)** |
| ETH | spread<3, 60s hold | +1.78 bps | Needs more data |
| XRP | — | Not tested | Needs data |

---

## Troubleshooting

### "No three-way time overlap"
The Bitfinex data and leader data don't cover the same time period. Make sure the book recorder and leader recorders are running simultaneously on EC2.

### Very few trades in research (<10)
Not enough overlapping data. Let the recorders run for 24-48 more hours and re-download.

### Research shows positive P&L but live trading loses money
Check the entry gap in live logs: `grep "gap=" logs/trader_*.log`. If entry gap > 5 bps consistently, the book is too thin for the position size — reduce POS_USD.

---

## Infrastructure

### EC2 Processes (always running)

| tmux session | Process | Purpose |
|-------------|---------|---------|
| `book_rec` | `bitfinex_book_recorder.py --assets btc eth sol xrp` | Records Bitfinex order book BBO |
| `recorder` | `recorder.py` | Records leader exchange data |
| `recorder_all` | `unified_recorder.py --assets btc eth sol xrp` | Records leader data (all assets) |
| `paper_btc` | `paper_trader_bitfinex.py` | BTC paper trading baseline |
| `trader_btc` | `live_trader_bitfinex.py` (EXEC_MODE=live) | BTC live trading |
| `trader_sol` | `live_trader_bitfinex.py` (EXEC_MODE=live, ASSET=sol) | SOL live trading |

### Auto-restart crons (paper + recorder only, NEVER live trader)

```
* * * * * tmux has-session -t paper_btc  → auto-restart
* * * * * tmux has-session -t book_rec   → auto-restart
*/5 * * * * aws s3 sync ... book data    → S3 backup
```

### S3 Buckets

| Bucket | Contents |
|--------|----------|
| `s3://bitfinex-orderbook/data/` | Bitfinex book-channel parquet files |
| `s3://bitso-orderbook/data/lead_lag/` | Leader exchange (BinanceUS + Coinbase) parquet files |
| `s3://bitfinex-orderbook/code/` | Trading bot source code |
