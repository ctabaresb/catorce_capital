# catorce_capital / hft

Bitso crypto strategy research lab and automated trading system.
Local-first development on Mac. EC2 deployment for live trading.

---

## Project structure

```
hft/
├── config/
│   └── settings.py          Central config — reads from .env / env vars
│
├── core/
│   └── features.py          Microstructure feature computation
│                            batch API (research) + TickFeatureEngine (live)
│
├── research/
│   ├── data_loader.py       Load parquet files, build feature DataFrame
│   ├── strategy_lab.py      Backtest engine: OBI, microprice, TFI + scorecard
│   ├── generate_synthetic_data.py  Synthetic parquet for local testing
│   └── run_research.py      Master entrypoint (synthetic / real / backtest modes)
│
├── execution/
│   └── risk.py              Risk engine: daily loss cap, spread gate, kill switch
│
├── agents/
│   ├── base.py              BaseAgent + MessageBus + message types
│   └── (next sprint)        SignalAgent, RiskAgent, ExecutionAgent, MonitorAgent
│
├── tests/
│   └── test_pipeline.py     pytest suite — runs without real data
│
├── data/                    Parquet files (gitignored, sync from S3)
├── results/                 Scorecard CSVs, feature parquet cache (gitignored)
├── logs/                    Live trader logs (gitignored)
│
├── Makefile                 All commands
├── requirements.txt         Python dependencies
└── .env.example             Credential template
```

---

## Quickstart — Mac, no data needed

```bash
cd /Users/carlos/Documents/GitHub/catorce_capital/hft

# 1. Install dependencies
make install

# 2. Copy env template
make env

# 3. Run full synthetic pipeline — generates data, runs lab, prints scorecard
make synthetic
```

---

## Quickstart — real EC2 data

```bash
# 1. Sync parquet files from S3
make sync-data

# 2. Build feature cache (slow once, fast forever)
make load-only ASSET=btc

# 3. Run backtest
make backtest ASSET=btc

# 4. Run all 3 assets
make all-assets
```

---

## Commands

| Command | What it does |
|---|---|
| `make install` | Install all Python deps |
| `make test` | Run pytest suite (no data needed) |
| `make synthetic` | Synthetic data + full lab run |
| `make real` | Load real parquet + run lab |
| `make load-only` | Build features.parquet cache |
| `make backtest` | Run backtest on cached features |
| `make sync-data` | Sync parquet from S3 |
| `make clean-data-s3` | Delete legacy S3 files (see HANDOFF.md) |

---

## Features computed

| Feature | Formula | Used by |
|---|---|---|
| `mid` | (bid + ask) / 2 | All |
| `spread_bps` | (ask - bid) / mid * 10000 | Cost model, entry gate |
| `microprice` | (bid * ask_sz1 + ask * bid_sz1) / (bid_sz1 + ask_sz1) | MICROPRICE strategy |
| `micro_dev_bps` | (microprice - mid) / mid * 10000 | MICROPRICE signal |
| `obi_1/2/3` | (bid_sz_N - ask_sz_N) / (bid_sz_N + ask_sz_N) | OBI strategy |
| `tfi_10s/30s/60s` | rolling buy_vol / total_vol | TFI strategy |
| `fwd_ret_1/3/5/10s` | (mid[t+h] - mid[t]) / mid[t] * 10000 | Labels |

---

## Strategy verdicts

Strategies are evaluated at IC thresholds of 1/3/5/10s horizons.

| Verdict | Meaning |
|---|---|
| PASS | IC > 0.05, p < 0.05, net_pnl > 0, n_trades >= 30 — worth paper trading |
| MARGINAL | One soft fail, positive PnL, IC > 0.03 — needs more data |
| FAIL | No signal, unprofitable after spread cost, or insufficient trades |

**Important**: PASS on synthetic/backtested data does NOT mean edge in live trading.
Every PASS strategy must:
1. Show stable IC in both first and second half of test set
2. Run as paper trade on EC2 for minimum 48h
3. Meet gate criteria: win_rate > 54%, avg_net > 1.5 bps, n_trades >= 50

---

## Cost model

```
Entry cost:  half-spread  (resting limit at best bid/ask)
Exit cost:   half-spread  (resting limit at other side)
Total:       full spread per round trip
Exchange fee: 0% (Bitso maker)
```

Any strategy with net_pnl < 1.5 bps on BTC is not executable given
typical Bitso BTC spread of 1.5-2.5 bps.

---

## EC2 infrastructure

```
Instance  : i-0ee682228d065e3d1 (t3.medium, us-east-1)
Access    : aws ssm start-session --target i-0ee682228d065e3d1
S3        : s3://bitso-orderbook/
Creds     : AWS SSM /bot/bitso/api_key, /bot/bitso/api_secret
```

Active tmux sessions (DO NOT TOUCH):
- `recorder`      — recorder.py BTC book+trades
- `recorder_all`  — unified_recorder.py BTC+ETH+SOL
- `trader_btc`    — live_trader.py v3.0 LIVE BTC

---

## Roadmap

### Done
- [x] Live BTC lead-lag trader on EC2 (live_trader.py v3.0)
- [x] Multi-asset IC research (BTC/ETH/SOL)
- [x] Data recorders running 24/7
- [x] Strategy research lab (this repo)

### Active
- [ ] Run OBI + microprice + TFI on real Bitso data
- [ ] Paper trade ETH lead-lag
- [ ] Identify PASS strategies for live deployment

### Next
- [ ] SignalAgent: real-time feature stream from TickFeatureEngine
- [ ] RiskAgent: async veto layer
- [ ] ExecutionAgent: async Bitso REST client
- [ ] Composite signal: OBI + TFI regime filter
- [ ] Multi-strategy portfolio: BTC lead-lag + OBI running simultaneously

---

## Execution realism notes

- Bitso EC2 us-east-1 round-trip: ~40-80ms (no co-location)
- Bitso book updates: ~5-10 Hz on BTC (OBI is 100-200ms stale)
- Bitso BTC spread: 1.5-2.5 bps typical, 4-8 bps at night
- Bitso BTC liquidity: top-of-book ~0.1-0.5 BTC at best ask
- No IOC orders (rejected, code 0302) — use plain limit
- No short selling on spot account
- API rate limit: ~60 req/min on order placement
