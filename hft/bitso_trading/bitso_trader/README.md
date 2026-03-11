# Bitso Automated Trading System
# Last updated: March 7, 2026

A production-structured, WebSocket-driven automated trading system for Bitso (BTC/USD),
deployed on AWS EC2 t3.medium. Collects synchronized multi-exchange tick data, computes
microstructure features, and executes short-horizon systematic strategies. Currently in
data collection and research phase, targeting live trading by Day 5-6.

---

## Current System Status (as of March 7, 2026)

| Component | Status | Details |
|---|---|---|
| EC2 instance | Running | i-0ee682228d065e3d1, t3.medium, us-east-1 |
| Bitso book recorder | Active | tmux session `recorder`, since 21:15 Mar 7 |
| Three-exchange lead-lag recorder | Active | tmux session `leadlag`, since 21:15 Mar 7 |
| S3 backup | Active | Hourly cron to s3://bitso-orderbook/data/ |
| Watchdog cron | Active | Checks + restarts both sessions every minute |
| Research phase | In progress | Awaiting 36h+ weekday data |

---

## Architecture: Two Parallel Data Pipelines

```
Pipeline 1: Bitso microstructure (EC2 tmux "recorder")
  Bitso WebSocket (orders + trades)
        |
  recorder.py
        |
  book_YYYYMMDD.parquet    (250ms book snapshots, ~15MB/day)
  trades_YYYYMMDD.parquet  (every trade tick)
        |
  research/book_signal_research.py
        |
  IC table: OBI, microprice, depth imbalance signals

Pipeline 2: Cross-exchange lead-lag (EC2 tmux "leadlag")
  BinanceUS WebSocket + Coinbase WebSocket + Bitso WebSocket
        |
  lead_lag_recorder.py
        |
  binance_YYYYMMDD.parquet
  coinbase_YYYYMMDD.parquet
  bitso_YYYYMMDD.parquet
        |
  research/lead_lag_research.py
        |
  IC table: which exchange leads Bitso, by how many seconds
```

---

## Strategy Viability: Current Assessment

Based on 12+ hours of real Bitso BTC/USD data collected March 7, 2026:

| Strategy | Data required | IC measured | Verdict |
|---|---|---|---|
| Lead-lag: Binance.com -> Bitso | Binance + Bitso | **0.31 (5s window)** | Primary candidate |
| Lead-lag: Coinbase -> Bitso | Coinbase + Bitso | Pending weekday data | High probability |
| Lead-lag: BinanceUS -> Bitso | BinanceUS + Bitso | Pending weekday data | Moderate |
| Passive MM with OBI filter | Book snapshots | 0.06 (weekend data) | Retest Monday |
| Microprice momentum | Book snapshots | 0.06 (weekend data) | Retest Monday |
| Large trade event momentum | Trades | Not testable yet | Need weekday data |
| CVD divergence | Trades | Not testable yet | Need 200+ trades/hr |

**Key finding from March 7 research run (1.28 hours, Binance.com from local Mac):**
- IC 0.31 at 5s window: genuine strong edge, rare on any liquid market
- Median Bitso lag behind Binance: 1.25 seconds
- Follow rate: 78.3% of Binance moves > 3 bps
- Net PnL at 3 bps threshold: 2.4 bps after spread cost
- Net PnL at 5 bps threshold: 4.9 bps (small sample, treat with caution)

**Critical spread finding:**
- Mean Bitso spread: 1.45 bps (tight)
- Spread regime matters: book signals have IC 0.08 when spread < 2 bps, IC 0.01 when spread > 2 bps
- All strategies should filter on spread_bps < 2.0 at entry

---

## Folder Structure

```
bitso_trading/                          <- project root on EC2 and locally
|-- recorder.py                         <- Bitso book + trades recorder (RUNNING on EC2)
|-- lead_lag_recorder.py                <- Three-exchange recorder (RUNNING on EC2)
|-- research/
|   |-- book_signal_research.py         <- Microstructure signal IC research
|   |-- lead_lag_research.py            <- Cross-exchange lead-lag IC research
|   `-- signal_research.py              <- Original single-exchange signal research
|-- strategies/
|   |-- lead_lag_signal.py              <- Live lead-lag signal engine (paper/live ready)
|   |-- strategy_a_event_momentum.py    <- Large trade momentum (shelved until weekday data)
|   `-- strategy_b_passive_mm.py        <- Passive market making with OBI filter
|-- bitso_trader/
|   |-- main.py                         <- Full system orchestrator (shadow mode default)
|   |-- requirements.txt
|   |-- config/settings.py
|   |-- core/
|   |   |-- types.py
|   |   |-- feed.py
|   |   |-- orderbook.py
|   |   `-- trade_tape.py
|   |-- features/microstructure.py
|   |-- signals/engine.py
|   |-- risk/engine.py
|   |-- execution/engine.py
|   |-- monitoring/logger.py
|   `-- tests/test_core.py
|-- data/                               <- auto-created, parquet files here
`-- logs/                               <- recorder.log, leadlag.log
```

---

## EC2 Infrastructure

| Parameter | Value |
|---|---|
| Instance ID | i-0ee682228d065e3d1 |
| Instance type | t3.medium (4GB RAM) |
| Region | us-east-1 |
| OS | Amazon Linux 2023 |
| Access | SSM Session Manager only (no SSH) |
| S3 bucket | s3://bitso-orderbook/ |
| IAM role | EC2_TradingBot_Role |

**Connect:**
```bash
aws ssm start-session --target i-0ee682228d065e3d1
# Or if alias is set:
ec2
```

**Once connected:**
```bash
sudo su - ec2-user
cd /home/ec2-user/bitso_trading
```

**Check both recorders:**
```bash
tmux list-sessions
tail -5 logs/recorder.log
tail -5 logs/leadlag.log
```

---

## Data Files in S3

```
s3://bitso-orderbook/
|-- data/
|   |-- book_YYYYMMDD_HHMMSS.parquet     <- Bitso full book snapshots (250ms)
|   |-- trades_YYYYMMDD_HHMMSS.parquet   <- Bitso trade ticks
|   |-- binance_YYYYMMDD_HHMMSS.parquet  <- BinanceUS best bid/ask ticks
|   |-- coinbase_YYYYMMDD_HHMMSS.parquet <- Coinbase best bid/ask ticks
|   `-- bitso_YYYYMMDD_HHMMSS.parquet    <- Bitso best bid/ask ticks (lead-lag feed)
`-- code/                                <- synced from local Mac
```

---

## Running Research Scripts

**Book signal research (on EC2, uses book_*.parquet):**
```bash
cd /home/ec2-user/bitso_trading
python3 research/book_signal_research.py --data-dir ./data --horizon 2 5 10
```

**Lead-lag research (on EC2, uses binance/coinbase/bitso_*.parquet):**
```bash
python3 research/lead_lag_research.py --data-dir ./data
```

**Sync code from Mac to EC2:**
```bash
# On Mac
aws s3 sync ~/path/to/bitso_trading s3://bitso-orderbook/code/ \
  --exclude "data/*" --exclude "__pycache__/*" --exclude "*.pyc" \
  --exclude ".DS_Store" --exclude "*.parquet"

# On EC2
aws s3 sync s3://bitso-orderbook/code/ /home/ec2-user/bitso_trading/ \
  --exclude "data/*"
```

---

## Crontab (active on EC2)

```
# Hourly S3 backup
0 * * * * aws s3 sync /home/ec2-user/bitso_trading/data/ s3://bitso-orderbook/data/ --quiet

# Watchdog: restart recorder if dead (checks every minute)
* * * * * tmux has-session -t recorder 2>/dev/null || tmux new-session -d -s recorder 'cd /home/ec2-user/bitso_trading && python3 recorder.py 2>&1 | tee logs/recorder.log'

# Watchdog: restart leadlag if dead (checks every minute)
* * * * * tmux has-session -t leadlag 2>/dev/null || tmux new-session -d -s leadlag 'cd /home/ec2-user/bitso_trading && python3 lead_lag_recorder.py 2>&1 | tee logs/leadlag.log'
```

---

## Lead-Lag Signal Engine

`strategies/lead_lag_signal.py` is the live signal engine. It maintains rolling price
buffers for both lead exchange and Bitso, fires when divergence exceeds threshold,
manages position entry and exit. Ready for paper and live mode.

Key parameters (tune from lead_lag_research.py output):

| Parameter | Default | Description |
|---|---|---|
| signal_window_sec | 2.0 | Binance/Coinbase lookback window |
| entry_threshold_bps | 4.0 | Min lead exchange move to trigger |
| max_bitso_already_moved_bps | 1.5 | Skip if Bitso already followed |
| hold_sec | 5.0 | Time stop |
| stop_loss_bps | 5.0 | Stop loss from entry mid |
| cooldown_sec | 5.0 | Min seconds between entries |
| mode | shadow | shadow, paper, or live |

---

## Going Live: Checklist

- [ ] Lead-lag IC confirmed > 0.15 on weekday peak session data (Monday)
- [ ] Paper trading 30+ trades with positive mean net PnL
- [ ] Coinbase vs BinanceUS comparison done, primary lead exchange selected
- [ ] Spread filter confirmed: only enter when Bitso spread < 2.0 bps
- [ ] Bitso API key added to EC2 environment (never in git)
- [ ] MAX_POS_BTC=0.01, MAX_DAILY_LOSS=50.0 for first live session
- [ ] Kill switch tested manually
- [ ] EC2 watchdog confirmed working after simulated crash

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Binance.com blocked from US EC2 (HTTP 451) | Using BinanceUS + Coinbase as lead exchanges |
| EC2 OOM crash kills SSM agent | Upgraded to t3.medium (4GB RAM) |
| Process crash loses data | Watchdog cron restarts both sessions within 60 seconds |
| S3 data loss on crash | Hourly cron backup, max 1 hour exposure |
| Weekend data unrepresentative | All IC decisions deferred to Monday weekday session data |
| Adverse selection on aggressive entry | Spread filter (< 2 bps), cooldown, stop loss |
| Bitso spread too wide | spread_bps < 2.0 filter at entry |
| Large trade events absent on weekend | Strategy A deferred to weekday data validation |
| Lead-lag arb already closed by others | Will be visible as IC < 0.10 on weekday data |

---

## Confidence Rating

**7.5 / 10**

The architecture is solid and the initial IC of 0.31 from Binance.com is genuinely strong.
Uncertainty sits on two questions: (1) whether Coinbase or BinanceUS produces comparable IC
to Binance.com from EC2, and (2) whether the IC holds during weekday peak sessions when more
arbitrageurs are active. Both questions answered Monday morning.
