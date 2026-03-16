#!/usr/bin/env python3
"""
live_trader.py  v4.5.8  — crossed-book threshold calibrated
Lead-lag live trading: Coinbase + BinanceUS -> Bitso

CHANGES v4.5.7 -> v4.5.8  (stop loss → immediate market order)

  Root cause of session-destroying losses identified from live data:
  Trade 17 on 2026-03-14: stop loss triggered at -8 bps, final loss -29 bps.
  The 21 extra bps came from submitting a passive limit on stop loss.
  When BTC is moving fast against us, the bid drops every tick.
  The passive limit sat unfilled for EXIT_CHASE_SEC (8s) while bid dropped.
  8 seconds in a fast market = 20+ bps of additional loss.
  This one trade cost .16 and turned a +.04 session into -.12.

  Fix: split stop loss vs time stop exit logic.
  STOP LOSS → immediate market order (costs ~1.35 bps, guaranteed fill).
  TIME STOP → passive limit first (free if fills), market fallback after 8s.

  Stop losses need instant execution — saving 1.35 bps is never worth
  risking 20+ bps of overshoot during a fast adverse move.
  Time stops fire when market is stable — passive limit can collect spread.

CHANGES v4.5.6 -> v4.5.7  (PnL zero + orphan after partial exit fix)

  Fix 1: PnL recording zero on market order exits
    Market orders on Bitso return price=0 in the order payload.
    The poller was falling back to risk.entry_mid as exit price,
    making every trade show 0.000 bps PnL regardless of real profit.
    Fix: fetch actual weighted fill price from GET /v3/order_trades/?oid=
    Applied to both: status=completed path and 0303/0304 not-found path.
    Fallback: current mid if trades API also unavailable.

  Fix 2: Orphan BTC after passive limit partial fill + cancel
    When passive limit partially fills across multiple poll cycles,
    position_asset was reduced by the increment seen in ONE poll cycle.
    After cancelling, subsequent partial fill settlements created orphans.
    Fix: after cancelling a partial limit, fetch actual BTC balance
    and set position_asset to the real amount in account.
    If balance=0 after cancel, position is fully exited — reset directly.

CHANGES v4.5.5 -> v4.5.6  (3 bug fixes found in deep audit)

  Fix 1: _cancel_order function signature restored (was stripped in v4.5.5)
    The async def _cancel_order line was accidentally removed when cleaning
    up the force close block. Function body existed but was unreachable.
    Result: NameError at runtime. Fixed by restoring the def line.

  Fix 2: Market buy preflight placeholder removed
    _submit_market_order had dead code: if amount_asset * 0 < 0 (always False).
    USD balance check for buy orders was never executed.
    Fixed: replaced with a clean USD > 1.0 guard. Bitso handles actual size
    validation — our preflight is now a simple sanity check.

  Fix 3: Partial market exit resubmission
    After market exit partially fills, poller cancels remainder and clears
    exit_oid. Handle_exit attempt 2 now detects exit_oid=empty and
    resubmits a market order for remaining position_asset.
    Previously: remaining BTC sat until reconciler caught it as orphan.
    Now: immediate market resubmission for any remaining size.

CHANGES v4.5.4 -> v4.5.5  (market orders — eliminate fill failures permanently)

  Root cause of 62%+ entry unfill rate:
  Limit orders at ask+30 ticks sat below the moving ask and never filled.
  Every unfill triggered cancel+retry cycles, partial fill cascades, open
  buy orders reserving USD, and orphan emergency sells at 1% slippage.

  Entry fix: market order replaces limit order.
  _submit_market_order() sends type=market, major=size, no price field.
  Fills instantly at best available ask. Cost: ~half spread (~1.35 bps mean).
  This is CHEAPER than ask+30 ticks (was 1.35 + 0.43 = 1.78 bps) and
  guarantees 100% fill rate vs 38% with limit orders.
  Eliminates: ENTRY UNFILLED, open buy orders, entry partial fill cascades.

  Exit fix: passive limit at bid + market fallback.
  Attempt 1: limit sell at current bid — collects spread, zero cost if fills.
  If unfilled after EXIT_CHASE_SEC: cancel and submit market order.
  Market exit costs ~1.35 bps but guarantees fill.
  Eliminates: force close at -0.5% (57 loss), 3-attempt chaser,
              exit partial fill cascades, reconciler nuclear exits.

  Net cost per round trip:
    Best case (passive exit fills): ~1.35 bps (entry only)
    Worst case (market entry + market exit): ~2.70 bps
    Previous worst case (force close): 50+ bps

  Net expected PnL at 5bps signal: 3.53 - 1.35 = +2.18 bps (passive exit)
                                    3.53 - 2.70 = +0.83 bps (market exit)

CHANGES v4.5.3 -> v4.5.4  (diff-orders channel — root cause of stale feed)

  Root cause of every stale feed reconnect since v1.0 identified:
  We subscribed to the 'orders' channel which only fires when the
  top-20 bid/ask snapshot changes. If the best bid stays flat while
  second-level orders update constantly (normal Bitso behavior),
  the channel sends ZERO messages. Stale timer fires. Reconnect.
  Repeat forever.

  Fix: switch to 'diff-orders' channel.
  diff-orders fires on EVERY book change at any level — millisecond
  rate when the book is active. Payload is a list of incremental diffs:
    r = price, t = side (0=bid / 1=ask), a = amount (0 = removed)
  We build the same bids/asks dict incrementally, same as before.
  Stale reconnect problem eliminated completely.

  Also retain 'trades' subscription as secondary heartbeat for
  periods when diff-orders is slow (e.g. between order changes).

CHANGES v4.5.2 -> v4.5.3  (overnight trading gate + feed quality guard)

  Root cause of $15.66 loss in v4.5.2 overnight session:
  Bitso BTC/USD has near-zero book depth from ~1am-8am Mexico City.
  The feed reconnects every 30 seconds (dead market signal).
  Entries fire on valid cross-exchange signals but Bitso has no
  counterparties. Exit limit sells fail all 3 attempts → force close
  at 0.5% below bid = $357 slippage per trade. Happened 3 times.

  Fix 1: Trading hours gate in handle_entry
    Only enter between 15:00-01:59 UTC (9am-7:59pm Mexico City).
    Configurable via TRADING_HOURS_UTC env var (comma-separated hours).
    During active hours, Bitso book depth is sufficient for 0.020 BTC exits.
    Overnight book depth: ~$40/level. Daytime depth: ~$800+/level.

  Fix 2: Feed quality guard in bitso_feed entry check
    Calls state.feed_quality_ok() before every entry evaluation.
    Returns False if ≥3 Bitso reconnects in the last 5 minutes.
    Dead market signature: reconnect every 30s = 10 reconnects/5min.
    Active market signature: 0-1 reconnects per hour.
    This catches edge cases where book is thin outside normal dead hours.
    Reconnects are tracked via state.record_reconnect() called on connect.

CHANGES v4.5.1 -> v4.5.2  (cancel retry — prevent open buy orders)

  Root cause of ,174 balance drain in v4.5.1 session:
  _cancel_unfilled_entry called _cancel_order once and continued regardless
  of whether the cancel succeeded. If cancel failed (network timeout, Bitso
  API error), the open buy order stayed on Bitso reserving USD and could
  fill at a stale price later.

  Fix: cancel with retry — up to 3 attempts with 0.5s between each.
  If cancel confirmed: proceed with partial fill emergency sell as before.
  If cancel not confirmed after 3 attempts: skip the emergency sell,
  log an ERROR, fall through to state reset. Reconciler catches any BTC
  that arrives as an orphan. Startup cancels any remaining open order
  on next session start.

CHANGES v4.5 -> v4.5.1  (orphan minimization + research-optimized params)

  FOUR FIXES targeting the remaining orphan and negative-trade sources:

  Fix 1: POST_RESET_COOLDOWN raised 8s → 20s
    The 8s default was chosen when WebSocket fill detection was expected
    to handle most exits. In practice, REST polling is the primary path.
    Bitso balance API settlement can take 5-30s. 20s covers the typical
    settlement window without killing frequency the way 45s did.

  Fix 2: SILENT FILL verification before reset (biggest orphan fix)
    Root cause of orphan cascades: Bitso shows balance=0 transiently
    after a partial exit fill (settlement lag). Reconciler saw 0 balance,
    called _reset_position, cleared the position. Minutes later the
    remaining BTC from the entry settled into the account as orphans.
    Fix: before accepting a SILENT FILL, call GET /v3/orders/{exit_oid}.
    If status is 'partially filled' or 'open', skip the reset this cycle.
    Only reset when order is 'completed', 'cancelled', or not found (0303/0304).

  Fix 3: Poller tracks partial exit fills — updates position_asset
    When exit order partially fills, the poller now subtracts the sold
    amount from position_asset before cancelling the remainder.
    This ensures the next exit order and PnL calculation use the correct
    remaining size, not the original entry size.

  Fix 4: bt_ret quality filter tightened 80% → 40% of threshold
    Research (module 11, 101 hours BTC data): entering when Bitso has
    already moved >40% of the threshold yields 1.4 bps less per trade.
    At 6bps threshold: now blocks entry if bt_ret > 2.4 bps (was 4.8 bps).
    Removes the worst late-entry signals. Reduces signal count ~15% but
    improves average PnL per filled trade.

  PARAMETER RESEARCH FINDINGS (deploy via env vars, not in code):
    ENTRY_THRESHOLD_BPS = 6.0   (was 8-10, research shows 6bps maximizes $/day)
    ENTRY_SLIPPAGE_TICKS:
      BTC: 30 ticks (/bin/sh.30, 0.43 bps cost) — good fill rate, negligible cost
      ETH: 3 ticks  (/bin/sh.03, 0.15 bps cost) — ETH tick cost is 3.5x BTC, keep low
    SPREAD_MAX_BPS = 4.0  (was 5.0, protects quality at lower threshold)
    MAX_POS_ASSET BTC: 0.020 (was 0.012, 67% larger position)

CHANGES v4.4 -> v4.5  (entry partial fill fix — orphan root cause)

  Root cause of persistent orphans identified from v4.4 live session:
  Bitso frequently partially fills ENTRY orders (buy 0.012 BTC, Bitso
  fills 0.003 BTC immediately, leaves 0.009 BTC as open order).
  When handle_exit fires at attempt 0 and sees BTC=0, it called
  _cancel_unfilled_entry which cancelled the open 0.009 BTC portion.
  The 0.003 BTC already filled then appeared in account as an orphan.
  Emergency sell fired at 1% below bid. Repeated 5 times in 1.3 hours.

  Fix: _cancel_unfilled_entry now checks GET /v3/orders/{oid} BEFORE
  cancelling. If status=partially filled:
    1. Cancel the unfilled remainder
    2. Wait 1s for Bitso to release the BTC
    3. Immediately sell the filled portion at bid*(1-FORCE_CLOSE_SLIPPAGE)
    4. Record a trade via _reset_position (entry_partial_fill reason)
  If fully unfilled: cancel and clear state as before (no change).

  This eliminates the most frequent orphan source in v4.4.

CHANGES v4.3 -> v4.4  (partial fill + stale order fixes)

  Root cause of 07 balance drain:
  Bitso sell orders can be partially filled — some BTC sold, rest stays open.
  Poller only checked status=completed. partially filled was silently ignored.
  The unfilled remainder kept the USD reserved indefinitely. On session restart
  the stale order from the prior session also stayed open undetected.

  Fix 1: Poller handles partially filled status.
    If exit order is partially filled and has been open > EXIT_CHASE_SEC seconds,
    cancel the remainder. The exit chaser will resubmit for the remaining size
    at the current bid on the next Bitso tick. No position reset — just cancels
    the stale partial so the chaser can work properly.

  Fix 2: startup_checks cancels all open orders from prior sessions.
    On every startup, fetches GET /v3/open_orders/?book=btc_usd and cancels
    any open or partially filled orders before trading begins. Prevents stale
    orders from prior sessions from silently draining available USD balance.

CHANGES v4.2 -> v4.3  (poller acts directly + entry fill rate fix)

  Fix 1: Poller calls _reset_position() directly on fill detection.
    Previous: poller set ws_fill_detected=True then waited for a Bitso
    book tick to fire handle_exit. During stale BT periods (30s), the
    flag sat unprocessed and reconciler caught the fill at 15s instead.
    Fix: poller now calls _reset_position(risk, pnl, fill_price, hold_sec,
    poller_fill) immediately. No Bitso tick required. Fill recorded in <3s.

  Fix 2: ENTRY_SLIPPAGE_TICKS raised to 10 in deploy command.
    38% entry fill rate: ask moves + during 300ms REST latency on
    an 8bps signal move. 3 ticks (/bin/sh.03) buffer was insufficient.
    10 ticks (/bin/sh.10) = 0.0148 bps cost on BTC — negligible vs 8bps edge.
    Expected fill rate improvement: 38% -> 65-75%.

CHANGES v4.1 -> v4.2  (payload list bug fixed)
  GET /v3/orders/{oid} returns payload as a LIST not a dict.
  order.get("status") was silently failing on a list object.
  Fix: unwrap payload[0] before accessing status field.
  One line. This is why the poller never detected any fills in v4.1.

CHANGES v4.0 -> v4.1  (order polling replaces broken WS auth)
  Bitso user_trades WebSocket returns {error: invalid message} for all
  tested auth formats — channel auth is undocumented. Replaced with
  fast REST order polling: GET /v3/orders/{oid} every 2 seconds when
  in position. Status=completed is unambiguous fill confirmation with
  no balance API settlement lag. Achieves 2-3s detection vs 15-30s
  reconciler. Sets ws_fill_detected=True so handle_exit fires on next
  Bitso tick. Eliminates orphan root cause completely.

CHANGES v3.11 -> v4.0  (WebSocket fill detection — architectural fix)

  THE ROOT CAUSE OF ALL PERFORMANCE PROBLEMS:
  Fills were detected by REST polling (reconciler every 15s), not by
  listening to Bitso in real time. This caused:
    - 15-30s exit detection lag on every trade
    - 36s avg hold vs 20s target (extra 16s per trade = 0.5-1.5 bps lost)
    - 45s cooldown needed to prevent orphans = trade frequency destroyed
    - 3.8 trades/hr live vs 75/hr research prediction

  FIX: user_trades_feed coroutine (new in v4.0)
  Subscribes to Bitso authenticated WebSocket channel type=user_trades.
  Fires in real time when any order fills. Sets risk.ws_fill_detected=True.
  handle_exit checks this flag on every Bitso book tick (<1s response).
  Result:
    - Exit detection latency: 15-30s -> <1s
    - Average hold: 36s -> ~20-22s (target)
    - POST_RESET_COOLDOWN reduced: 45s -> 8s (WS handles fast path)
    - Trade frequency: should recover toward research prediction
    - Orphans: nearly impossible when exits detected in <1s

  Also added ws_fill_price for accurate P&L using actual fill price
  rather than mid price at detection time.

CHANGES v3.10 -> v3.11  (cancel+fill race protection)

  Remaining orphan source identified: _cancel_unfilled_entry race.
  When entry order is cancelled at time_stop, Bitso can fill the order
  AND acknowledge the cancel simultaneously. BTC appears in account but
  system thinks entry never filled (ORPHAN). Cooldown was only 8s here
  vs 45s in reconciler resets — insufficient for Bitso settlement.
  Fix: apply POST_RESET_COOLDOWN (45s) in _cancel_unfilled_entry.
  All three orphan-producing paths now have 45s cooldown:
    Case A: reconciler orphan sell
    Case B: reconciler silent fill reset
    Case C: _cancel_unfilled_entry (new)

CHANGES v3.9 -> v3.10  (two-speed stale threshold)

  ROOT CAUSE: stale feed while IN_POSITION swallows stop losses.
  Stop loss set at -8 bps. Feed goes stale 30s. Price falls to -16 bps
  during blind window. handle_exit cannot fire without valid ticks.
  When feed reconnects, first tick sees -16.9 bps — double the intended
  maximum loss. This cannot be fixed by tightening stop loss bps because
  the loss happens entirely within the blind window.

  Fix: two-speed stale threshold in bitso_feed.
    IN_POSITION:  stale_threshold = 8s  (reconnect fast, protect open trades)
    FLAT:         stale_threshold = STALE_RECONNECT_SEC (default 30s)

  When holding a position, an 8s blind window is the maximum tolerated.
  At BTC volatility of ~0.5-1%/min, 8s exposure adds ~0.07-0.13 bps of
  extra stop loss slippage — acceptable. 30s adds 2.5 bps — not acceptable.

  When flat, 30s is correct. No position at risk. Avoids the excessive
  reconnect frequency that caused orphan events in earlier versions.

CHANGES v3.8 -> v3.9  (orphan race condition fix)

  ROOT CAUSE IDENTIFIED from 16h live trading diagnostic:
  7 orphan events + 1 nuclear exit cost estimated $62 in forced slippage
  against a gross strategy P&L of $62/day — breaking even on a
  profitable strategy.

  The Bitso balance API shows 0 for 2-10 seconds after a fill
  (settlement lag). The reconciler fires during this window, sees
  asset_bal=0, performs a SILENT FILL reset, sets cooldown=8s.
  Signal fires 8s later, orphan guard REST call also returns 0.
  New entry placed. Original BTC from prior fill appears. Orphan.

  Fix 1: STALE_RECONNECT_SEC default raised from 15s to 30s.
    15s was causing reconnects every 15-20s continuously, keeping
    BT above 5s threshold, blocking handle_exit, routing 88% of
    exits through the reconciler (15-30s delayed detection).
    30s reduces reconnect frequency while keeping the safety net.

  Fix 2: POST_RESET_COOLDOWN = 45s after every reconciler reset.
    After Case A (orphan sell) and Case B (silent fill reset),
    last_exit_ts is set to time.time() + 37s (45s - 8s cooldown).
    This gives the Bitso balance API full time to settle before
    any new entry is allowed. Eliminates the race condition.

  Fix 3: SOL terminated. 27 live trades at -2.168 bps avg confirms
    structural unprofitability. Tick cost of 1.17 bps/tick at $85
    combined with 100% reconciler exits creates unavoidable losses.

  Audit findings from v3.8 remain valid. No other changes.

CHANGES v3.7 -> v3.8  (full state machine audit)
  Fix 1: PnL size accounting mismatch.
    _submit_order now returns actual submitted size (after preflight
    adjustment). handle_entry uses actual_size not MAX_POS_ASSET.
    When USD is insufficient and preflight reduces size, PnL was being
    calculated on a larger notional than actually traded.

  Fix 2: Rate limiter moved inside attempt 0, after all early returns.
    Previously the 2s rate limiter fired at the top of handle_exit and
    blocked attempt 0 even during normal operation. Now it only fires
    immediately before an actual API call, not on deferred exits or
    trigger checks. Exit latency at attempt 0 reduced by up to 2s.

  Audit findings (no code change needed):
    Scenario 1: Normal fill path - OK
    Scenario 3: Stale feed hold extension - FIXED in v3.7 (15s default)
    Scenario 4: Multiple stale events worst case 80s hold - ACCEPTABLE
    Scenario 5: Permanent feed failure - reconciler backstop OK
    Scenario 6: REST call rate - 4-5/min sustained, safe under 60/min limit
    Scenario 9: feeds_healthy blocks entries during stale - CORRECT behavior
    Scenario 10: Stale bid in stop loss calc (15s max) - ACCEPTABLE RISK

CHANGES v3.6.1 -> v3.7
  Fix 1: EXIT deferred trap removed.
    Floor guard now has a hard max deferral of 2x HOLD_SEC.
    Previously bid < floor caused infinite deferral on every tick,
    holding positions 3-5x longer than intended and accumulating
    large losses. After 2x HOLD_SEC the system exits regardless.
    Log level changed to DEBUG to prevent spam on every tick.

  Fix 2: STALE_RECONNECT_SEC default lowered from 60s to 15s.
    60s stale window was causing exits to be detected by reconciler
    (30s cycle) instead of handle_exit, producing 60-90s actual holds
    vs 20s intended. At 15s stale threshold, reconnect happens faster,
    handle_exit fires on real ticks, and hold times match design.

CHANGES v3.6 -> v3.6.1
  STALE_RECONNECT_SEC raised from 30s to 60s default and moved to top-level
  env-configurable config. 30s was too aggressive for BTC on Bitso at night —
  the book genuinely has 30s gaps in valid ticks during low-volume periods.
  60s catches real feed failures while not firing on quiet market conditions.

CHANGES v3.5 -> v3.6
  Moved stale guard BEFORE the crossed-book continue.
  In v3.5 the stale guard was placed after the crossed-book skip.
  When all ticks are crossed the continue fires first on every message
  and the stale guard is never reached — bitso.age() grows indefinitely.
  Fix: check staleness on every incoming message before any skip logic.

CHANGES v3.4 -> v3.5
  Removed consecutive-crossed-tick counter entirely.
  v3.1 through v3.4 all used some form of this counter and all caused
  spurious reconnects. Bitso interleaves crossed ticks with valid ticks
  at high frequency during normal operation — no counter threshold works.
  Stale guard (30s no valid tick) is the only mechanism needed and is
  sufficient for all real failure modes.

CHANGES v3.3 -> v3.4
  CROSSED_RECONNECT_THRESH raised from 10 to 300.
  Bitso sends ~1 crossed message per 2s during normal operation.
  Threshold of 10 fired a reconnect every ~20s, same symptom as v3.1.
  BT=0.0s in v3.3 logs confirmed the feed was healthy between reconnects,
  so the reconnects were spurious. The stale guard (30s no valid tick)
  is the correct primary defence. 300 ticks is a last-resort backstop.

CHANGES v3.2 -> v3.3
  Root cause: crossed-book handler was wrong in v3.0, v3.1, and v3.2.
  v3.3 handles all three cases in one implementation:

  Case 1 — Transient single-tick crossing (1-9 consecutive):
    continue — skip tick, keep connection and dict state intact.

  Case 2 — Persistent crossing (10+ consecutive crossed ticks):
    break — reconnect for fresh snapshot. 10 ticks ~= 160ms at ETH
    cadence, fires fast when genuinely stuck, no false-positives on
    normal transient crossings.

  Case 3 — Stale feed despite live connection (no valid tick for 30s):
    break — reconnect. Catches the v3.2 failure mode where all startup
    ticks were crossed, consecutive_crossed never hit the threshold, and
    bitso.age() grew to 30+ minutes causing permanent zero-trade blindness.
    Guard activates only after 15s of connection age to avoid false
    positives during initial snapshot population.

  v3.0: clear dicts + continue -> lost incremental baseline
  v3.1: break on every crossing -> 2-3s reconnect blindness every ~15s
  v3.2: continue always -> permanent blindness when startup ticks all crossed
  v3.3: three-case handler, correct for all scenarios

CHANGES v3.0 -> v3.1
  Fix 1: _NO_ASSET_ERRORS expanded for ETH and SOL error codes.
  Fix 2: Crossed book (superseded by v3.3).
  Fix 3: Silent fill reason uses actual_reason not hardcoded time_stop.

SUPPORTED ASSETS:  btc_usd  eth_usd  sol_usd   (set via BITSO_BOOK env var)

MODES
  paper  simulate fills, zero real orders (default, safe)
  live   submit real limit orders to Bitso REST API

USAGE — single asset
  EXEC_MODE=live BITSO_BOOK=btc_usd python3 live_trader.py

USAGE — multi-asset (one process per asset)
  tmux new -d -s trader_btc 'EXEC_MODE=live BITSO_BOOK=btc_usd  MAX_POS_ASSET=0.001 python3 live_trader.py'
  tmux new -d -s trader_eth 'EXEC_MODE=live BITSO_BOOK=eth_usd  MAX_POS_ASSET=0.026 python3 live_trader.py'
  tmux new -d -s trader_sol 'EXEC_MODE=live BITSO_BOOK=sol_usd  MAX_POS_ASSET=0.37  python3 live_trader.py'

MULTI-ASSET CAPITAL ALLOCATION
  One process per asset is the correct architecture. Reasons:
  - Separate kill switches, logs, and PnL trackers per asset.
  - An ETH crash cannot kill the BTC session.
  - Can restart one asset without touching the others.

  Balance is the SINGLE source of truth (Bitso REST API).
  The USD preflight check prevents over-commitment automatically.

  Sizing rule: MAX_POS_ASSET * price * num_assets <= 80% of account.
  Example with $200 account, 3 assets ~$53 each:
    BTC: MAX_POS_ASSET=0.00080   (~$53)
    ETH: MAX_POS_ASSET=0.026     (~$53)
    SOL: MAX_POS_ASSET=0.37      (~$53)
  3 * $53 = $159 = 79.5% utilisation. If all three enter simultaneously,
  the third preflight sees only $94 left, adjusts size down automatically.

EXIT CHASER — DEFINITIVE FIX (v3.0)
  Root cause of all previous orphan-asset incidents:
    force_px = bid - 5 * $0.01 = bid - $0.05
  In fast markets the price outran $0.05 in the 200ms between reading
  the bid and Bitso receiving the order. The limit landed ABOVE the new
  best bid and sat on the book unfilled. Then _reset_position() was called
  optimistically, system thought flat, new BUY fired, USD drained.

  TWO mechanisms work together to prevent this permanently:

  1. FORCE CLOSE PRICE: bid * (1 - FORCE_CLOSE_SLIPPAGE)
     Default FORCE_CLOSE_SLIPPAGE = 0.005 = 0.5% below bid.
     At $66,600: force_px = $66,267. This WILL sweep the book.

  2. RECONCILER TASK: runs every RECONCILE_SEC (default 30s).
     Checks actual Bitso balance vs internal RiskState.
     Three failure modes handled deterministically:
       A. Orphan: exchange has asset, internal=FLAT      -> emergency sell
       B. Silent fill: IN_POSITION but no asset          -> reset + record
       C. Stuck exit: IN_POS + asset + attempt >= 4      -> nuclear exit (2%)

  KEY INVARIANT:
  After force close submits (attempt 3 -> 4), handle_exit returns immediately.
  _reset_position() is NEVER called from handle_exit after that point.
  The reconciler is the ONLY thing that calls _reset_position() from attempt 4+.
  This means the system cannot optimistically reset while asset sits in account.

RISK CONTROLS
  MAX_DAILY_LOSS_USD     hard kill switch                  default 50.0
  MAX_POS_ASSET          order size in base asset units    default 0.001
  ENTRY_THRESHOLD_BPS    min lead divergence to enter      default 5.0
  SIGNAL_WINDOW_SEC      lookback window                   default 5.0
  HOLD_SEC               time stop                         default 8.0
  EXIT_CHASE_SEC         seconds per chase step            default 8.0
  STOP_LOSS_BPS          per-trade stop                    default 5.0
  COOLDOWN_SEC           min seconds between entries       default 8.0
  SPREAD_MAX_BPS         skip entry if spread too wide     default 3.0
  COMBINED_SIGNAL        require both exchanges agree      default true
  FORCE_CLOSE_SLIPPAGE   fraction below bid for force sell default 0.005
  RECONCILE_SEC          balance reconcile interval        default 30.0
  STALE_RECONNECT_SEC    seconds no valid Bitso tick before reconnect default 60.0

CREDENTIALS (checked in order)
  1. BITSO_API_KEY / BITSO_API_SECRET env vars
  2. AWS SSM: /bot/bitso/api_key  /bot/bitso/api_secret
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import websockets

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

REGION        = os.getenv("AWS_REGION", "us-east-1")
EXEC_MODE     = os.environ.get("EXEC_MODE", "paper")
BITSO_API_URL = "https://api.bitso.com"

BITSO_BOOK = os.environ.get("BITSO_BOOK", "btc_usd")
ASSET      = BITSO_BOOK.split("_")[0].lower()

_BINANCE_SYMS  = {"btc": "btcusdt",  "eth": "ethusdt",  "sol": "solusdt"}
_COINBASE_SYMS = {"btc": "BTC-USD",  "eth": "ETH-USD",  "sol": "SOL-USD"}
BINANCE_SYMBOL  = _BINANCE_SYMS.get(ASSET,  f"{ASSET}usdt")
COINBASE_SYMBOL = _COINBASE_SYMS.get(ASSET, f"{ASSET.upper()}-USD")

MAX_DAILY_LOSS_USD   = float(os.environ.get("MAX_DAILY_LOSS_USD",   "50.0"))
MAX_POS_ASSET        = float(os.environ.get("MAX_POS_ASSET",        "0.001"))
ENTRY_THRESHOLD_BPS  = float(os.environ.get("ENTRY_THRESHOLD_BPS",  "5.0"))
SIGNAL_WINDOW_SEC    = float(os.environ.get("SIGNAL_WINDOW_SEC",    "5.0"))
HOLD_SEC             = float(os.environ.get("HOLD_SEC",             "8.0"))
STOP_LOSS_BPS        = float(os.environ.get("STOP_LOSS_BPS",        "5.0"))
COOLDOWN_SEC         = float(os.environ.get("COOLDOWN_SEC",         "8.0"))
COMBINED_SIGNAL      = os.environ.get("COMBINED_SIGNAL", "true").lower() == "true"
SPREAD_MAX_BPS       = float(os.environ.get("SPREAD_MAX_BPS",       "3.0"))
EXIT_CHASE_SEC       = float(os.environ.get("EXIT_CHASE_SEC",       "8.0"))
FORCE_CLOSE_SLIPPAGE = float(os.environ.get("FORCE_CLOSE_SLIPPAGE", "0.005"))
RECONCILE_SEC        = float(os.environ.get("RECONCILE_SEC",        "30.0"))
ENTRY_SLIPPAGE_TICKS = int(os.environ.get("ENTRY_SLIPPAGE_TICKS",   "2"))    # ticks above ask on buy entry
STALE_RECONNECT_SEC  = float(os.environ.get("STALE_RECONNECT_SEC",  "30.0")) # seconds of no valid Bitso tick before reconnect
POST_RESET_COOLDOWN  = float(os.environ.get("POST_RESET_COOLDOWN",  "20.0")) # seconds to block entries after any reset (Bitso balance settlement)

_MIN_SIZES     = {"btc": 0.00001, "eth": 0.0001, "sol": 0.001}
MIN_TRADE_SIZE = _MIN_SIZES.get(ASSET, 0.00001)

ENABLE_TELEGRAM       = os.environ.get("ENABLE_TELEGRAM", "1").strip() == "1"
TELEGRAM_TOKEN_PARAM  = os.environ.get("TELEGRAM_TOKEN_PARAM", "/bot/telegram/token")
TELEGRAM_CHAT_PARAM   = os.environ.get("TELEGRAM_CHAT_PARAM",  "/bot/telegram/chat_id")
TELEGRAM_REPORT_HOURS = float(os.environ.get("TELEGRAM_REPORT_HOURS", "1.0"))

_BITSO_API_KEY    = os.environ.get("BITSO_API_KEY",    "")
_BITSO_API_SECRET = os.environ.get("BITSO_API_SECRET", "")

LOG_DIR    = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADE_LOG  = LOG_DIR / f"trades_{ASSET}_{SESSION_TS}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"live_{ASSET}_{SESSION_TS}.log"),
    ],
)
log = logging.getLogger(__name__)

# All error strings meaning "no asset available to sell"
# Must cover all traded assets — ETH/SOL have their own Bitso error codes.
# Missing codes cause exit chaser to stay IN_POSITION after a silent fill.
_NO_ASSET_ERRORS = frozenset({
    "no_asset_to_sell",
    "balance_too_low",
    # BTC
    "no_btc_to_sell",
    "insufficient_btc",
    # ETH
    "no_eth_to_sell",
    "insufficient_eth",
    # SOL
    "no_sol_to_sell",
    "insufficient_sol",
})


# ──────────────────────────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────────────────────────

_tg_tok: Optional[str] = None
_tg_cid: Optional[str] = None
_tg_creds_loaded        = False


def _load_telegram_creds() -> Tuple[Optional[str], Optional[str]]:
    global _tg_tok, _tg_cid, _tg_creds_loaded
    if _tg_creds_loaded:
        return _tg_tok, _tg_cid
    if not ENABLE_TELEGRAM:
        _tg_creds_loaded = True
        return None, None
    try:
        import boto3
        ssm     = boto3.client("ssm", region_name=REGION)
        _tg_tok = ssm.get_parameter(Name=TELEGRAM_TOKEN_PARAM, WithDecryption=True)["Parameter"]["Value"]
        _tg_cid = ssm.get_parameter(Name=TELEGRAM_CHAT_PARAM,  WithDecryption=True)["Parameter"]["Value"]
        log.info("[Telegram] Credentials loaded from SSM.")
    except Exception as e:
        log.warning("[Telegram] SSM load failed: %s. Telegram disabled.", e)
    _tg_creds_loaded = True
    return _tg_tok, _tg_cid


def _send_telegram_sync(text: str):
    tok, cid = _load_telegram_creds()
    if not tok or not cid:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id": cid, "text": text},
            timeout=10,
        )
    except Exception as e:
        log.warning("[Telegram] Send failed: %s", e)


async def tg(text: str):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _send_telegram_sync, text)


# ──────────────────────────────────────────────────────────────────
# PnL TRACKER
# ──────────────────────────────────────────────────────────────────

class PnLTracker:
    def __init__(self):
        self._trades:       list  = []
        self.daily_pnl_usd: float = 0.0

    def record(
        self,
        direction:  str,
        entry_mid:  float,
        exit_mid:   float,
        size_asset: float,
        hold_sec:   float,
        reason:     str,
        entry_oid:  str = "",
        exit_oid:   str = "",
    ) -> Tuple[float, float]:
        if direction == "buy":
            pnl_bps = (exit_mid - entry_mid) / entry_mid * 10_000
        else:
            pnl_bps = (entry_mid - exit_mid) / entry_mid * 10_000

        pnl_usd             = pnl_bps / 10_000 * entry_mid * size_asset
        self.daily_pnl_usd += pnl_usd

        trade = {
            "ts":            time.time(),
            "asset":         ASSET,
            "direction":     direction,
            "entry_mid":     round(entry_mid,  2),
            "exit_mid":      round(exit_mid,   2),
            "size_asset":    size_asset,
            "pnl_bps":       round(pnl_bps,   4),
            "pnl_usd":       round(pnl_usd,   6),
            "hold_sec":      round(hold_sec,   2),
            "reason":        reason,
            "entry_oid":     entry_oid,
            "exit_oid":      exit_oid,
            "daily_pnl_usd": round(self.daily_pnl_usd, 4),
        }
        self._trades.append(trade)
        with open(TRADE_LOG, "a") as fh:
            fh.write(json.dumps(trade) + "\n")
        return pnl_bps, pnl_usd

    @property
    def n_trades(self) -> int:   return len(self._trades)
    @property
    def n_wins(self) -> int:     return sum(1 for t in self._trades if t["pnl_bps"] > 0)
    @property
    def win_rate(self) -> float: return self.n_wins / max(self.n_trades, 1)
    @property
    def avg_pnl_bps(self) -> float:
        return sum(t["pnl_bps"] for t in self._trades) / max(self.n_trades, 1)
    @property
    def best_trade_bps(self) -> float:
        return max((t["pnl_bps"] for t in self._trades), default=0.0)
    @property
    def worst_trade_bps(self) -> float:
        return min((t["pnl_bps"] for t in self._trades), default=0.0)
    @property
    def n_stop_losses(self) -> int:
        return sum(1 for t in self._trades if t["reason"] == "stop_loss")
    @property
    def n_time_stops(self) -> int:
        return sum(1 for t in self._trades if t["reason"] == "time_stop")

    def summary_text(self, mode: str, runtime_hr: float) -> str:
        trades_hr = self.n_trades / max(runtime_hr, 0.01)
        return "\n".join([
            f"Bitso Lead-Lag v4.5.8 [{mode.upper()}] {ASSET.upper()}",
            f"Runtime:      {runtime_hr:.1f}h",
            f"Trades:       {self.n_trades}  ({trades_hr:.1f}/hr)",
            f"Win rate:     {self.win_rate*100:.0f}%  ({self.n_wins}W/{self.n_trades-self.n_wins}L)",
            f"Avg PnL:      {self.avg_pnl_bps:+.3f} bps",
            f"Best trade:   {self.best_trade_bps:+.3f} bps",
            f"Worst trade:  {self.worst_trade_bps:+.3f} bps",
            f"Time stops:   {self.n_time_stops}",
            f"Stop losses:  {self.n_stop_losses}",
            f"Daily PnL:    ${self.daily_pnl_usd:+.4f}",
        ])


# ──────────────────────────────────────────────────────────────────
# PRICE BUFFER
# ──────────────────────────────────────────────────────────────────

class PriceBuffer:
    def __init__(self, maxlen: int = 2000):
        self._buf: deque = deque(maxlen=maxlen)

    def append(self, ts: float, price: float):
        self._buf.append((ts, price))

    def current(self) -> Optional[float]:
        return self._buf[-1][1] if self._buf else None

    def price_n_sec_ago(self, sec: float) -> Optional[float]:
        target = time.time() - sec
        result = None
        for ts, px in self._buf:
            if ts <= target:
                result = px
            else:
                break
        return result

    def return_bps(self, sec: float) -> Optional[float]:
        cur  = self.current()
        past = self.price_n_sec_ago(sec)
        if cur is None or past is None or past == 0:
            return None
        return (cur - past) / past * 10_000

    def age(self) -> float:
        return time.time() - self._buf[-1][0] if self._buf else float("inf")


# ──────────────────────────────────────────────────────────────────
# MARKET STATE
# ──────────────────────────────────────────────────────────────────

class MarketState:
    def __init__(self):
        self.binance  = PriceBuffer()
        self.coinbase = PriceBuffer()
        self.bitso    = PriceBuffer()
        self.bitso_bid:        float = 0.0
        self.bitso_ask:        float = 0.0
        self.bitso_spread_bps: float = 0.0
        # Reconnect tracking: timestamps of recent Bitso reconnects.
        # Used by feed_quality_ok() to detect dead-market conditions.
        self._reconnect_ts: deque = deque(maxlen=20)

    def record_reconnect(self):
        """Call every time the Bitso WebSocket reconnects."""
        self._reconnect_ts.append(time.time())

    def feed_quality_ok(self) -> bool:
        """
        Returns False if the Bitso feed is in a reconnect loop.
        A reconnect loop means the overnight book is dead — no liquidity,
        no fills, and any entry will likely end in a force close.

        Rule: block entries if ≥3 reconnects in the last 5 minutes.
        In a healthy active-hours session: 0-1 reconnects per hour.
        In a dead overnight session: 1 reconnect every 30 seconds = 10/5min.
        """
        now     = time.time()
        cutoff  = now - 300   # 5 minutes
        recent  = sum(1 for ts in self._reconnect_ts if ts > cutoff)
        if recent >= 3:
            return False
        return True

    def update_bitso_top(self, bid: float, ask: float):
        if bid <= 0 or ask <= 0 or bid >= ask:
            return
        mid = (bid + ask) / 2
        self.bitso_bid        = bid
        self.bitso_ask        = ask
        self.bitso_spread_bps = (ask - bid) / mid * 10_000
        self.bitso.append(time.time(), mid)

    def feeds_healthy(self) -> bool:
        return (
            self.binance.age()  < 10.0
            and self.coinbase.age() < 10.0
            and self.bitso.age()    < 5.0
        )


# ──────────────────────────────────────────────────────────────────
# RISK STATE
# ──────────────────────────────────────────────────────────────────

class RiskState:
    def __init__(self):
        self.position_asset:     float = 0.0
        self.kill_switch:        bool  = False
        self.entry_mid:          float = 0.0
        self.entry_ts:           float = 0.0
        self.entry_direction:    str   = "none"
        self.entry_oid:          str   = ""
        self.last_exit_ts:       float = 0.0
        self.exit_oid:           str   = ""
        self.exit_submitted_ts:  float = 0.0
        self.exit_attempt:       int   = 0
        self.last_exit_api_call: float = 0.0
        # WebSocket fill detection: set by user_trades_feed when an order fills.
        # Checked by handle_exit every tick. Eliminates reconciler as primary
        # exit detection mechanism — reduces exit latency from 15-30s to <1s.
        self.ws_fill_detected:   bool  = False
        self.ws_fill_price:      float = 0.0
        self.ws_filled_oid:      str   = ""

    def in_position(self) -> bool:
        return self.entry_direction != "none"

    def check_daily_loss(self, pnl: PnLTracker) -> bool:
        if self.kill_switch:
            return True
        if pnl.daily_pnl_usd <= -MAX_DAILY_LOSS_USD:
            log.error("KILL SWITCH: daily_pnl=$%.4f <= -$%.2f",
                      pnl.daily_pnl_usd, MAX_DAILY_LOSS_USD)
            self.kill_switch = True
        return self.kill_switch


# ──────────────────────────────────────────────────────────────────
# BITSO REST API
# ──────────────────────────────────────────────────────────────────

def _bitso_headers(method: str, path: str, body: str = "") -> dict:
    nonce = str(int(time.time() * 1000))
    msg   = nonce + method.upper() + path + body
    sig   = hmac.new(
        _BITSO_API_SECRET.encode(), msg.encode(), hashlib.sha256,
    ).hexdigest()
    return {
        "Authorization": f"Bitso {_BITSO_API_KEY}:{nonce}:{sig}",
        "Content-Type":  "application/json",
    }


async def _check_balance() -> dict:
    """Returns {success, usd, btc, eth, sol} — all available balances."""
    try:
        import aiohttp
        path    = "/v3/balance/"
        headers = _bitso_headers("GET", path)
        async with aiohttp.ClientSession() as s:
            async with s.get(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    bals = {b["currency"]: b for b in data["payload"]["balances"]}
                    return {
                        "success": True,
                        "usd": float(bals.get("usd", {}).get("available", 0)),
                        "btc": float(bals.get("btc", {}).get("available", 0)),
                        "eth": float(bals.get("eth", {}).get("available", 0)),
                        "sol": float(bals.get("sol", {}).get("available", 0)),
                    }
                return {"success": False, "error": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _submit_order(side: str, price: float, amount_asset: float) -> dict:
    if EXEC_MODE != "live":
        return {"success": True, "paper": True, "oid": f"paper_{int(time.time()*1000)}"}

    bal = await _check_balance()
    if bal.get("success"):
        if side == "buy":
            required_usd = price * amount_asset * 1.002
            if bal["usd"] < required_usd:
                amount_asset = round((bal["usd"] * 0.99) / price, 8)
                log.warning("PREFLIGHT: adjusted BUY size to %.8f %s (USD=$%.2f)",
                            amount_asset, ASSET.upper(), bal["usd"])
                if amount_asset < MIN_TRADE_SIZE:
                    return {"success": False, "error": "balance_too_low"}
        elif side == "sell":
            asset_bal = bal.get(ASSET, 0.0)
            if asset_bal < amount_asset:
                amount_asset = round(asset_bal, 8)
                log.warning("PREFLIGHT: adjusted SELL size to %.8f %s",
                            amount_asset, ASSET.upper())
                if amount_asset < MIN_TRADE_SIZE:
                    return {"success": False, "error": "no_asset_to_sell"}

    try:
        import aiohttp
        path      = "/v3/orders/"
        body_dict = {
            "book":  BITSO_BOOK,
            "side":  side,
            "type":  "limit",
            "major": f"{amount_asset:.8f}",
            "price": f"{price:.2f}",
        }
        body    = json.dumps(body_dict)
        headers = _bitso_headers("POST", path, body)
        async with aiohttp.ClientSession() as s:
            async with s.post(
                BITSO_API_URL + path, headers=headers, data=body,
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    oid = data["payload"].get("oid", "unknown")
                    log.info("ORDER %s %.8f %s @ $%.2f  oid=%s",
                             side.upper(), amount_asset, ASSET.upper(), price, oid)
                    return {"success": True, "oid": oid, "amount": amount_asset}
                log.error("ORDER REJECTED: %s", data)
                return {"success": False, "error": data}
    except Exception as e:
        log.error("ORDER EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}


async def _submit_market_order(side: str, amount_asset: float) -> dict:
    """
    Submit a market order. Fills immediately at best available price.
    Used for entries only — guarantees 100% fill rate.
    Cost: ~half spread (paying the ask on buy, bid on sell).
    At mean BTC/USD spread of 2.69 bps, cost = ~1.35 bps per entry.
    This is cheaper than limit at ask+30 ticks which paid 1.35 + 0.43 = 1.78 bps
    and still had 60%+ unfill rate.
    """
    if EXEC_MODE != "live":
        return {"success": True, "paper": True, "oid": f"paper_{int(time.time()*1000)}"}

    bal = await _check_balance()
    if bal.get("success"):
        if side == "buy":
            usd_avail = bal["usd"]
            if usd_avail < 1.0:
                return {"success": False, "error": "balance_too_low"}
            # For market buy we don't know exact fill price.
            # Use a conservative estimate: mid + 0.5% to check if we have enough USD.
            # Bitso will reject if insufficient — this is an early guard only.
            # We do not resize here — Bitso handles the actual size vs balance check.
        elif side == "sell":
            asset_bal = bal.get(ASSET, 0.0)
            if asset_bal < amount_asset:
                amount_asset = round(asset_bal, 8)
                log.warning("PREFLIGHT MARKET: adjusted SELL size to %.8f %s",
                            amount_asset, ASSET.upper())
                if amount_asset < MIN_TRADE_SIZE:
                    return {"success": False, "error": "no_asset_to_sell"}

    try:
        import aiohttp
        path      = "/v3/orders/"
        body_dict = {
            "book":  BITSO_BOOK,
            "side":  side,
            "type":  "market",
            "major": f"{amount_asset:.8f}",
        }
        body    = json.dumps(body_dict)
        headers = _bitso_headers("POST", path, body)
        async with aiohttp.ClientSession() as s:
            async with s.post(
                BITSO_API_URL + path, headers=headers, data=body,
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    oid = data["payload"].get("oid", "unknown")
                    log.info("MARKET ORDER %s %.8f %s  oid=%s",
                             side.upper(), amount_asset, ASSET.upper(), oid)
                    return {"success": True, "oid": oid, "amount": amount_asset}
                log.error("MARKET ORDER REJECTED: %s", data)
                return {"success": False, "error": data}
    except Exception as e:
        log.error("MARKET ORDER EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}


async def _cancel_order(oid: str) -> bool:
    if EXEC_MODE != "live" or not oid or oid.startswith("paper_"):
        return True
    try:
        import aiohttp
        path    = f"/v3/orders/{oid}"
        headers = _bitso_headers("DELETE", path)
        async with aiohttp.ClientSession() as s:
            async with s.delete(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                data = await r.json()
                ok   = data.get("success", False)
                if not ok:
                    code = data.get("error", {}).get("code", "")
                    if code in ("0303", "0304"):
                        return True
                    log.warning("CANCEL failed oid=%s: %s", oid, data)
                return ok
    except Exception as e:
        log.warning("CANCEL exception oid=%s: %s", oid, e)
        return False


# ──────────────────────────────────────────────────────────────────
# SIGNAL
# ──────────────────────────────────────────────────────────────────

def evaluate_signal(state: MarketState) -> Optional[str]:
    if state.bitso_spread_bps > SPREAD_MAX_BPS:
        return None

    bn_ret = state.binance.return_bps(SIGNAL_WINDOW_SEC)
    cb_ret = state.coinbase.return_bps(SIGNAL_WINDOW_SEC)
    bt_ret = state.bitso.return_bps(SIGNAL_WINDOW_SEC)

    if bn_ret is None or cb_ret is None or bt_ret is None:
        return None

    lead_move = max(abs(bn_ret), abs(cb_ret))
    if lead_move < ENTRY_THRESHOLD_BPS * 0.5:
        return None
    # Research (module 11): early entries (Bitso unmoved) yield +1.4 bps more than
    # late entries. Tightened from 80% to 40% — if Bitso has already moved >40% of
    # the threshold, the lag window is closing fast and entry quality degrades sharply.
    # At 6bps: blocks if bt_ret > 2.4 bps. Removes the worst 40% of late signals.
    if abs(bt_ret) > ENTRY_THRESHOLD_BPS * 0.4:
        return None

    bn_div = bn_ret - bt_ret
    cb_div = cb_ret - bt_ret
    best   = cb_div if abs(cb_div) >= abs(bn_div) else bn_div

    # Signal probe: log near-threshold events (once per 2s max to avoid spam).
    # Shows live divergence so we can monitor signal quality.
    if abs(best) > ENTRY_THRESHOLD_BPS * 0.5:
        now_sig = time.time()
        if not hasattr(evaluate_signal, '_last_log') or now_sig - evaluate_signal._last_log > 2.0:
            log.info(
                "[Signal] bn=%+.2f cb=%+.2f best=%+.2f thr=%.1f bt=%+.2f",
                bn_div, cb_div, best, ENTRY_THRESHOLD_BPS, bt_ret,
            )
            evaluate_signal._last_log = now_sig

    if COMBINED_SIGNAL:
        bn_dir = (1 if bn_div >  ENTRY_THRESHOLD_BPS else
                 -1 if bn_div < -ENTRY_THRESHOLD_BPS else 0)
        cb_dir = (1 if cb_div >  ENTRY_THRESHOLD_BPS else
                 -1 if cb_div < -ENTRY_THRESHOLD_BPS else 0)
        if bn_dir == 0 or cb_dir == 0 or bn_dir != cb_dir:
            return None
        return "buy" if bn_dir > 0 else "sell"
    else:
        if best >  ENTRY_THRESHOLD_BPS: return "buy"
        if best < -ENTRY_THRESHOLD_BPS: return "sell"
        return None


# ──────────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────────

async def handle_entry(
    direction: str,
    state:     MarketState,
    risk:      RiskState,
    pnl:       PnLTracker,
):
    if risk.check_daily_loss(pnl): return
    if time.time() - risk.last_exit_ts < COOLDOWN_SEC: return
    if risk.in_position(): return
    if direction == "sell": return   # spot only

    # ORPHAN GUARD: fast-path check before every entry.
    # If asset > 0 when internal state says flat, a prior exit did not fill.
    # Block entry and let the reconciler (30s cycle) handle the forced sell.
    if EXEC_MODE == "live":
        bal = await _check_balance()
        if bal.get("success") and bal.get(ASSET, 0.0) > MIN_TRADE_SIZE:
            log.warning(
                "ORPHAN GUARD: %.8f %s in account, internal=FLAT. "
                "Blocking entry. Reconciler handles in %ds.",
                bal[ASSET], ASSET.upper(), int(RECONCILE_SEC),
            )
            risk.last_exit_ts = time.time()
            return

    tick        = 0.01
    entry_mid   = (state.bitso_bid + state.bitso_ask) / 2
    bn_ret      = state.binance.return_bps(SIGNAL_WINDOW_SEC)  or 0.0
    cb_ret      = state.coinbase.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    bt_ret      = state.bitso.return_bps(SIGNAL_WINDOW_SEC)    or 0.0

    log.info("[%s] ENTRY %s (MARKET)  mid=$%.2f  spread=%.2fbps  bn=%+.2f cb=%+.2f bt=%+.2f",
             EXEC_MODE.upper(), direction.upper(), entry_mid,
             state.bitso_spread_bps, bn_ret, cb_ret, bt_ret)

    # Market order: fills immediately at best available ask.
    # Cost: ~half spread (~1.35 bps mean). No partial fills, no cancel race,
    # no ENTRY UNFILLED events, no open buy orders left on exchange.
    result = await _submit_market_order(direction, MAX_POS_ASSET)
    if not result.get("success"):
        risk.last_exit_ts = time.time()
        return

    # Use actual submitted size from preflight adjustment, not MAX_POS_ASSET.
    # When USD is insufficient, preflight reduces size. Using MAX_POS_ASSET
    # would cause PnL to be calculated on a larger size than actually traded.
    actual_size = result.get("amount", MAX_POS_ASSET)

    risk.position_asset  = actual_size if direction == "buy" else -actual_size
    risk.entry_mid       = entry_mid
    risk.entry_ts        = time.time()
    risk.entry_direction = direction
    risk.entry_oid       = result.get("oid", "")


# ──────────────────────────────────────────────────────────────────
# EXIT CHASER
# ──────────────────────────────────────────────────────────────────

def _reset_position(
    risk:     RiskState,
    pnl:      PnLTracker,
    exit_mid: float,
    hold_sec: float,
    reason:   str,
) -> None:
    """
    Record the completed trade and reset all position state atomically.
    This is the ONLY place a trade gets recorded.
    Called by: handle_exit (on confirmed no-asset signals) and reconciler_loop.
    NEVER called optimistically after a force-close submission.
    """
    pnl_bps, pnl_usd = pnl.record(
        direction  = risk.entry_direction,
        entry_mid  = risk.entry_mid,
        exit_mid   = exit_mid,
        size_asset = abs(risk.position_asset),
        hold_sec   = hold_sec,
        reason     = reason,
        entry_oid  = risk.entry_oid,
        exit_oid   = risk.exit_oid,
    )
    log.info(
        "[%s] EXIT RECORDED %s  pnl=%+.3fbps ($%+.6f)  hold=%.1fs  %s"
        "  | trades=%d win=%.0f%%  daily=$%+.4f",
        EXEC_MODE.upper(), risk.entry_direction.upper(),
        pnl_bps, pnl_usd, hold_sec, reason,
        pnl.n_trades, pnl.win_rate * 100, pnl.daily_pnl_usd,
    )
    risk.position_asset     = 0.0
    risk.entry_direction    = "none"
    risk.entry_oid          = ""
    risk.exit_oid           = ""
    risk.exit_attempt       = 0
    risk.last_exit_api_call = 0.0
    risk.ws_fill_detected   = False
    risk.ws_fill_price      = 0.0
    risk.ws_filled_oid      = ""
    risk.last_exit_ts       = time.time()
    risk.check_daily_loss(pnl)


async def _cancel_unfilled_entry(risk: "RiskState", state: "MarketState", pnl: "PnLTracker") -> None:
    """
    Called when exit attempt 0 sees no asset in account.
    At attempt 0, no exit order has been submitted yet, so "no asset"
    means the ENTRY order never filled — or only partially filled.

    v4.5 fix: check order status BEFORE cancelling.
    If the entry is 'partially filled', some BTC was already bought.
    We must cancel the remainder AND immediately sell the filled portion
    rather than going FLAT and letting it become an orphan.
    """
    oid = risk.entry_oid

    # Check if entry was partially filled before cancelling
    partial_fill_size = 0.0
    if EXEC_MODE == "live" and oid:
        try:
            import aiohttp
            path    = f"/v3/orders/{oid}"
            headers = _bitso_headers("GET", path)
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    BITSO_API_URL + path, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as r:
                    data = await r.json()
            if data.get("success"):
                order = data.get("payload", [{}])
                if isinstance(order, list):
                    order = order[0] if order else {}
                status          = order.get("status", "")
                original_amount = float(order.get("original_amount", 0))
                unfilled_amount = float(order.get("unfilled_amount", 0))
                filled_amount   = original_amount - unfilled_amount
                if status == "partially filled" and filled_amount > MIN_TRADE_SIZE:
                    partial_fill_size = filled_amount
                    log.warning(
                        "[%s] ENTRY PARTIAL FILL: oid=%s filled=%.8f unfilled=%.8f — "
                        "will cancel remainder and sell filled portion.",
                        EXEC_MODE.upper(), oid, filled_amount, unfilled_amount,
                    )
        except Exception as e:
            log.warning("[%s] ENTRY status check failed: %s", EXEC_MODE.upper(), e)

    log.warning(
        "[%s] ENTRY UNFILLED: no %s in account at time_stop. "
        "Cancelling entry oid=%s. partial_fill=%.8f",
        EXEC_MODE.upper(), ASSET.upper(), oid, partial_fill_size,
    )

    # Cancel with retry — if first cancel fails, the open buy order will
    # reserve USD indefinitely and may fill at a stale price later.
    cancel_confirmed = False
    for attempt in range(3):
        ok = await _cancel_order(oid)
        if ok:
            cancel_confirmed = True
            break
        log.warning("[%s] ENTRY cancel attempt %d failed, retrying...",
                    EXEC_MODE.upper(), attempt + 1)
        await asyncio.sleep(0.5)

    if not cancel_confirmed:
        log.error(
            "[%s] ENTRY cancel FAILED after 3 attempts oid=%s — "
            "order may remain open on Bitso. Startup will clean it next session.",
            EXEC_MODE.upper(), oid,
        )

    if partial_fill_size > MIN_TRADE_SIZE:
        # Partial fill: sell the BTC that did fill to avoid orphan.
        # Only proceed if cancel confirmed — if cancel failed, the unfilled
        # portion may still fill and create a larger position than expected.
        if not cancel_confirmed:
            log.warning(
                "[%s] ENTRY PARTIAL FILL: skipping emergency sell because "
                "cancel not confirmed. Reconciler will handle BTC if it arrives.",
                EXEC_MODE.upper(),
            )
            # Fall through to state reset — reconciler catches BTC as orphan
        else:
            # Wait briefly for Bitso to release the filled BTC after cancel
            await asyncio.sleep(1.0)
            bid = state.bitso_bid if state.bitso_bid > 0 else 0
            if bid > 0:
                sell_px = bid * (1 - FORCE_CLOSE_SLIPPAGE)
                result  = await _submit_order("sell", sell_px, partial_fill_size)
                log.warning(
                    "[%s] ENTRY PARTIAL FILL: emergency sell %.8f @ $%.2f ok=%s",
                    EXEC_MODE.upper(), partial_fill_size, sell_px,
                    result.get("success"),
                )
            # Record a trade for the partial fill using current mid as exit
            exit_mid = (state.bitso_bid + state.bitso_ask) / 2 if state.bitso_bid > 0 else risk.entry_mid
            hold_sec = time.time() - risk.entry_ts
            # Temporarily set position_asset to the filled amount for accurate PnL
            risk.position_asset = partial_fill_size
            _reset_position(risk, pnl, exit_mid, hold_sec, "entry_partial_fill")
            return

    # Fully unfilled: clear state without recording a trade
    risk.position_asset     = 0.0
    risk.entry_direction    = "none"
    risk.entry_oid          = ""
    risk.exit_oid           = ""
    risk.exit_attempt       = 0
    risk.last_exit_api_call = 0.0
    risk.last_exit_ts = time.time() + POST_RESET_COOLDOWN - COOLDOWN_SEC
    log.warning(
        "[%s] ENTRY UNFILLED cooldown: blocking entries for %.0fs "
        "(cancel+fill race protection).",
        EXEC_MODE.upper(), POST_RESET_COOLDOWN,
    )


async def handle_exit(
    state: MarketState,
    risk:  RiskState,
    pnl:   PnLTracker,
):
    """
    Exit state machine — v4.5.5 hybrid approach.

    attempt 0  wait for time_stop or stop_loss trigger
               submit passive limit sell at bid (zero cost if fills)
    attempt 1  passive limit open — wait EXIT_CHASE_SEC
               if unfilled: cancel + submit market order
    attempt 2  market order submitted — wait for poller/reconciler
               if poller cancels partial: resubmit market for remainder
               if timeout (2x EXIT_CHASE_SEC): force reconciler reset

    Force close removed. Market order guarantees exit at ~1.35 bps cost.
    No more $357 force close losses.
    """
    if not risk.in_position():
        return

    # ── WebSocket fill detection (fast path) ─────────────────────
    # user_trades_feed sets ws_fill_detected=True the instant an order
    # fills. Check this on every Bitso tick. When detected, record the
    # trade immediately using the actual fill price from the WS event.
    # This replaces the reconciler as the primary fill detection mechanism,
    # reducing exit latency from 15-30s to <1s and eliminating the need
    # for the 45s cooldown in most cases.
    # IMPORTANT: this check must be BEFORE the attempt >= 4 guard so that
    # force close fills (attempt 4) are also detected via WS, not only
    # via the reconciler fallback.
    if risk.ws_fill_detected:
        now_ts   = time.time()
        hold_sec = now_ts - risk.entry_ts
        exit_mid = risk.ws_fill_price if risk.ws_fill_price > 0 else (state.bitso_bid + state.bitso_ask) / 2
        reason   = "ws_fill"
        log.info(
            "[%s] WS FILL DETECTED oid=%s price=$%.2f  hold=%.1fs",
            EXEC_MODE.upper(), risk.ws_filled_oid, exit_mid, hold_sec,
        )
        _reset_position(risk, pnl, exit_mid, hold_sec, reason)
        return

    # Chaser is done after force close. Reconciler handles outcome.
    if risk.exit_attempt >= 4:
        return

    now_ts      = time.time()
    current_mid = (state.bitso_bid + state.bitso_ask) / 2
    hold_sec    = now_ts - risk.entry_ts
    tick        = 0.01

    if risk.entry_direction == "buy":
        pnl_bps_live  = (current_mid - risk.entry_mid) / risk.entry_mid * 10_000
        exit_side     = "sell"
        passive_px    = state.bitso_bid
        aggressive_px = state.bitso_bid - tick
        floor_px      = risk.entry_mid * (1 - STOP_LOSS_BPS / 10_000)
    else:
        pnl_bps_live  = (risk.entry_mid - current_mid) / risk.entry_mid * 10_000
        exit_side     = "buy"
        passive_px    = state.bitso_ask
        aggressive_px = state.bitso_ask + tick
        floor_px      = None

    is_stop_loss = pnl_bps_live < -STOP_LOSS_BPS
    is_time_stop = hold_sec >= HOLD_SEC

    # ── attempt 0: trigger check ─────────────────────────────────
    if risk.exit_attempt == 0:
        reason = "stop_loss" if is_stop_loss else ("time_stop" if is_time_stop else None)
        if reason is None:
            return
        # Floor guard for time stops only.
        # Stop losses must never be deferred — the market is already moving against us.
        max_deferral = hold_sec >= HOLD_SEC * 2
        if (exit_side == "sell" and floor_px
                and passive_px < floor_px
                and not is_stop_loss
                and not max_deferral):
            log.debug("[%s] EXIT deferred: bid $%.2f < floor $%.2f (hold=%.1fs)",
                     EXEC_MODE.upper(), passive_px, floor_px, hold_sec)
            return

        if now_ts - risk.last_exit_api_call < 2.0:
            return

        # STOP LOSS → immediate market order.
        # When price is moving against us, a passive limit at bid will not fill
        # for up to EXIT_CHASE_SEC (8s) while the bid drops further.
        # 8s of fast-moving BTC = 20+ bps additional loss (confirmed in live logs).
        # Market order costs ~1.35 bps. Passive limit overshoot costs 20-50 bps.
        if is_stop_loss:
            log.warning("[%s] EXIT STOP LOSS (MARKET): %s  pnl=%.3fbps — direct market order.",
                        EXEC_MODE.upper(), exit_side.upper(), pnl_bps_live)
            risk.last_exit_api_call = time.time()
            result = await _submit_market_order(exit_side, abs(risk.position_asset))
            if result.get("success"):
                risk.exit_oid          = result.get("oid", "")
                risk.exit_submitted_ts = time.time()
                risk.exit_attempt      = 2   # skip to attempt 2 — market already submitted
                log.info("[%s] STOP LOSS market order submitted oid=%s",
                         EXEC_MODE.upper(), risk.exit_oid)
            elif result.get("error") in _NO_ASSET_ERRORS:
                await _cancel_unfilled_entry(risk, state, pnl)
            return

        # TIME STOP → passive limit first (free if fills), market fallback after 8s.
        log.info("[%s] EXIT attempt 1 (passive limit): %s @ $%.2f  pnl=%.3fbps  %s",
                 EXEC_MODE.upper(), exit_side.upper(), passive_px, pnl_bps_live, reason)
        risk.last_exit_api_call = time.time()
        result = await _submit_order(exit_side, passive_px, abs(risk.position_asset))
        if result.get("success"):
            risk.exit_oid          = result.get("oid", "")
            risk.exit_submitted_ts = time.time()
            risk.exit_attempt      = 1
        elif result.get("error") in _NO_ASSET_ERRORS:
            # Entry never filled — cancel and clear state.
            await _cancel_unfilled_entry(risk, state, pnl)
        return

    # ── wait before switching to market ──────────────────────────
    time_since_exit = time.time() - risk.exit_submitted_ts
    if time_since_exit < EXIT_CHASE_SEC:
        return

    # ── attempt 1 → market fallback ──────────────────────────────
    # Passive limit did not fill within EXIT_CHASE_SEC seconds.
    # Cancel it and submit a market order — guaranteed fill, ~1.35 bps cost.
    # This replaces the 3-attempt chaser + force close that caused $357 losses.
    if risk.exit_attempt == 1:
        log.warning(
            "[%s] EXIT market fallback: passive limit unfilled for %.0fs. "
            "Cancelling oid=%s → market sell.",
            EXEC_MODE.upper(), time_since_exit, risk.exit_oid,
        )
        await _cancel_order(risk.exit_oid)
        risk.last_exit_api_call = time.time()
        result = await _submit_market_order(exit_side, abs(risk.position_asset))
        if result.get("success"):
            risk.exit_oid          = result.get("oid", "")
            risk.exit_submitted_ts = time.time()
            risk.exit_attempt      = 2
            log.info("[%s] EXIT market order submitted oid=%s",
                     EXEC_MODE.upper(), risk.exit_oid)
        elif result.get("error") in _NO_ASSET_ERRORS:
            # Passive limit filled silently — reset position.
            log.warning("[%s] EXIT market: no asset — passive limit filled. Resetting.",
                        EXEC_MODE.upper())
            actual_reason = "stop_loss" if is_stop_loss else "time_stop"
            _reset_position(risk, pnl, current_mid, hold_sec, actual_reason)
        return

    # ── attempt 2: market submitted, waiting for poller/reconciler ──
    # Market order was submitted. Poller checks every 2s for fill confirmation.
    # If the market exit partially filled and the poller cancelled the remainder,
    # exit_oid will be cleared (poller sets it to "" after cancel). In that case,
    # resubmit a market order for the remaining position_asset amount.
    if risk.exit_attempt == 2:
        # Poller cancelled a partial market exit — resubmit for remainder
        if not risk.exit_oid and abs(risk.position_asset) > MIN_TRADE_SIZE:
            log.warning(
                "[%s] EXIT attempt 2: market partial fill — resubmitting market "
                "order for remaining %.8f %s.",
                EXEC_MODE.upper(), abs(risk.position_asset), ASSET.upper(),
            )
            risk.last_exit_api_call = time.time()
            result = await _submit_market_order(exit_side, abs(risk.position_asset))
            if result.get("success"):
                risk.exit_oid          = result.get("oid", "")
                risk.exit_submitted_ts = time.time()
            elif result.get("error") in _NO_ASSET_ERRORS:
                _reset_position(risk, pnl, current_mid, hold_sec, "market_remainder_filled")
            return

        if time_since_exit > EXIT_CHASE_SEC * 2:
            log.error(
                "[%s] EXIT market order not detected after %.0fs. "
                "Forcing reconciler reset.",
                EXEC_MODE.upper(), time_since_exit,
            )
            _reset_position(risk, pnl, current_mid, hold_sec, "market_timeout")
        return


# ──────────────────────────────────────────────────────────────────
# RECONCILER
# Runs every RECONCILE_SEC. Checks real Bitso balance vs internal state.
# Handles all failure modes that the exit chaser cannot reach.
# ──────────────────────────────────────────────────────────────────

async def reconciler_loop(
    state: MarketState,
    risk:  RiskState,
    pnl:   PnLTracker,
):
    await asyncio.sleep(RECONCILE_SEC)   # initial warmup

    while True:
        await asyncio.sleep(RECONCILE_SEC)

        if EXEC_MODE != "live":
            continue

        bal = await _check_balance()
        if not bal.get("success"):
            log.warning("[Reconciler] Balance check failed. Skipping cycle.")
            continue

        asset_bal     = bal.get(ASSET, 0.0)
        internal_flat = not risk.in_position()

        # ── Case A: Orphan ─────────────────────────────────────────
        # Exchange has asset, internal=FLAT.
        # Previous session crashed after fill, or force close filled after reset.
        if internal_flat and asset_bal > MIN_TRADE_SIZE:
            log.error(
                "[Reconciler] ORPHAN: %.8f %s in account, internal=FLAT. "
                "Emergency sell.",
                asset_bal, ASSET.upper(),
            )
            if state.bitso_bid > 0:
                px     = state.bitso_bid * (1 - FORCE_CLOSE_SLIPPAGE * 2)
                result = await _submit_order("sell", px, asset_bal)
                msg = (
                    f"RECONCILER ORPHAN [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                    f"Balance: {asset_bal:.8f} {ASSET.upper()}\n"
                    f"Emergency sell @ ${px:.2f}\n"
                    f"OK: {result.get('success')}"
                )
                log.error("[Reconciler] %s", msg.replace('\n', ' | '))
                await tg(msg)
            # Block new entries for POST_RESET_COOLDOWN seconds.
            # Orphan events happen because the Bitso balance API shows 0 briefly
            # after a fill (settlement lag), triggering a false SILENT FILL reset,
            # then a new entry fires before the balance settles and the original
            # asset reappears. A 45s cooldown fully covers the settlement window.
            risk.last_exit_ts = time.time() + POST_RESET_COOLDOWN - COOLDOWN_SEC
            log.warning(
                "[Reconciler] ORPHAN cooldown: blocking entries for %.0fs.",
                POST_RESET_COOLDOWN,
            )

        # ── Case B: Silent fill ────────────────────────────────────
        # Internal=IN_POSITION but no asset in account.
        # Exit chaser's previous attempt filled without detection.
        # v4.5.1: Verify exit order is truly completed before resetting.
        # Root cause of orphan cascade: Bitso shows balance=0 transiently
        # after a PARTIAL exit fill (settlement lag). If we reset here,
        # the remaining BTC from the entry appears minutes later as an orphan.
        # Fix: check exit order status first. Only reset if 'completed' or gone.
        elif (not internal_flat
              and asset_bal < MIN_TRADE_SIZE
              and risk.exit_attempt > 0):

            # Verify exit order is truly done before accepting this as a full fill
            exit_truly_done = True   # default: proceed with reset
            if EXEC_MODE == "live" and risk.exit_oid:
                try:
                    import aiohttp
                    chk_path    = f"/v3/orders/{risk.exit_oid}"
                    chk_headers = _bitso_headers("GET", chk_path)
                    async with aiohttp.ClientSession() as s:
                        async with s.get(
                            BITSO_API_URL + chk_path, headers=chk_headers,
                            timeout=aiohttp.ClientTimeout(total=3),
                        ) as r:
                            chk_data = await r.json()
                    if chk_data.get("success"):
                        chk_order  = chk_data.get("payload", [{}])
                        if isinstance(chk_order, list):
                            chk_order = chk_order[0] if chk_order else {}
                        chk_status = chk_order.get("status", "")
                        if chk_status == "partially filled":
                            # Exit only partially filled — do NOT reset yet.
                            # The unfilled remainder is still open (or being chased).
                            # Reconciler will catch it on the next cycle.
                            log.warning(
                                "[Reconciler] SILENT FILL skipped: exit oid=%s "
                                "is still 'partially filled'. Waiting for full fill.",
                                risk.exit_oid,
                            )
                            exit_truly_done = False
                        elif chk_status in ("open",):
                            log.warning(
                                "[Reconciler] SILENT FILL skipped: exit oid=%s "
                                "is still 'open'. Chaser will handle it.",
                                risk.exit_oid,
                            )
                            exit_truly_done = False
                        # 'completed', 'cancelled', or not found → proceed with reset
                    else:
                        err_code = chk_data.get("error", {}).get("code", "")
                        if err_code not in ("0303", "0304"):
                            # Unexpected error — skip reset, try again next cycle
                            log.warning(
                                "[Reconciler] SILENT FILL: exit order check failed "
                                "(code=%s). Skipping reset this cycle.", err_code,
                            )
                            exit_truly_done = False
                except Exception as e:
                    log.warning("[Reconciler] SILENT FILL: exit order check error: %s. "
                                "Proceeding with reset.", e)

            if not exit_truly_done:
                continue

            current_mid = (state.bitso_bid + state.bitso_ask) / 2
            hold_sec    = time.time() - risk.entry_ts
            log.warning(
                "[Reconciler] SILENT FILL: internal=IN_POSITION, "
                "%.8f %s in account. Resetting.",
                asset_bal, ASSET.upper(),
            )
            _reset_position(risk, pnl, current_mid, hold_sec, "reconcile_silent_fill")
            risk.last_exit_ts = time.time() + POST_RESET_COOLDOWN - COOLDOWN_SEC
            log.warning(
                "[Reconciler] SILENT FILL cooldown: blocking entries for %.0fs.",
                POST_RESET_COOLDOWN,
            )
            await tg(
                f"RECONCILER SILENT FILL [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                f"Position reset and trade recorded.\n"
                f"Entry blocked for {POST_RESET_COOLDOWN:.0f}s (balance settlement)."
            )

        # ── Case C: Stuck force close ──────────────────────────────
        # IN_POSITION + asset still in account + attempt >= 4.
        # Force close order placed but not yet filled. Nuclear exit at 2%.
        elif (not internal_flat
              and asset_bal > MIN_TRADE_SIZE
              and risk.exit_attempt >= 4):
            log.error(
                "[Reconciler] STUCK EXIT: %.8f %s in account after "
                "force close. Nuclear exit at 2%% below bid.",
                asset_bal, ASSET.upper(),
            )
            if risk.exit_oid:
                await _cancel_order(risk.exit_oid)
            if state.bitso_bid > 0:
                px     = state.bitso_bid * (1 - FORCE_CLOSE_SLIPPAGE * 4)
                result = await _submit_order("sell", px, asset_bal)
                if result.get("success"):
                    risk.exit_oid          = result.get("oid", "")
                    risk.exit_submitted_ts = time.time()
                msg = (
                    f"RECONCILER NUCLEAR EXIT [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                    f"Balance: {asset_bal:.8f} {ASSET.upper()}\n"
                    f"Nuclear sell @ ${px:.2f} (2% below bid)\n"
                    f"OK: {result.get('success')}"
                )
                log.error("[Reconciler] %s", msg.replace('\n', ' | '))
                await tg(msg)

        else:
            log.info(
                "[Reconciler] OK  %s=%.8f  USD=$%.2f  "
                "internal=%s  attempt=%d",
                ASSET.upper(), asset_bal, bal["usd"],
                "IN_POS" if risk.in_position() else "FLAT",
                risk.exit_attempt,
            )


# ──────────────────────────────────────────────────────────────────
# WEBSOCKET FEEDS
# ──────────────────────────────────────────────────────────────────

async def user_trades_feed(state: MarketState, risk: RiskState, pnl: PnLTracker):
    """
    Fast order status poller — replaces broken WebSocket user_trades approach.

    Bitso's user_trades WebSocket authentication format is undocumented and
    returns "invalid message" for all tested auth formats. This alternative
    polls the REST API GET /v3/orders/{oid} every ORDER_POLL_SEC seconds
    when in position. Order status "completed" means the exit filled.

    v4.3 fix: poller now calls _reset_position DIRECTLY on fill detection
    instead of setting ws_fill_detected flag and waiting for a Bitso tick.
    Previous approach failed when BT was stale — flag was set but handle_exit
    never fired, reconciler caught it 15s later instead.
    """
    if EXEC_MODE != "live":
        return

    ORDER_POLL_SEC = 2.0   # poll every 2 seconds when in position

    log.info("[OrderPoller] Started. Polling interval: %.0fs when in position.", ORDER_POLL_SEC)

    while True:
        try:
            await asyncio.sleep(ORDER_POLL_SEC)

            # Only poll when we have an open exit order to check
            if not risk.in_position() or not risk.exit_oid:
                continue

            # Already detected via this path — skip
            if risk.ws_fill_detected:
                continue

            oid = risk.exit_oid
            try:
                import aiohttp
                path    = f"/v3/orders/{oid}"
                headers = _bitso_headers("GET", path)
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        BITSO_API_URL + path, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r:
                        data = await r.json()

                if not data.get("success"):
                    # Order not found (0303/0304) = already filled and cleared
                    err_code = data.get("error", {}).get("code", "")
                    if err_code in ("0303", "0304"):
                        log.info(
                            "[OrderPoller] oid=%s not found (code %s) — filled.",
                            oid, err_code,
                        )
                        if risk.in_position():
                            hold_sec = time.time() - risk.entry_ts
                            # Try to fetch actual fill price from order_trades API
                            # Market orders return price=0 in the order payload,
                            # but order_trades gives the actual weighted fill price.
                            fill_price = 0.0
                            try:
                                trades_path = f"/v3/order_trades/?oid={oid}"
                                trades_hdrs = _bitso_headers("GET", trades_path)
                                async with aiohttp.ClientSession() as s2:
                                    async with s2.get(
                                        BITSO_API_URL + trades_path,
                                        headers=trades_hdrs,
                                        timeout=aiohttp.ClientTimeout(total=3),
                                    ) as r2:
                                        tdata = await r2.json()
                                if tdata.get("success"):
                                    trades_list = tdata.get("payload", [])
                                    if trades_list:
                                        total_val  = sum(float(t["price"]) * float(t["major"]) for t in trades_list)
                                        total_size = sum(float(t["major"]) for t in trades_list)
                                        if total_size > 0:
                                            fill_price = total_val / total_size
                            except Exception as e:
                                log.debug("[OrderPoller] fill price fetch error: %s", e)
                            if fill_price <= 0:
                                fill_price = (state.bitso_bid + state.bitso_ask) / 2
                                log.debug("[OrderPoller] using mid as fill price fallback: $%.2f", fill_price)
                            _reset_position(risk, pnl, fill_price, hold_sec, "poller_fill")
                    continue

                order   = data.get("payload", [{}])
                # Bitso GET /v3/orders/{oid} returns payload as a list
                if isinstance(order, list):
                    order = order[0] if order else {}
                status  = order.get("status", "")

                if status == "completed":
                    # Fetch actual fill price from order_trades API.
                    # Market orders return price=0 in the order payload.
                    # order_trades gives the real weighted average fill price.
                    fill_price = 0.0
                    try:
                        raw_price = float(order.get("price", 0))
                        if raw_price > 0:
                            fill_price = raw_price   # limit order — price is reliable
                        else:
                            # Market order — fetch from trades
                            trades_path = f"/v3/order_trades/?oid={oid}"
                            trades_hdrs = _bitso_headers("GET", trades_path)
                            async with aiohttp.ClientSession() as s2:
                                async with s2.get(
                                    BITSO_API_URL + trades_path,
                                    headers=trades_hdrs,
                                    timeout=aiohttp.ClientTimeout(total=3),
                                ) as r2:
                                    tdata = await r2.json()
                            if tdata.get("success"):
                                trades_list = tdata.get("payload", [])
                                if trades_list:
                                    total_val  = sum(float(t["price"]) * float(t["major"]) for t in trades_list)
                                    total_size = sum(float(t["major"]) for t in trades_list)
                                    if total_size > 0:
                                        fill_price = total_val / total_size
                    except Exception:
                        fill_price = 0.0
                    if fill_price <= 0:
                        fill_price = (state.bitso_bid + state.bitso_ask) / 2
                    log.info(
                        "[OrderPoller] EXIT FILL CONFIRMED: oid=%s price=$%.2f — resetting directly.",
                        oid, fill_price,
                    )
                    if risk.in_position():
                        hold_sec = time.time() - risk.entry_ts
                        _reset_position(risk, pnl, fill_price, hold_sec, "poller_fill")

                elif status == "partially filled":
                    # Partially filled exit order: some BTC sold, rest still open.
                    # v4.5.1: Update position_asset to reflect actual remaining BTC.
                    # Without this, when the next exit fills, PnL is calculated on
                    # the original size instead of the actual remaining size.
                    try:
                        original_amount = float(order.get("original_amount", 0))
                        unfilled_amount = float(order.get("unfilled_amount", 0))
                        already_sold    = original_amount - unfilled_amount
                        if already_sold > MIN_TRADE_SIZE and risk.in_position():
                            remaining = abs(risk.position_asset) - already_sold
                            if remaining > MIN_TRADE_SIZE:
                                risk.position_asset = remaining if risk.position_asset > 0 else -remaining
                                log.warning(
                                    "[OrderPoller] PARTIAL EXIT: oid=%s sold=%.8f remaining=%.8f "
                                    "— updated position_asset.",
                                    oid, already_sold, remaining,
                                )
                    except Exception as e:
                        log.debug("[OrderPoller] Partial exit size update error: %s", e)

                    # If order has been sitting for longer than EXIT_CHASE_SEC, cancel
                    # remainder and let the chaser submit a market order.
                    time_since_submit = time.time() - risk.exit_submitted_ts
                    if time_since_submit > EXIT_CHASE_SEC:
                        log.warning(
                            "[OrderPoller] PARTIAL FILL: oid=%s open for %.0fs — "
                            "cancelling remainder, chaser will resubmit.",
                            oid, time_since_submit,
                        )
                        await _cancel_order(oid)
                        # Reset position_asset to actual BTC balance after cancel.
                        # Incremental polling under-counts fills due to settlement lag —
                        # using the real balance prevents orphan BTC appearing later.
                        if EXEC_MODE == "live":
                            try:
                                actual_bal = await _check_balance()
                                if actual_bal.get("success"):
                                    actual_btc = actual_bal.get(ASSET, 0.0)
                                    if actual_btc > MIN_TRADE_SIZE:
                                        risk.position_asset = actual_btc if risk.position_asset > 0 else -actual_btc
                                        log.warning(
                                            "[OrderPoller] PARTIAL CANCEL: reset position_asset "
                                            "to actual balance %.8f %s",
                                            actual_btc, ASSET.upper(),
                                        )
                                    else:
                                        # Balance shows zero — position already fully exited
                                        log.warning("[OrderPoller] PARTIAL CANCEL: balance=0, position fully exited.")
                                        risk.exit_oid = ""
                                        hold_sec = time.time() - risk.entry_ts
                                        _reset_position(risk, pnl,
                                                        (state.bitso_bid + state.bitso_ask) / 2,
                                                        hold_sec, "partial_limit_fully_filled")
                                        continue
                            except Exception as e:
                                log.debug("[OrderPoller] balance check after cancel error: %s", e)
                        # Clear exit_oid so handle_exit attempt 2 detects the cancel
                        # and resubmits a market order for the remaining amount.
                        risk.exit_oid = ""

                elif status in ("cancelled",):
                    log.warning(
                        "[OrderPoller] Exit order cancelled: oid=%s. Chaser will resubmit.",
                        oid,
                    )

            except Exception as e:
                log.debug("[OrderPoller] Poll error oid=%s: %s", oid, e)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[OrderPoller] Unexpected error: %s", e)
            await asyncio.sleep(ORDER_POLL_SEC)


async def binance_feed(state: MarketState):
    url     = f"wss://stream.binance.us:9443/ws/{BINANCE_SYMBOL}@bookTicker"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                log.info("[BinanceUS] Connected. Symbol: %s", BINANCE_SYMBOL)
                async for raw in ws:
                    msg = json.loads(raw)
                    b, a = float(msg.get("b", 0)), float(msg.get("a", 0))
                    if b > 0 and a > 0 and b < a:
                        state.binance.append(time.time(), (b + a) / 2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[BinanceUS] %s - retry in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def coinbase_feed(state: MarketState):
    url     = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "type":        "subscribe",
                    "product_ids": [COINBASE_SYMBOL],
                    "channels":    ["ticker"],
                }))
                backoff = 1.0
                log.info("[Coinbase] Connected. Symbol: %s", COINBASE_SYMBOL)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        continue
                    b, a = msg.get("best_bid"), msg.get("best_ask")
                    if b and a:
                        state.coinbase.append(time.time(), (float(b) + float(a)) / 2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Coinbase] %s - retry in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def bitso_feed(state: MarketState, risk: RiskState, pnl: PnLTracker):
    url     = "wss://ws.bitso.com"
    backoff = 1.0
    bids: dict = {}
    asks: dict = {}

    # STALE_RECONNECT_SEC is set at module level from env var (default 60s).
    # 60s suits low-liquidity periods on Bitso. Raise via env if needed.

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**21,
            ) as ws:
                await ws.send(json.dumps({
                    "action": "subscribe", "book": BITSO_BOOK, "type": "diff-orders",
                }))
                # Also subscribe to trades for additional price confirmation.
                await ws.send(json.dumps({
                    "action": "subscribe", "book": BITSO_BOOK, "type": "trades",
                }))
                backoff    = 1.0
                connect_ts = time.time()
                bids.clear()
                asks.clear()
                state.record_reconnect()   # track reconnect for feed quality guard
                log.info("[Bitso] Connected. Book: %s (diff-orders + trades)", BITSO_BOOK)

                async for raw in ws:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        continue

                    msg_type = msg.get("type", "")

                    # ── Trades channel: price buffer heartbeat ────────────
                    if msg_type == "trades":
                        payload = msg.get("payload", [])
                        if isinstance(payload, list) and payload:
                            try:
                                trade_px = float(payload[-1]["r"])
                                if trade_px > 0:
                                    state.bitso.append(time.time(), trade_px)
                                    if bids and asks:
                                        bb, ba = max(bids), min(asks)
                                        if bb > 0 and ba > 0 and bb < ba:
                                            state.update_bitso_top(bb, ba)
                                            await handle_exit(state, risk, pnl)
                                            if (not risk.in_position()
                                                    and not risk.kill_switch
                                                    and state.feeds_healthy()
                                                    and state.feed_quality_ok()):
                                                direction = evaluate_signal(state)
                                                if direction:
                                                    await handle_entry(direction, state, risk, pnl)
                            except Exception:
                                pass
                        continue

                    if msg_type != "diff-orders":
                        continue

                    # ── diff-orders channel ───────────────────────────────
                    # Payload is a list of order diffs. Each entry:
                    #   r = price, t = side (0=bid, 1=ask), a = amount
                    #   a == 0 means order removed; a > 0 means placed/updated
                    # Fires on every book change at any level — millisecond rate.
                    payload = msg.get("payload", [])
                    if not isinstance(payload, list):
                        continue

                    for row in payload:
                        try:
                            px   = float(row["r"])
                            sz   = float(row.get("a", 0))
                            side = int(row.get("t", -1))
                            if side == 0:   # bid
                                bids.pop(px, None) if sz == 0 else bids.__setitem__(px, sz)
                            elif side == 1: # ask
                                asks.pop(px, None) if sz == 0 else asks.__setitem__(px, sz)
                        except Exception:
                            continue

                    if not bids or not asks:
                        continue

                    # Stale guard: fires on EVERY message regardless of book state.
                    # Must be checked here, before the crossed-book continue below,
                    # otherwise it is unreachable when all ticks are crossed and
                    # bitso.age() grows indefinitely (the v3.2/v3.5 failure mode).
                    # Only activates after 15s of connection age to allow the
                    # initial snapshot to populate.
                    #
                    # TWO-SPEED THRESHOLD:
                    # When IN_POSITION: 8s max stale. A 30s blind window while
                    # holding a position means a stop loss at -8 bps becomes -16 bps
                    # because handle_exit cannot fire without valid ticks. Reconnect
                    # fast to protect open positions.
                    # When FLAT: 30s max stale. No position at risk, so we can
                    # tolerate longer quiet periods without unnecessary reconnects
                    # that previously caused orphan events.
                    stale_threshold = 8.0 if risk.in_position() else STALE_RECONNECT_SEC
                    if (state.bitso.age() > stale_threshold
                            and time.time() - connect_ts > 15.0):
                        log.warning(
                            "[Bitso] No valid tick for %.0fs (in_position=%s) — reconnecting.",
                            state.bitso.age(), risk.in_position(),
                        )
                        break

                    bb, ba = max(bids), min(asks)

                    # Crossed book: normal Bitso feed behavior.
                    # An incremental update removes the current best level before
                    # the replacement arrives. Skip this tick silently.
                    # Dict state stays intact; next message resolves the cross.
                    if bb >= ba:
                        continue

                    state.update_bitso_top(bb, ba)
                    await handle_exit(state, risk, pnl)

                    if (not risk.in_position()
                            and not risk.kill_switch
                            and state.feeds_healthy()
                            and state.feed_quality_ok()):
                        direction = evaluate_signal(state)
                        if direction:
                            await handle_entry(direction, state, risk, pnl)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Bitso] %s - retry in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ──────────────────────────────────────────────────────────────────
# MONITOR
# ──────────────────────────────────────────────────────────────────

async def monitor_loop(state: MarketState, risk: RiskState, pnl: PnLTracker):
    start_ts        = time.time()
    last_report_ts  = time.time()
    report_interval = TELEGRAM_REPORT_HOURS * 3600
    kill_alerted    = False

    while True:
        await asyncio.sleep(60)
        runtime_hr = (time.time() - start_ts) / 3600

        if risk.kill_switch and not kill_alerted:
            kill_alerted = True
            await tg(
                f"KILL SWITCH [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                f"Daily PnL: ${pnl.daily_pnl_usd:+.4f}  Limit: -${MAX_DAILY_LOSS_USD:.2f}\n"
                f"Trades: {pnl.n_trades}  System halted."
            )

        log.info(
            "STATUS [%s] %s %.1fh  trades=%d win=%.0f%% avg=%+.3fbps "
            "daily=$%+.4f  BN=%.1fs CB=%.1fs BT=%.1fs  "
            "spread=%.2fbps  exit_att=%d",
            EXEC_MODE.upper(), ASSET.upper(), runtime_hr,
            pnl.n_trades, pnl.win_rate * 100, pnl.avg_pnl_bps,
            pnl.daily_pnl_usd,
            state.binance.age(), state.coinbase.age(), state.bitso.age(),
            state.bitso_spread_bps, risk.exit_attempt,
        )

        if ENABLE_TELEGRAM and (time.time() - last_report_ts) >= report_interval:
            last_report_ts = time.time()
            await tg(pnl.summary_text(EXEC_MODE, runtime_hr))


# ──────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ──────────────────────────────────────────────────────────────────

async def startup_checks() -> bool:
    global _BITSO_API_KEY, _BITSO_API_SECRET

    if EXEC_MODE != "live":
        log.info("Paper mode: skipping credential check.")
        return True

    if not _BITSO_API_KEY or not _BITSO_API_SECRET:
        log.info("Credentials not in env - trying SSM...")
        try:
            import boto3
            ssm               = boto3.client("ssm", region_name=REGION)
            _BITSO_API_KEY    = ssm.get_parameter(
                Name="/bot/bitso/api_key",    WithDecryption=True)["Parameter"]["Value"]
            _BITSO_API_SECRET = ssm.get_parameter(
                Name="/bot/bitso/api_secret", WithDecryption=True)["Parameter"]["Value"]
            log.info("Bitso credentials loaded from SSM.")
        except Exception as e:
            log.error("SSM load failed: %s", e)
            return False

    bal = await _check_balance()
    if not bal.get("success"):
        log.error("Balance check failed: %s", bal.get("error"))
        return False

    log.info(
        "Bitso balance: %s=%.6f  USD=$%.2f  (BTC=%.6f ETH=%.6f SOL=%.4f)",
        ASSET.upper(), bal[ASSET], bal["usd"],
        bal["btc"], bal["eth"], bal["sol"],
    )

    # Block only if no USD AND no asset to continue with
    if bal["usd"] < 5.0 and bal.get(ASSET, 0.0) < MIN_TRADE_SIZE:
        log.error("USD=$%.2f and no %s in account. Deposit funds.",
                  bal["usd"], ASSET.upper())
        return False

    # Warn about orphan from prior session - reconciler will sell it
    if bal.get(ASSET, 0.0) > MIN_TRADE_SIZE:
        log.warning(
            "STARTUP: %.8f %s already in account (prior session orphan). "
            "Reconciler will sell within %ds.",
            bal[ASSET], ASSET.upper(), int(RECONCILE_SEC),
        )
        await tg(
            f"STARTUP WARNING [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
            f"{bal[ASSET]:.8f} {ASSET.upper()} in account from prior session.\n"
            f"Reconciler auto-sells within {int(RECONCILE_SEC)}s of first price tick."
        )

    # Cancel any open orders from prior sessions.
    # Partially filled exit orders left open drain available USD balance
    # and can fill at stale prices. Cancel all on startup to start clean.
    try:
        import aiohttp
        path    = f"/v3/open_orders/?book={BITSO_BOOK}"
        headers = _bitso_headers("GET", path)
        async with aiohttp.ClientSession() as s:
            async with s.get(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                data = await r.json()
        if data.get("success"):
            open_orders = data.get("payload", [])
            if open_orders:
                log.warning(
                    "STARTUP: %d open order(s) found from prior session. Cancelling.",
                    len(open_orders),
                )
                for o in open_orders:
                    oid = o.get("oid", "")
                    if oid:
                        cancelled = await _cancel_order(oid)
                        log.warning(
                            "STARTUP: cancelled stale order oid=%s side=%s status=%s ok=%s",
                            oid, o.get("side"), o.get("status"), cancelled,
                        )
            else:
                log.info("STARTUP: no open orders from prior session.")
    except Exception as e:
        log.warning("STARTUP: could not check open orders: %s", e)

    return True


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 66)
    log.info("Bitso Lead-Lag Trader v4.5.8  |  %s  |  %s",
             ASSET.upper(), EXEC_MODE.upper())
    log.info("Book: %s  Binance: %s  Coinbase: %s",
             BITSO_BOOK, BINANCE_SYMBOL, COINBASE_SYMBOL)
    log.info("Threshold: %.1fbps  Window: %.1fs  Size: %.6f %s",
             ENTRY_THRESHOLD_BPS, SIGNAL_WINDOW_SEC, MAX_POS_ASSET, ASSET.upper())
    log.info("Stop: %.1fbps  Hold: %.1fs  Cooldown: %.1fs  Spread max: %.1fbps",
             STOP_LOSS_BPS, HOLD_SEC, COOLDOWN_SEC, SPREAD_MAX_BPS)
    log.info("Force close: %.2f%%  Reconcile: %.0fs  Chase: %.1fs  Entry slippage: %d ticks",
             FORCE_CLOSE_SLIPPAGE * 100, RECONCILE_SEC, EXIT_CHASE_SEC, ENTRY_SLIPPAGE_TICKS)
    log.info("Daily limit: $%.2f  Combined: %s  Trade log: %s",
             MAX_DAILY_LOSS_USD, COMBINED_SIGNAL, TRADE_LOG)
    log.info("=" * 66)

    ok = await startup_checks()
    if not ok:
        return

    state = MarketState()
    risk  = RiskState()
    pnl   = PnLTracker()

    await tg(
        f"Bitso Lead-Lag v4.5.8 [{EXEC_MODE.upper()}] {ASSET.upper()} started\n"
        f"Book: {BITSO_BOOK}  Threshold: {ENTRY_THRESHOLD_BPS}bps  Window: {SIGNAL_WINDOW_SEC}s\n"
        f"Size: {MAX_POS_ASSET} {ASSET.upper()}  Limit: ${MAX_DAILY_LOSS_USD}\n"
        f"Force close: {FORCE_CLOSE_SLIPPAGE*100:.1f}%  Reconciler: {int(RECONCILE_SEC)}s"
    )

    tasks = [
        asyncio.create_task(binance_feed(state),                    name="binance"),
        asyncio.create_task(coinbase_feed(state),                   name="coinbase"),
        asyncio.create_task(bitso_feed(state, risk, pnl),           name="bitso"),
        asyncio.create_task(user_trades_feed(state, risk, pnl),       name="user_trades"),
        asyncio.create_task(reconciler_loop(state, risk, pnl),      name="reconciler"),
        asyncio.create_task(monitor_loop(state, risk, pnl),         name="monitor"),
    ]

    log.info("Warming up feeds (10s)...")
    await asyncio.sleep(10)
    log.info("Signal evaluation active.")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        log.info("Shutdown.\n%s", pnl.summary_text(EXEC_MODE, 0))
        _send_telegram_sync(
            f"Bitso Lead-Lag v4.5.8 STOPPED [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
            + pnl.summary_text(EXEC_MODE, 0)
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
