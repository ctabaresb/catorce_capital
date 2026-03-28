#!/usr/bin/env python3
"""
live_trader.py  v4.5.22  — REST-only book, diff-orders as trigger only

CHANGES v4.5.21 -> v4.5.22  (eliminate book corruption permanently)

  THE PROBLEM (confirmed from v4.5.21 session, March 26):
    REST seeded 50 aggregated levels correctly. But within minutes,
    diff-orders corrupted the book. SOL trade at 13:59 showed 10.9 bps
    gap between local ask and actual fill. XRP showed 0.07 bps phantom
    spread again at 14:00.

    Root cause: diff-orders sends INDIVIDUAL order changes (add/remove
    one order at a price). Our dict uses price as key with aggregated
    size. When diff-orders removes one order at $87.850, we pop the
    entire $87.850 level even though 494 XRP of other orders remain.
    This is fundamentally incompatible. No amount of sequence filtering
    or refresh frequency can fix it.

  THE FIX:
    1. REST is the SOLE source of truth for bids/asks. Polled every 5s.
       diff-orders NEVER touches bids/asks dicts.
    2. diff-orders used only as trigger for handle_exit/handle_entry
       and stale guard timer reset.
    3. REST refresh interval reduced from 30s to 5s.
       Cost: 12 calls/min/asset = 24/min total. Limit is 60/min.

  Max book staleness: 5 seconds (was: unbounded due to corruption).
  Entry fill gap should now be < 2 bps consistently.

CHANGES v4.5.20 -> v4.5.21  (periodic REST refresh + aggregate fix)

  THE PROBLEM (confirmed from March 25-26, 12 trades total, 2 wins):
    v4.5.20 seeded the REST book on every reconnect. Trade 1 had 0.1 bps
    entry gap (accurate). But trades 2-5 had 10-20 bps gaps again.

    Two causes identified:

    1. aggregate=false in REST URL returned individual orders (1,126 asks),
       not price levels (50 asks). Multiple orders at the same price
       overwrote each other in our dict. Diff-orders (which sends level
       aggregates) conflicted with the individual-order REST data.
       Book integrity degraded within minutes of each seed.

    2. REST book only re-seeded on WebSocket reconnect (~every 3 minutes).
       Between reconnects, diff-orders messages are lost (301 reconnects
       in 15h confirms unreliable channel). The book drifted 10-20 bps
       from reality within minutes, creating phantom signals.

  THE FIX:
    1. Removed aggregate=false from REST URL. Now uses default (aggregate=true).
       Returns ~50 price levels per side with correct aggregated amounts.
       Matches how our bids/asks dict works with diff-orders updates.

    2. Added periodic REST book refresh every 30 seconds inside bitso_feed.
       On every incoming message, checks if 30s have passed since last
       refresh. If so, re-fetches REST book and replaces local bids/asks.
       Book drift is now capped at 30 seconds, not 3 minutes.
       Cost: 1 public REST call per 30s per asset = 2/min for XRP+SOL.
       Public rate limit is 60/min. Safe.

  Expected outcome:
    Entry fill gap (fallback vs corrected) should be < 2 bps consistently.
    Signals fire on REAL divergences, not phantom ones.
    Win rate should approach the research 60-62%.

CHANGES v4.5.19 -> v4.5.20  (book accuracy — the REAL root cause of live losses)

  THE PROBLEM (confirmed from March 25 live session, 4 trades, 0% win rate):
    After every WebSocket reconnect, bids/asks dicts were cleared and rebuilt
    solely from diff-orders messages. But diff-orders only sends CHANGES that
    occur AFTER the connection. All orders already resting on the book are
    invisible to the local book reconstruction.

    Result: local book had 3-10 levels while real Bitso book had 50.
    Local best ask was WRONG by 7-18 bps on every single trade.

    Trade-by-trade evidence:
      Trade 1: local ask $1.40921, real fill $1.41135 = 15.2 bps gap
      Trade 2: local ask $1.41520, real fill $1.41622 =  7.2 bps gap
      Trade 3: local ask $1.41092, real fill $1.41220 =  9.1 bps gap
      Trade 4: local ask $1.41357, real fill $1.41615 = 18.2 bps gap

    The signal computed divergence using the WRONG local mid. A true 2-3 bps
    divergence appeared as 8-10 bps because the local Bitso mid was stale.
    The strategy entered on PHANTOM signals. Every entry started 7-18 bps
    underwater. Stop loss fired on 3 of 4 trades within 2 seconds.

    86 stale reconnects in 4.3 hours (one every 3 minutes) meant the local
    book was degraded most of the session.

    App data confirms: all 4 entries filled at a SINGLE price (no multi-level
    sweep). The depth was there (495 XRP at best ask). The order did not
    sweep the book. Our local book was simply wrong about where the ask was.

  THE FIX:
    After every WebSocket reconnect, BEFORE processing any diff-orders:
    1. Fetch full order book via GET /v3/order_book/?book={book}
    2. Seed local bids/asks dicts from REST snapshot (50 levels each side)
    3. Record snapshot sequence number
    4. Discard diff-orders with sequence <= snapshot sequence
    5. Apply subsequent diff-orders normally

    This is exactly what the Bitso docs prescribe for keeping a local book.
    The code was missing step 1-4 since v1.0.

    REST fetch adds ~300-500ms to reconnect time. Acceptable: the alternative
    is trading on a phantom book for 30-60 seconds.

CHANGES v4.5.18 -> v4.5.19  (entry latency — the real root cause of live losses)

  THE PROBLEM (traced from code analysis of entry hot path):
    Signal-to-fill latency consumed the ENTIRE 4.5s lead-lag window.
    The entry path made 3 sequential REST calls before the order reached Bitso:

      T+0.0s   Signal fires on Bitso WS tick
      T+1.5s   Orphan guard: await _check_balance() — new TCP+TLS + REST RT
      T+3.0s   Preflight inside _submit_market_order: _check_balance() AGAIN
      T+4.5s   POST /v3/orders/ — order finally reaches Bitso

    Research lag median: 4.5s. Execution latency: 4.5s.
    Edge remaining at fill time: ZERO.
    Bitso had already caught up to BinanceUS/Coinbase by T+4.5s.
    Every entry was at the NEW price, not the OLD price.
    Research entered at bt_ask[signal_time] with zero latency.
    Live entered at bt_ask[signal_time + 4.5s] — completely different price.

    Additionally, every REST call created a NEW aiohttp.ClientSession(),
    adding ~50-200ms TCP+TLS handshake overhead per call.

  THE FIX (four changes):
    1. REMOVED orphan guard _check_balance() from handle_entry hot path.
       Saves ~1.5s per entry. Reconciler (30s cycle) already catches orphans.
       120s cooldown + 5s settlement buffer = 4+ reconciler cycles before
       any new entry is possible. Risk of doubled position: near zero.

    2. REMOVED _check_balance() from _submit_market_order().
       Saves ~1.5s per entry. Bitso rejects insufficient balance orders
       with error 0379. The caller handles rejection in 0ms vs pre-checking
       in 1500ms. For sells, Bitso rejects with no_{asset}_to_sell.

    3. PERSISTENT aiohttp.ClientSession created once, reused for ALL calls.
       TCP keep-alive eliminates repeated TLS handshakes.
       Saves ~50-200ms per REST call. Module-level _get_session() helper.

    4. MOVED aiohttp import to module level (minor, ~5ms per call saved).

  NEW ENTRY HOT PATH:
      T+0.0s   Signal fires
      T+0.5s   POST /v3/orders/ on persistent TCP connection
      T+0.5s   Order fills on Bitso

    Lag window remaining: 4.0s of 4.5s = 89% of edge preserved.
    Previous: 0.0s of 4.5s = 0% of edge preserved.

CHANGES v4.5.17 -> v4.5.18  (non-blocking entry fill price — critical P&L fix)

  THE PROBLEM (confirmed from March 25 live session, 5 trades, 0% win rate):
    handle_entry() awaited _fetch_fill_price_from_user_trades inline (1-3s).
    handle_entry and handle_exit run in the SAME bitso_feed coroutine.
    While handle_entry was blocked in the fetch, bitso_feed was suspended,
    NO new WebSocket messages were processed, and handle_exit COULD NOT FIRE.
    The stop loss was completely blind for 1-3 seconds after every entry.

    Trade 3: entry $1.42089, exit $1.41706 = -26.96 bps (0s logged hold)
    Trade 4: entry $1.43229, exit $1.42985 = -17.04 bps (0s logged hold)
    Trade 5: entry $1.42509, exit $1.42187 = -22.60 bps (0s logged hold)

    All three: position registered AFTER fetch returned. By the time
    handle_exit fired on the next tick, price had already dropped 17-27 bps.
    Stop loss at -15 bps triggered immediately on first tick = instant loss.

  THE FIX (two changes):
    1. Register position IMMEDIATELY after market order confirms, BEFORE
       any fill price fetch. entry_mid = state.bitso_ask (conservative).
       entry_ts = time.time() at order confirmation, not after fetch.

    2. Fetch actual fill price in a BACKGROUND asyncio task via
       asyncio.create_task(_update_entry_fill_price(...)). handle_entry
       returns immediately (~0.1s). bitso_feed processes the next tick.
       handle_exit fires with stop loss active. Background task corrects
       entry_mid when fill price arrives (1-3s later).

    Safety guards in background task:
      - Re-checks risk.in_position() and risk.entry_oid before updating
      - If stop loss fired during fetch, logs and discards the result
      - All exceptions caught — never crashes the event loop

  Expected outcome:
    - Stop loss protection active from first tick after entry (~0.1s)
    - Previous blind window of 1-3s eliminated
    - Trades 3,4,5 would have been protected at -15 bps instead of -17/-23/-27
    - entry_ts accuracy restored (accurate hold times in logs)

CHANGES v4.5.17 -> v4.5.17  (fill price accuracy — confirmed against Bitso trade history)

  THE PROBLEM (confirmed with real trade data):
    Live trade comparison showed both entry AND exit fill prices were wrong,
    turning actual losses into reported wins in the P&L log.

    Root cause: GET /v3/order_trades/?oid=XXX returns empty for MARKET orders
    on Bitso in the vast majority of calls. This makes every fallback the
    operative path — and both fallbacks are systematically biased:

    Entry fallback (signal-time mid): Market BUY sweeps the ask book, filling
    at prices $0.001-$0.003 ABOVE the mid at signal time. Fallback understates
    entry cost by 10-15 bps on every trade.

    Exit fallback (state.bitso_bid at detection time): Market SELL sweeps bids
    at the depressed price. 3 seconds later when the poller fires, the book
    has recovered and state.bitso_bid is HIGHER than the actual fill. Fallback
    overstates exit proceeds by 2-7 bps.

    Combined per-trade overstatement confirmed: +12 to +19 bps.
    Actual losses were recorded as significant wins.

  THE FIX: Replace order_trades with user_trades in both price fetch functions.

  Change 1 — _fetch_fill_price_from_user_trades() (new shared helper):
    GET /v3/user_trades/?book={BITSO_BOOK}&limit=10
    This endpoint is ALWAYS immediately available — no timing/delay issues.
    Filter by OID to match the exact order. Compute weighted avg fill price.
    Used for both entry and exit price fetches.

  Change 2 — handle_entry(): entry fill price now uses user_trades.
    Same 3-retry logic, but hitting user_trades instead of order_trades.
    If all retries fail, logs a WARNING (not debug) so the issue is visible.
    Fallback: state.bitso_ask (ask at signal time — correct direction for buys).

  Change 3 — _fetch_exit_fill_price(): exit fill price now uses user_trades.
    Same change: user_trades instead of order_trades.
    Fallback: fallback_px passed by caller (state.bitso_bid — correct for sells).

  Expected outcome: logged P&L matches Bitso account balance change exactly.

CHANGES v4.5.15 -> v4.5.17  (167h research findings applied)

  Change 1 NEW — ENTRY_MAX_BPS signal ceiling (critical, research-proven).
    167h research (master_leadlag_research.py v2.0) showed:
      Signal 7-9 bps:   +0.720 bps avg  54% win  ← good
      Signal 9-12 bps:  +0.664 bps avg  57% win  ← good
      Signal 12-16 bps: -0.201 bps avg  58% win  ← NEGATIVE
      Signal > 16 bps:  -5.102 bps avg  62% win  ← CATASTROPHIC
    Very large signals occur when BTC/Binance spikes while XRP is
    decoupled and falling. The divergence is huge but XRP never follows.
    Confirmed cause of -45, -37, -30 bps trades in 86-trade session.
    Fix: ENTRY_MAX_BPS (default 12.0) blocks entries when best > threshold.
    One line in evaluate_signal() after the existing threshold check.

  Change 2 NEW — TIME STOP exits directly as market order (not passive limit).
    Research showed the passive limit on time stop NEVER fills in practice:
      - 100% market fallback rate across all 86 live trades
      - 10s passive wait costs 0.055 bps per trade (passive timing analysis)
      - More importantly: passive limit submission causes reconciler race condition
        (Bug 3 below) and adds code complexity with zero benefit
    Fix: TIME STOP now submits direct market order immediately, same as STOP LOSS.
    EXIT_CHASE_SEC is retained only for the attempt-2 timeout detection.

  Bug 3 FIXED — Reconciler double-exit race condition.
    Root cause: reconciler and order poller run as concurrent asyncio tasks.
    Both can detect a fill simultaneously. Poller resets position first
    (entry_direction="none", exit_attempt=0). Reconciler had already read the
    pre-reset state (IN_POSITION, exit_attempt>0) from _check_balance(), then
    proceeds to call _reset_position() again with stale state.
    Result: second EXIT RECORDED with direction="none", $0 USD, corrupted
    avg_pnl_bps. Observed 3 times in 86 trades. Not a financial bug but
    inflates trade count and corrupts performance metrics.
    Fix: add `if not risk.in_position(): continue` guard in Case B
    immediately before calling _reset_position(). If poller already reset
    the position during the async order status check, skip the reconciler reset.

  Bug 4 FIXED — NameError: current_mid undefined in reconciler_loop.
    reconciler_loop used `current_mid` in the _fetch_exit_fill_price fallback
    (line ~2024). `current_mid` is only defined in handle_exit scope — it is
    NEVER defined in reconciler_loop. This would crash with NameError the first
    time exit_passive_px == 0 and the reconciler reaches Case B.
    With Change 2 (passive limit removed), exit_passive_px will always be 0,
    making this crash guaranteed on every reconcile_silent_fill path.
    Fix: compute fallback as (state.bitso_bid + state.bitso_ask) / 2 inline,
    with state.bitso_bid preferred (market sell fills at bid).

  Research parameters confirmed (set via env vars, not code):
    SIGNAL_WINDOW_SEC = 10.0  (research IC optimal at 10s, not 15s — was wrong)
    SPREAD_MAX_BPS    = 5.0   (Section 3: spread≤5.0 → +2.587bps vs +0.677bps all)
    ENTRY_MAX_BPS     = 12.0  (new — Section 8d)
    STOP_LOSS_BPS     = 15.0  (SL sweep: 15bps → $2.18/day vs 10bps → $2.08/day)
    HOLD_SEC          = 60.0  (confirmed optimal, hold time sweep peak)
    MAX_POS_ASSET     = 300   (start at 300 XRP/$435, validate 20 trades first)

CHANGES v4.5.14 -> v4.5.15  (circuit breaker self-perpetuation fix + entry log fix)

  Bug FIXED (CRITICAL): circuit breaker re-fired every ~58 minutes indefinitely.
    Root cause: pause detection computed pause_until from risk.last_exit_ts,
    which the CB itself had just set. After the pause expired, consecutive_losses
    was still >= MAX (no trades during pause). The CB re-fired, advanced
    last_exit_ts again, and the cycle repeated forever.
    Observed: CB fired at 09:13, 11:03, 12:05, 13:20, 14:20 — session froze
    after 4 trades (05:03-05:35) and never traded again for 9+ hours.

    Fix: add risk.cb_pause_until (absolute timestamp) to RiskState.
    CB fires → risk.cb_pause_until = time.time() + CONSECUTIVE_LOSS_PAUSE.
    Check: if cb_pause_until > now → return (paused). No computation from
    last_exit_ts. After pause expires (cb_pause_until <= now, cb_pause_until > 0):
    reset cb_pause_until = 0.0 and fall through for one trade.
    If that trade wins: consecutive_losses resets to 0, CB does not re-fire.
    If that trade loses: consecutive_losses still >= MAX, cb_pause_until == 0,
    CB fires again on the NEXT signal (after the trade exits). Correct behavior.
    cb_pause_until is NOT reset in _reset_position — it survives position
    resets so the pause stays active across multiple position cycles.

  Bug FIXED (DISPLAY): ENTRY log showed raw returns, not divergences.
    bn=%+.2f was logging bn_ret (raw Binance return over window).
    The COMBINED signal uses bn_div = bn_ret - bt_ret. With bt_ret=-2.64,
    a trade logged as bn=+4.81 actually had bn_div=+7.45 — above threshold.
    Caused confusion reading logs (appeared to show sub-threshold entries).
    Fix: compute bn_div/cb_div in handle_entry and log those explicitly.

  Fix: version string in summary_text updated to v4.5.15.

CHANGES v4.5.13 -> v4.5.14  (exit price precision + log price decimals)

  Bug FIXED (CRITICAL): order_trades API returns empty for fast market sells.
    Bitso clears market sell trade records within 1-2s of execution.
    By the time the poller detects the fill (~2-3s later), order_trades is empty.
    Fallback was state.bitso_bid — but bid at detection time can be stale/wrong.

    Trade 2 example (confirmed from app vs logs):
      App sell price:  $1.45433
      Fallback bid:    $1.45156  (stale, 19 bps below actual)
      Logged P&L:      -20.75 bps  (real was -1.72 bps)

    Fix: store risk.exit_passive_px = passive_px when passive limit submitted.
    This is the price the passive limit was placed at (= current bid at T+60s).
    Use as fallback instead of current state.bitso_bid when order_trades is empty.
    The market sell fallback fires close to this price level.
    Also: 3 retry attempts on order_trades with 500ms delay before giving up,
    giving Bitso time to write the trade record.

  Fix 2: All price log formats changed from %.2f to %.5f.
    XRP price is $1.45xxx. At %.2f resolution, $1.45293 displays as $1.45 —
    impossible to verify fills from logs. %.5f gives full XRP tick precision.
    Affects: ENTRY mid, fill price, EXIT FILL CONFIRMED, WS FILL, deferred exit,
    ORDER limit price, poller fallback, PREFLIGHT, balance, ORPHAN sell price.

CHANGES v4.5.12 -> v4.5.13  (exit price accuracy + circuit breaker)

  Bug 1 FIXED (CRITICAL): reconcile_silent_fill used current_mid as exit price.
    Market sell fills at BID, not mid. Every reconcile_silent_fill exit overstated
    P&L by spread/2 (~1.59 bps for XRP). Root cause of balance vs logged P&L gap.
    Fix: try order_trades API for actual fill price, fallback to state.bitso_bid.

  Bug 2 FIXED (CRITICAL): market_timeout used current_mid as exit price.
    Same problem: market sell fills at bid. Fires when market order detection
    takes > EXIT_CHASE_SEC*2. Fix: same as Bug 1.

  Bug 3 FIXED (MODERATE): passive limit silent fill used current_mid.
    Passive limit filled at bid (the price it was placed at). current_mid is
    always higher by spread/2. Fix: try order_trades, fallback to bitso_bid.

  Bug 4 FIXED (MODERATE): partial_limit_fully_filled used mid.
    Same category. Fix: try order_trades, fallback to bitso_bid.

  Bug 5 NEW: consecutive loss circuit breaker.
    After CONSECUTIVE_LOSS_MAX (default 3) consecutive losses, block new entries
    for CONSECUTIVE_LOSS_PAUSE (default 1800s = 30 min). Prevents cascade losses
    during directional market crashes (confirmed cause of XRP session losses).
    Configurable via env vars. Resets automatically when pause expires.

  Bug 6 FIXED (MINOR): dead code removed.
    tick = 0.01 in handle_entry and handle_exit was hardcoded for BTC (XRP tick
    is 0.00001). aggressive_px computed in handle_exit was never used. Removed.

  Research finding applied: ENTRY_THRESHOLD_BPS=7.0 is optimal for XRP.
    Research (master_leadlag_research.py, 50h XRP data) showed:
      5bps: net +1.228 bps, $7.65/day live
      7bps: net +2.926 bps, $8.61/day live
    7bps gives 2.4× better net per trade at 47% of signal frequency.
    Set ENTRY_THRESHOLD_BPS=7.0 in launch command for XRP sessions.

CHANGES v4.5.11 -> v4.5.12  (minimum spread guard)

  Bug FIXED: entries fired on partially-populated Bitso order book.
    Root cause: on WebSocket reconnect, bids/asks dicts are cleared.
    diff-orders messages repopulate the book one side at a time.
    The crossed-book guard (bb >= ba) catches bid > ask but not bid ≈ ask.
    During the first 1-2 seconds after reconnect, best_bid and best_ask
    can be artificially close (e.g. spread = 0.07-0.28 bps) because
    only one side of the book has populated so far.
    Paper session confirmed: 3 of 19 entries fired at spread < 0.5 bps.
    In paper mode these record P&L at a fake mid price.
    In live mode the market order would fill at the real ask, which
    could be 5-10 bps from the fake mid — guaranteed losing trade.

  Fix: SPREAD_MIN_BPS env var (default 0.5 bps).
    evaluate_signal() blocks entries when spread < SPREAD_MIN_BPS.
    Consistent with all other parameters: env var with sensible default,
    logged at startup, tunable per asset without code changes.
    Set SPREAD_MIN_BPS=0.0 to disable if ever needed.

  No other logic changes. Paper and live modes both protected.

CHANGES v4.5.10 -> v4.5.11  (multi-asset correctness fixes)

  Bug 1 FIXED (CRITICAL — live mode): _check_balance hardcoded btc/eth/sol only.
    When ASSET=xrp, bal.get("xrp") returned 0.0 on every call, causing:
    - Startup KeyError crash on bal["xrp"] at line 2229
    - Orphan guard never fired (saw 0 balance)
    - Reconciler Case A never triggered (saw 0 balance)
    - All market sell preflights rejected ("no_asset_to_sell")
    Fix: added xrp/ada/doge/xlm/hbar/dot to the return dict.
    Also added generic ASSET key as permanent future-proof fallback.

  Bug 2 FIXED (MODERATE — live mode): _NO_ASSET_ERRORS missing altcoin codes.
    If Bitso returns "no_xrp_to_sell" on exit, handle_exit did not recognize
    it as a no-asset signal, leaving the position stuck IN_POSITION until
    the reconciler rescued it 30s later.
    Fix: added no_{asset}_to_sell and insufficient_{asset} for all 6 new assets.

  Bug 4 FIXED (MINOR): MIN_TRADE_SIZE fallback was 0.00001 for unknown assets.
    Bitso minimum for XRP is 0.03, ADA 0.04, DOGE 0.08. Using 0.00001 as the
    guard threshold meant sell preflights could submit dust orders Bitso rejects.
    Fix: explicit MIN_TRADE_SIZE per asset. Fallback raised to 0.01.

  No logic changes. Paper mode unaffected by all three fixes.
  These fixes are required before switching any altcoin session to EXEC_MODE=live.

CHANGES v4.5.9 -> v4.5.10  (actual entry fill price — critical P&L fix)

  Root cause of systematic P&L overstatement discovered by comparing
  Bitso API user_trades against our recorded EXIT RECORDED values.

  Bug: entry_mid = (bid+ask)/2 at SIGNAL TIME, not actual fill price.
  Market orders fill at the ASK, which moves between signal evaluation
  and order execution. The difference can be $30-60/BTC = 4-8 bps.

  Effect: every trade P&L was overstated by (ask_fill - mid_at_signal).
  Example: buy=74296, sell=74254 → true=-5.65 bps, reported=+2.09 bps.
  A clear LOSS was recorded as a WIN.

  Fix: after market entry order submits, fetch actual weighted average
  fill price from GET /v3/order_trades/?oid={entry_oid}.
  Set risk.entry_mid = actual_fill_px (not signal-time mid).
  Fallback: signal-time mid if API call fails (same behavior as before).

  This also corrects stop_loss trigger: pnl_bps_live was computed from
  mid instead of actual cost, meaning stop losses fired at wrong threshold.

CHANGES v4.5.8 -> v4.5.9  (orphan settlement buffer)

  Root cause of orphans (9 in 24 trades = 38% of trades):
  Passive limit partially fills across multiple poll cycles.
  Poller cancels remainder, market order fills the rest.
  _reset_position called: last_exit_ts = time.time().
  At T+8s (COOLDOWN_SEC), orphan guard calls _check_balance().
  Bitso balance shows 0 BTC — orphan guard passes, entry fires.
  At T+9-12s: passive limit partial fill BTC settles on Bitso.
  Reconciler sees BTC in account with internal=FLAT → ORPHAN.

  Fix: last_exit_ts = time.time() + 5.0
  This extends the effective cooldown from 8s to 13s.
  Bitso partial fill settlement always completes within 5-8s.
  At T+13s: all settlements done, balance check is accurate.
  No orphan. One-line change, zero impact on any other logic.

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
import aiohttp

# ── Persistent HTTP session (v4.5.19) ────────────────────────────
# In v4.5.17, every REST call created a new aiohttp.ClientSession(),
# which means a fresh TCP connection + TLS handshake each time
# (~50-200ms overhead per call). On the entry hot path this happened
# 3 times in sequence: orphan guard + preflight + order submission.
# Total overhead: ~150-600ms just in session setup, on top of the
# actual Bitso API latency.
#
# Fix: one persistent session created in main(), reused for ALL
# REST calls. TCP keep-alive eliminates repeated handshakes.
# The session is stored in a module-level variable.
_http_session: Optional[aiohttp.ClientSession] = None

def _get_session() -> aiohttp.ClientSession:
    """Return the persistent HTTP session. Creates one if needed."""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5),
        )
    return _http_session

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
SPREAD_MIN_BPS       = float(os.environ.get("SPREAD_MIN_BPS",       "0.5"))  # blocks post-reconnect partial-book entries
EXIT_CHASE_SEC       = float(os.environ.get("EXIT_CHASE_SEC",       "8.0"))
FORCE_CLOSE_SLIPPAGE = float(os.environ.get("FORCE_CLOSE_SLIPPAGE", "0.005"))
RECONCILE_SEC        = float(os.environ.get("RECONCILE_SEC",        "30.0"))
ENTRY_SLIPPAGE_TICKS = int(os.environ.get("ENTRY_SLIPPAGE_TICKS",   "2"))    # ticks above ask on buy entry
STALE_RECONNECT_SEC  = float(os.environ.get("STALE_RECONNECT_SEC",  "30.0")) # seconds of no valid Bitso tick before reconnect
POST_RESET_COOLDOWN  = float(os.environ.get("POST_RESET_COOLDOWN",  "20.0")) # seconds to block entries after any reset (Bitso balance settlement)
# Consecutive loss circuit breaker: after N consecutive losses, pause trading.
# Prevents cascade losses during directional market crashes (confirmed XRP issue).
CONSECUTIVE_LOSS_MAX   = int(os.environ.get("CONSECUTIVE_LOSS_MAX",   "3"))
CONSECUTIVE_LOSS_PAUSE = float(os.environ.get("CONSECUTIVE_LOSS_PAUSE", "1800.0"))  # 30 min
# Signal ceiling: block entries when divergence is too large.
# Research (167h): signals >12 bps produce -5.1 bps avg — XRP decoupled from BTC.
# These fire when BTC spikes violently while XRP is stuck/falling.
ENTRY_MAX_BPS = float(os.environ.get("ENTRY_MAX_BPS", "12.0"))

_MIN_SIZES     = {
    "btc":  0.00001,
    "eth":  0.0001,
    "sol":  0.001,
    "xrp":  0.03,
    "ada":  0.04,
    "doge": 0.08,
    "xlm":  0.1,
    "hbar": 0.1,
    "dot":  0.01,
}
MIN_TRADE_SIZE = _MIN_SIZES.get(ASSET, 0.01)

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
    # XRP
    "no_xrp_to_sell",
    "insufficient_xrp",
    # ADA
    "no_ada_to_sell",
    "insufficient_ada",
    # DOGE
    "no_doge_to_sell",
    "insufficient_doge",
    # XLM
    "no_xlm_to_sell",
    "insufficient_xlm",
    # HBAR
    "no_hbar_to_sell",
    "insufficient_hbar",
    # DOT
    "no_dot_to_sell",
    "insufficient_dot",
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
    @property
    def consecutive_losses(self) -> int:
        """Count of consecutive losing trades at the tail of the trade list."""
        count = 0
        for t in reversed(self._trades):
            if t["pnl_bps"] < 0:
                count += 1
            else:
                break
        return count

    def summary_text(self, mode: str, runtime_hr: float) -> str:
        trades_hr = self.n_trades / max(runtime_hr, 0.01)
        return "\n".join([
            f"Bitso Lead-Lag v4.5.22 [{mode.upper()}] {ASSET.upper()}",
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
        # Passive exit limit price — stored when passive limit is submitted.
        # Used as fallback when order_trades API returns empty for market sells.
        # The market sell fallback fires close to this price level.
        self.exit_passive_px:    float = 0.0
        # Circuit breaker: absolute timestamp when current CB pause expires.
        # 0.0 = no active pause. Set when CB fires. NOT cleared in _reset_position
        # so the pause stays active across full position cycles during a bad streak.
        self.cb_pause_until:     float = 0.0

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
        path    = "/v3/balance/"
        headers = _bitso_headers("GET", path)
        s = _get_session()
        async with s.get(
            BITSO_API_URL + path, headers=headers,
        ) as r:
            data = await r.json()
            if data.get("success"):
                bals = {b["currency"]: b for b in data["payload"]["balances"]}
                return {
                    "success": True,
                    "usd":  float(bals.get("usd",  {}).get("available", 0)),
                    "btc":  float(bals.get("btc",  {}).get("available", 0)),
                    "eth":  float(bals.get("eth",  {}).get("available", 0)),
                    "sol":  float(bals.get("sol",  {}).get("available", 0)),
                    "xrp":  float(bals.get("xrp",  {}).get("available", 0)),
                    "ada":  float(bals.get("ada",  {}).get("available", 0)),
                    "doge": float(bals.get("doge", {}).get("available", 0)),
                    "xlm":  float(bals.get("xlm",  {}).get("available", 0)),
                    "hbar": float(bals.get("hbar", {}).get("available", 0)),
                    "dot":  float(bals.get("dot",  {}).get("available", 0)),
                    ASSET:  float(bals.get(ASSET, {}).get("available", 0)),
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
        s = _get_session()
        async with s.post(
            BITSO_API_URL + path, headers=headers, data=body,
        ) as r:
                data = await r.json()
                if data.get("success"):
                    oid = data["payload"].get("oid", "unknown")
                    log.info("ORDER %s %.8f %s @ $%.5f  oid=%s",
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

    v4.5.19: BALANCE CHECK REMOVED FROM HOT PATH.
    Previously this function called _check_balance() before every order,
    adding ~1.5s of REST latency to the entry path. On a strategy with
    4.5s median lag, this alone consumed 33% of the edge window.

    Bitso rejects orders with insufficient balance via error code 0379.
    The caller handles rejection in 0ms vs pre-checking in 1500ms.
    For SELL: Bitso rejects with no_{asset}_to_sell, handled by _NO_ASSET_ERRORS.
    """
    if EXEC_MODE != "live":
        return {"success": True, "paper": True, "oid": f"paper_{int(time.time()*1000)}"}

    try:
        path      = "/v3/orders/"
        body_dict = {
            "book":  BITSO_BOOK,
            "side":  side,
            "type":  "market",
            "major": f"{amount_asset:.8f}",
        }
        body    = json.dumps(body_dict)
        headers = _bitso_headers("POST", path, body)
        s = _get_session()
        async with s.post(
            BITSO_API_URL + path, headers=headers, data=body,
        ) as r:
            data = await r.json()
            if data.get("success"):
                oid = data["payload"].get("oid", "unknown")
                log.info("MARKET ORDER %s %.8f %s  oid=%s",
                         side.upper(), amount_asset, ASSET.upper(), oid)
                return {"success": True, "oid": oid, "amount": amount_asset}
            # Map Bitso rejection codes to our internal error strings
            err_code = data.get("error", {}).get("code", "")
            err_msg  = data.get("error", {}).get("message", "")
            if err_code in ("0379", "0343"):
                log.warning("MARKET ORDER REJECTED (balance): %s %s", err_code, err_msg)
                return {"success": False, "error": "balance_too_low"}
            log.error("MARKET ORDER REJECTED: %s", data)
            return {"success": False, "error": data}
    except Exception as e:
        log.error("MARKET ORDER EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}


async def _cancel_order(oid: str) -> bool:
    if EXEC_MODE != "live" or not oid or oid.startswith("paper_"):
        return True
    try:
        path    = f"/v3/orders/{oid}"
        headers = _bitso_headers("DELETE", path)
        s = _get_session()
        async with s.delete(
            BITSO_API_URL + path, headers=headers,
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
    # Block entries when book is partially populated after a reconnect.
    # diff-orders repopulates one side at a time — bid ≈ ask for 1-2s.
    # A market buy at that moment fills at the real ask, which is far
    # from the fake mid, guaranteeing a loss. Default guard: 0.5 bps.
    if state.bitso_spread_bps < SPREAD_MIN_BPS:
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

    # Signal ceiling: block when divergence is too large.
    # Research (167h, Section 8d): signals >12 bps are NEGATIVE (-5.1 bps avg).
    # Cause: BTC spikes violently while XRP is decoupled and falling.
    # The huge divergence never resolves — XRP stays down, we lose.
    if abs(best) > ENTRY_MAX_BPS:
        return None

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
# ENTRY FILL PRICE BACKGROUND UPDATE (v4.5.18)
# ──────────────────────────────────────────────────────────────────

async def _update_entry_fill_price(risk: RiskState, entry_oid: str, fallback_px: float):
    """
    Background task: fetch actual entry fill price and correct risk.entry_mid.

    Runs as asyncio.create_task from handle_entry so handle_entry can return
    immediately. This unblocks bitso_feed to process the next tick, allowing
    handle_exit (stop loss) to fire within milliseconds of entry.

    In v4.5.17 this was awaited inline, blocking bitso_feed for 1-3 seconds.
    During that window: no new messages processed, handle_exit never fires,
    stop loss blind. Confirmed cause of 3 catastrophic losses on March 25.

    Safety guards:
      - Re-checks risk.in_position() and risk.entry_oid before updating
      - If stop loss fired during fetch, position is already reset — skip
      - All exceptions caught and logged — never crashes the event loop
    """
    try:
        entry_fill_px = await _fetch_fill_price_from_user_trades(
            entry_oid,
            fallback_px,
            label="ENTRY",
        )
        # Only update if still in same position. If stop loss fired during
        # this fetch (now possible because bitso_feed is unblocked), the
        # position is already reset. Do NOT overwrite the cleared state.
        if risk.in_position() and risk.entry_oid == entry_oid:
            risk.entry_mid = entry_fill_px
            log.info("[ENTRY] entry_mid corrected to $%.5f (was $%.5f fallback)",
                     entry_fill_px, fallback_px)
        elif not risk.in_position():
            log.info("[ENTRY] fill price fetched ($%.5f) but position already closed "
                     "(stop loss fired during fetch). Discarding.", entry_fill_px)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("[ENTRY] background fill price fetch failed: %s — "
                    "entry_mid stays at fallback $%.5f", e, fallback_px)


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

    # CONSECUTIVE LOSS CIRCUIT BREAKER
    # After CONSECUTIVE_LOSS_MAX losses in a row, pause entries for PAUSE seconds.
    # Uses risk.cb_pause_until (absolute timestamp) — NOT derived from last_exit_ts.
    # This prevents the v4.5.14 bug where CB re-fired every ~58 min indefinitely.
    #
    # State machine:
    #   cb_pause_until == 0:  no active pause → check streak, fire if >= MAX
    #   cb_pause_until > now: pause active → block entry
    #   cb_pause_until > 0 and <= now: pause just expired → reset, allow one trade
    #     (if that trade wins: streak→0, CB quiet; if loses: CB re-fires next signal)
    now_ts = time.time()
    if risk.cb_pause_until > 0:
        if risk.cb_pause_until > now_ts:
            return   # actively paused — block entry
        # Pause has expired — reset and allow one trade through
        log.info("CIRCUIT BREAKER pause expired. Resuming trading. "
                 "consecutive_losses=%d (will re-fire if streak continues).",
                 pnl.consecutive_losses)
        risk.cb_pause_until = 0.0
        # Fall through to normal entry — do NOT re-check consecutive_losses here.
        # The next signal after this trade will re-evaluate if the streak continues.
    elif pnl.consecutive_losses >= CONSECUTIVE_LOSS_MAX:
        # Fire CB for the first time on this streak (or after streak reset via win)
        log.warning(
            "CIRCUIT BREAKER: %d consecutive losses. Pausing entries for %.0fs.",
            pnl.consecutive_losses, CONSECUTIVE_LOSS_PAUSE,
        )
        await tg(
            f"CIRCUIT BREAKER [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
            f"{pnl.consecutive_losses} consecutive losses. "
            f"Pausing {CONSECUTIVE_LOSS_PAUSE/60:.0f} min."
        )
        risk.cb_pause_until = now_ts + CONSECUTIVE_LOSS_PAUSE
        return

    # ORPHAN GUARD: removed from hot path in v4.5.19.
    # Previously called _check_balance() here, adding ~1.5s REST latency
    # to EVERY entry. On a 4.5s lag strategy, this consumed 33% of the edge.
    # The reconciler already checks for orphans every 30s. The cooldown
    # (120s + 5s settlement buffer) ensures entries only fire 125s+ after
    # last exit, giving the reconciler 4+ cycles to catch any orphan.
    # If Bitso has orphan XRP AND the reconciler missed it AND the cooldown
    # expired, the worst case is a doubled position (600 XRP = $870).
    # This is acceptable given the 120s cooldown + 30s reconciler window.

    tick        = 0.01
    entry_mid_est = (state.bitso_bid + state.bitso_ask) / 2  # estimate only
    bn_ret      = state.binance.return_bps(SIGNAL_WINDOW_SEC)  or 0.0
    cb_ret      = state.coinbase.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    bt_ret      = state.bitso.return_bps(SIGNAL_WINDOW_SEC)    or 0.0
    # Compute divergences for logging — matches what evaluate_signal() uses.
    # bn_div/cb_div are the actual values checked against ENTRY_THRESHOLD_BPS.
    # Previously logged raw returns (bn_ret), which caused confusion when
    # bt_ret was negative (making div > ret, appearing below threshold in logs).
    bn_div      = bn_ret - bt_ret
    cb_div      = cb_ret - bt_ret

    log.info("[%s] ENTRY %s (MARKET)  mid=$%.5f  spread=%.2fbps  "
             "bn_div=%+.2f cb_div=%+.2f bt=%+.2f",
             EXEC_MODE.upper(), direction.upper(), entry_mid_est,
             state.bitso_spread_bps, bn_div, cb_div, bt_ret)

    # Market order: fills immediately at best available ask.
    # v4.5.19: single REST call, no balance checks, persistent session.
    # Expected latency: ~300-700ms (was ~4500ms in v4.5.17).
    entry_submit_ts = time.time()
    result = await _submit_market_order(direction, MAX_POS_ASSET)
    entry_latency_ms = (time.time() - entry_submit_ts) * 1000
    log.info("[%s] ENTRY ORDER latency: %.0fms  ok=%s",
             EXEC_MODE.upper(), entry_latency_ms, result.get("success"))
    if not result.get("success"):
        risk.last_exit_ts = time.time()
        return

    # Use actual submitted size from preflight adjustment, not MAX_POS_ASSET.
    actual_size = result.get("amount", MAX_POS_ASSET)
    entry_oid   = result.get("oid", "")

    # ── v4.5.18 FIX: Register position IMMEDIATELY, fetch price in background ──
    #
    # TWO BUGS in v4.5.17:
    #
    # Bug A (CRITICAL): handle_entry blocked for 1-3 seconds in
    #   _fetch_fill_price_from_user_trades (3 retries x 1s delay).
    #   handle_entry and handle_exit run in the SAME bitso_feed coroutine.
    #   While handle_entry is awaiting the fetch, bitso_feed is suspended,
    #   NO new WebSocket messages are processed, and handle_exit CANNOT FIRE.
    #   The stop loss is completely blind for 1-3 seconds after every entry.
    #
    #   March 25 evidence: trades 3,4,5 show 0-1 second hold in logs because
    #   entry_ts was set AFTER the fetch. By the time handle_entry returned
    #   and the next tick fired handle_exit, price had already dropped 17-27
    #   bps. Stop loss at -15 bps fired on the FIRST tick.
    #
    # Bug B (MODERATE): entry_ts set after fetch, not at actual fill time.
    #   Hold timer started 1-3s late. Logged hold times were incorrect.
    #
    # FIX: Register position with entry_mid = bitso_ask (conservative).
    #   Return from handle_entry IMMEDIATELY. Fetch actual fill price in a
    #   background asyncio task that corrects entry_mid when done.
    #   handle_entry now returns in ~0.1s (order submission time only).
    #   The next bitso tick fires handle_exit with full stop loss protection.
    #
    #   During the background fetch (1-3s), stop loss uses bitso_ask as
    #   entry_mid. This overstates entry cost by ~half-spread (~1.5 bps),
    #   making the stop loss trigger ~1.5 bps later than ideal. Acceptable.
    entry_ask_fallback = state.bitso_ask if state.bitso_ask > 0 else entry_mid_est

    risk.position_asset  = actual_size if direction == "buy" else -actual_size
    risk.entry_mid       = entry_ask_fallback  # conservative — corrected by bg task
    risk.entry_ts        = time.time()         # hold timer starts NOW, not after fetch
    risk.entry_direction = direction
    risk.entry_oid       = entry_oid

    # Fire-and-forget background task to fetch actual fill price.
    # handle_entry returns immediately so bitso_feed can process next tick.
    if EXEC_MODE == "live" and entry_oid:
        asyncio.create_task(
            _update_entry_fill_price(risk, entry_oid, entry_ask_fallback),
            name=f"entry_fill_{entry_oid[:8]}",
        )


# ──────────────────────────────────────────────────────────────────
# EXIT CHASER
# ──────────────────────────────────────────────────────────────────

async def _fetch_fill_price_from_user_trades(oid: str, fallback_px: float,
                                              label: str = "") -> float:
    """
    Fetch actual weighted fill price for any order using GET /v3/user_trades/.

    WHY user_trades instead of order_trades:
      GET /v3/order_trades/?oid=XXX returns EMPTY for market orders in the
      vast majority of calls (confirmed in live session with real trade data).
      Both entry and exit fetches were falling back on every trade, causing
      systematic P&L overstatement of +12 to +19 bps per trade.

      Entry fallback (signal-time mid) understates cost because the market BUY
      sweeps ask levels above mid. Exit fallback (state.bitso_bid at detection
      time) overstates proceeds because the book recovers after the sell sweeps.

    GET /v3/user_trades/?book={book}&limit=10 is ALWAYS immediately available.
    Filter by OID, compute weighted avg. Handles partial fills correctly.
    Retried up to 3 times with 1s delay for any transient API lag.
    Logs at WARNING level if all retries fail so the issue is visible.
    """
    if EXEC_MODE != "live" or not oid or oid.startswith("paper_"):
        return fallback_px
    try:
        path = f"/v3/user_trades/?book={BITSO_BOOK}&limit=10"
        for attempt in range(3):
            if attempt > 0:
                await asyncio.sleep(1.0)
            hdrs = _bitso_headers("GET", path)
            s = _get_session()
            async with s.get(
                BITSO_API_URL + path, headers=hdrs,
            ) as r:
                data = await r.json()
            if data.get("success"):
                trades = [t for t in data.get("payload", []) if t.get("oid") == oid]
                if trades:
                    tv = sum(float(t["price"]) * abs(float(t["major"])) for t in trades)
                    ts = sum(abs(float(t["major"])) for t in trades)
                    if ts > 0:
                        px = tv / ts
                        log.info("[%s] %s fill price from user_trades (attempt %d): "
                                 "$%.5f  oid=%s  fills=%d",
                                 EXEC_MODE.upper(), label, attempt + 1, px, oid, len(trades))
                        return px
    except Exception as e:
        log.warning("[%s] %s user_trades fetch error oid=%s: %s — using fallback $%.5f",
                    EXEC_MODE.upper(), label, oid, e, fallback_px)
    log.warning("[%s] %s user_trades: oid=%s not found after retries — using fallback $%.5f",
                EXEC_MODE.upper(), label, oid, fallback_px)
    return fallback_px


async def _fetch_exit_fill_price(exit_oid: str, fallback_px: float) -> float:
    """Fetch actual exit fill price via user_trades. See _fetch_fill_price_from_user_trades."""
    return await _fetch_fill_price_from_user_trades(exit_oid, fallback_px, label="EXIT")

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
    risk.exit_passive_px    = 0.0
    # Settlement buffer: Bitso partial fill settlements can arrive 3-8s after
    # the final exit order fills. Without this buffer, the orphan guard balance
    # check fires at COOLDOWN_SEC (8s) and sees 0 BTC — passes — entry fires —
    # then the partial fill BTC arrives and becomes an orphan.
    # +5s buffer means orphan guard runs at T+13s, after all settlements complete.
    risk.last_exit_ts       = time.time() + 5.0
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
            path    = f"/v3/orders/{oid}"
            headers = _bitso_headers("GET", path)
            s = _get_session()
            async with s.get(
                BITSO_API_URL + path, headers=headers,
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
                    "[%s] ENTRY PARTIAL FILL: emergency sell %.8f @ $%.5f ok=%s",
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
    Exit state machine — v4.5.17 simplified.

    attempt 0  wait for time_stop or stop_loss trigger
               BOTH paths: submit direct market order immediately
               (passive limit on time stop removed — 100% fallback rate,
               never fills in practice, causes reconciler race condition)
    attempt 2  market order submitted — wait for poller/reconciler
               if poller cancels partial: resubmit market for remainder
               if timeout (2x EXIT_CHASE_SEC): force reconciler reset
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
            "[%s] WS FILL DETECTED oid=%s price=$%.5f  hold=%.1fs",
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

    if risk.entry_direction == "buy":
        pnl_bps_live  = (current_mid - risk.entry_mid) / risk.entry_mid * 10_000
        exit_side     = "sell"
        passive_px    = state.bitso_bid
        floor_px      = risk.entry_mid * (1 - STOP_LOSS_BPS / 10_000)
    else:
        pnl_bps_live  = (risk.entry_mid - current_mid) / risk.entry_mid * 10_000
        exit_side     = "buy"
        passive_px    = state.bitso_ask
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
            log.debug("[%s] EXIT deferred: bid $%.5f < floor $%.5f (hold=%.1fs)",
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

        # TIME STOP → direct market order (same as stop loss).
        # Passive limit was removed: 100% market fallback rate in 86 live trades,
        # 10s wait costs 0.055 bps/trade, and caused reconciler race condition.
        # Research hold-time sweep confirmed 60s direct exit is optimal.
        log.info("[%s] EXIT TIME STOP (MARKET): %s  pnl=%.3fbps — direct market order.",
                 EXEC_MODE.upper(), exit_side.upper(), pnl_bps_live)
        risk.last_exit_api_call = time.time()
        result = await _submit_market_order(exit_side, abs(risk.position_asset))
        if result.get("success"):
            risk.exit_oid          = result.get("oid", "")
            risk.exit_submitted_ts = time.time()
            risk.exit_attempt      = 2   # skip directly to attempt 2
            log.info("[%s] TIME STOP market order submitted oid=%s",
                     EXEC_MODE.upper(), risk.exit_oid)
        elif result.get("error") in _NO_ASSET_ERRORS:
            await _cancel_unfilled_entry(risk, state, pnl)
        return

    # ── attempt 2: market submitted, waiting for poller/reconciler ──
    # Market order was submitted. Poller checks every 2s for fill confirmation.
    # If the market exit partially filled and the poller cancelled the remainder,
    # exit_oid will be cleared (poller sets it to "" after cancel). In that case,
    # resubmit a market order for the remaining position_asset amount.
    # NOTE: attempt 1 (passive limit) removed in v4.5.17 — both exit paths
    # now go directly to market order → attempt 2.
    time_since_exit = time.time() - risk.exit_submitted_ts
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
                # Market order rejected — asset already gone (fully filled earlier).
                exit_fill_px = await _fetch_exit_fill_price(
                    risk.exit_oid,
                    state.bitso_bid if state.bitso_bid > 0
                    else (state.bitso_bid + state.bitso_ask) / 2,
                )
                _reset_position(risk, pnl, exit_fill_px, hold_sec, "market_remainder_filled")
            return

        if time_since_exit > EXIT_CHASE_SEC * 2:
            log.error(
                "[%s] EXIT market order not detected after %.0fs. "
                "Forcing reconciler reset.",
                EXEC_MODE.upper(), time_since_exit,
            )
            exit_fill_px = await _fetch_exit_fill_price(
                risk.exit_oid,
                state.bitso_bid if state.bitso_bid > 0
                else (state.bitso_bid + state.bitso_ask) / 2,
            )
            _reset_position(risk, pnl, exit_fill_px, hold_sec, "market_timeout")
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
                    chk_path    = f"/v3/orders/{risk.exit_oid}"
                    chk_headers = _bitso_headers("GET", chk_path)
                    s = _get_session()
                    async with s.get(
                        BITSO_API_URL + chk_path, headers=chk_headers,
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

            # ── DOUBLE-EXIT GUARD (v4.5.17) ──────────────────────────
            # Asyncio race condition: reconciler evaluated internal_flat=False
            # before the poller ran, then awaited the order status check.
            # During that await, the poller reset the position. Reconciler
            # resumes with stale internal_flat=False and calls _reset_position
            # on an already-flat state → direction="none" fake trade recorded.
            # Fix: re-check current position state after the await completes.
            if not risk.in_position():
                log.debug(
                    "[Reconciler] SILENT FILL: position already reset by poller. "
                    "Skipping duplicate reset."
                )
                continue

            hold_sec = time.time() - risk.entry_ts
            log.warning(
                "[Reconciler] SILENT FILL: internal=IN_POSITION, "
                "%.8f %s in account. Resetting.",
                asset_bal, ASSET.upper(),
            )
            # Fetch actual fill price. Market sell fills at bid not mid.
            # exit_passive_px is always 0 in v4.5.17 (passive limit removed).
            # Use state.bitso_bid as the most accurate available fallback.
            exit_fill_px = await _fetch_exit_fill_price(
                risk.exit_oid,
                state.bitso_bid if state.bitso_bid > 0
                else (state.bitso_bid + state.bitso_ask) / 2,
            )
            _reset_position(risk, pnl, exit_fill_px, hold_sec, "reconcile_silent_fill")
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
                path    = f"/v3/orders/{oid}"
                headers = _bitso_headers("GET", path)
                s = _get_session()
                async with s.get(
                    BITSO_API_URL + path, headers=headers,
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
                            # Use _fetch_exit_fill_price: 3 retries with 500ms
                            # delay. Bitso clears market sell records in 1-2s —
                            # a single call often returns empty. Retries catch it.
                            # Fallback = state.bitso_bid (market sells fill at bid).
                            # exit_passive_px is always 0 in v4.5.17 (no passive
                            # limit on time stops), so bid is the correct fallback.
                            fill_price = await _fetch_exit_fill_price(
                                oid,
                                state.bitso_bid if state.bitso_bid > 0
                                else (state.bitso_bid + state.bitso_ask) / 2,
                            )
                            _reset_position(risk, pnl, fill_price, hold_sec, "poller_fill")
                    continue

                order   = data.get("payload", [{}])
                # Bitso GET /v3/orders/{oid} returns payload as a list
                if isinstance(order, list):
                    order = order[0] if order else {}
                status  = order.get("status", "")

                if status == "completed":
                    # Use _fetch_exit_fill_price: 3 retries + 500ms delay.
                    # Market orders return price=0 in the order payload.
                    # order_trades gives actual weighted fill price.
                    # Bitso clears records in 1-2s — retries are critical.
                    # If order has a price field (limit order), use it directly.
                    # For market orders, _fetch_exit_fill_price handles retries.
                    raw_price = float(order.get("price", 0))
                    if raw_price > 0:
                        fill_price = raw_price  # limit order — price is reliable
                    else:
                        # Market order: use helper with 3 retries + bid fallback
                        fill_price = await _fetch_exit_fill_price(
                            oid,
                            state.bitso_bid if state.bitso_bid > 0
                            else (state.bitso_bid + state.bitso_ask) / 2,
                        )
                    log.info(
                        "[OrderPoller] EXIT FILL CONFIRMED: oid=%s price=$%.5f — resetting directly.",
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
                                        saved_oid = risk.exit_oid
                                        risk.exit_oid = ""
                                        hold_sec = time.time() - risk.entry_ts
                                        exit_fill_px = await _fetch_exit_fill_price(
                                            saved_oid,
                                            risk.exit_passive_px if risk.exit_passive_px > 0
                                            else (state.bitso_bid + state.bitso_ask) / 2,
                                        )
                                        _reset_position(risk, pnl, exit_fill_px,
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


async def _fetch_rest_order_book() -> dict:
    """
    Fetch full order book snapshot via Bitso REST API.
    Returns {success, bids: {price: size}, asks: {price: size}, sequence: int}.

    v4.5.20: This is the critical fix for phantom signal entries.
    After every WebSocket reconnect, the local bids/asks dicts were rebuilt
    solely from diff-orders messages. But diff-orders only sends CHANGES
    that occur AFTER the connection. All orders already resting on the book
    are invisible. Result: local book has 3-10 levels while real book has 50.
    Local mid is wrong. Signals fire on phantom divergences.

    The Bitso docs prescribe: fetch REST book after subscribing to diff-orders,
    seed local dicts from snapshot, discard diff-orders with sequence <= snapshot.
    """
    try:
        path = f"/v3/order_book/?book={BITSO_BOOK}"
        s = _get_session()
        async with s.get(BITSO_API_URL + path) as r:
            data = await r.json()
        if data.get("success"):
            payload  = data["payload"]
            sequence = int(payload.get("sequence", 0))
            bid_dict = {}
            ask_dict = {}
            for b in payload.get("bids", []):
                px = float(b["price"])
                sz = float(b["amount"])
                if sz > 0:
                    bid_dict[px] = sz
            for a in payload.get("asks", []):
                px = float(a["price"])
                sz = float(a["amount"])
                if sz > 0:
                    ask_dict[px] = sz
            return {
                "success": True,
                "bids": bid_dict,
                "asks": ask_dict,
                "sequence": sequence,
                "n_bids": len(bid_dict),
                "n_asks": len(ask_dict),
            }
        return {"success": False, "error": data}
    except Exception as e:
        log.warning("[Bitso] REST order book fetch failed: %s", e)
        return {"success": False, "error": str(e)}


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

                # ── v4.5.20 FIX: Seed local book from REST snapshot ──────
                #
                # CRITICAL BUG in v4.5.19 and all prior versions:
                # After every reconnect, bids/asks dicts were cleared and rebuilt
                # solely from diff-orders. But diff-orders only sends CHANGES
                # after connection. All resting orders are invisible.
                #
                # Result: local book had 3-10 levels while real book had 50.
                # Local mid was wrong by 7-18 bps. Signals fired on phantom
                # divergences. Entries filled at the REAL ask (much higher),
                # then stop loss fired instantly on the stale local mid.
                #
                # March 25: 4 trades, all losses at -21 bps avg, 0% win rate.
                # All 4 entries filled 7-18 bps above the local best ask.
                # All fills were at a SINGLE price (no market impact).
                # Root cause: local ask was wrong, not market impact.
                #
                # FIX: Bitso docs prescribe fetching the REST order book after
                # subscribing, seeding local dicts, and discarding diff-orders
                # with sequence <= snapshot sequence. This is now implemented.
                #
                # The REST fetch adds ~300-500ms to reconnect time. Acceptable:
                # the alternative is trading on a phantom book for 30-60 seconds
                # until enough diffs arrive to approximate the real book.
                snapshot_seq = 0
                rest_book = await _fetch_rest_order_book()
                if rest_book.get("success"):
                    bids.update(rest_book["bids"])
                    asks.update(rest_book["asks"])
                    snapshot_seq = rest_book["sequence"]
                    # Immediately update state with accurate top-of-book
                    if bids and asks:
                        bb, ba = max(bids), min(asks)
                        if bb > 0 and ba > 0 and bb < ba:
                            state.update_bitso_top(bb, ba)
                    log.info(
                        "[Bitso] REST book seeded: %d bids, %d asks, seq=%d, "
                        "best_bid=$%.5f, best_ask=$%.5f, spread=%.2f bps",
                        rest_book["n_bids"], rest_book["n_asks"], snapshot_seq,
                        max(bids) if bids else 0,
                        min(asks) if asks else 0,
                        state.bitso_spread_bps,
                    )
                else:
                    log.warning(
                        "[Bitso] REST book fetch FAILED: %s. "
                        "Falling back to diff-orders only (degraded accuracy).",
                        rest_book.get("error", "unknown"),
                    )

                log.info("[Bitso] Connected. Book: %s (diff-orders + trades + REST seed)",
                         BITSO_BOOK)

                # v4.5.21: periodic REST refresh timer
                last_rest_refresh = time.time()
                BOOK_REFRESH_SEC  = 5.0  # re-fetch REST book every 5s

                async for raw in ws:
                    # ── v4.5.22: REST-ONLY book, polled every 5 seconds ───
                    #
                    # v4.5.20-21 attempted to seed the REST book on reconnect
                    # and/or every 30s, then apply diff-orders on top.
                    # This FAILS because diff-orders sends individual order
                    # changes while our dict uses price-level aggregates.
                    # When diff-orders removes ONE order at a price, we pop
                    # the entire level (including other orders at that price).
                    # Book corrupts within seconds of each REST seed.
                    #
                    # Evidence: SOL trade at 13:59:44 showed 10.9 bps gap
                    # between local ask and actual fill, 4.5 min after seed.
                    # XRP at 14:00:07 showed spread=0.07 bps (phantom book).
                    #
                    # FIX: REST is the SOLE source of truth for bids/asks.
                    # Polled every 5 seconds. diff-orders and trades messages
                    # are used ONLY as triggers for handle_exit/handle_entry,
                    # NOT for updating the book. This eliminates the corruption
                    # entirely. Max book staleness = 5 seconds.
                    #
                    # Cost: 12 public REST calls/min per asset = 24/min total.
                    # Public rate limit is 60/min. Safe.
                    now_rf = time.time()
                    if now_rf - last_rest_refresh > BOOK_REFRESH_SEC:
                        try:
                            refresh = await _fetch_rest_order_book()
                            if refresh.get("success"):
                                bids.clear()
                                asks.clear()
                                bids.update(refresh["bids"])
                                asks.update(refresh["asks"])
                                snapshot_seq = refresh["sequence"]
                                if bids and asks:
                                    bb_r, ba_r = max(bids), min(asks)
                                    if bb_r > 0 and ba_r > 0 and bb_r < ba_r:
                                        state.update_bitso_top(bb_r, ba_r)
                                log.debug(
                                    "[Bitso] REST refresh: %d bids, %d asks, spread=%.2f bps",
                                    refresh["n_bids"], refresh["n_asks"],
                                    state.bitso_spread_bps,
                                )
                        except Exception as e:
                            log.debug("[Bitso] REST refresh failed: %s", e)
                        last_rest_refresh = time.time()

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

                    # ── diff-orders channel (v4.5.22: TRIGGER ONLY) ───────
                    # diff-orders is NO LONGER used to update bids/asks.
                    # REST poll every 5s is the sole source of truth.
                    #
                    # diff-orders still serves two purposes:
                    # 1. Trigger handle_exit (stop loss) on every message
                    # 2. Keep the stale guard timer alive (prevents reconnect)
                    #
                    # The bids/asks dicts are only modified by REST refresh.

                    if not bids or not asks:
                        continue

                    # Stale guard: fires on EVERY message regardless of book state.
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
        path    = f"/v3/open_orders/?book={BITSO_BOOK}"
        headers = _bitso_headers("GET", path)
        s = _get_session()
        async with s.get(
            BITSO_API_URL + path, headers=headers,
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
    log.info("Bitso Lead-Lag Trader v4.5.22  |  %s  |  %s",
             ASSET.upper(), EXEC_MODE.upper())
    log.info("Book: %s  Binance: %s  Coinbase: %s",
             BITSO_BOOK, BINANCE_SYMBOL, COINBASE_SYMBOL)
    log.info("Threshold: %.1f-%.1fbps  Window: %.1fs  Size: %.6f %s",
             ENTRY_THRESHOLD_BPS, ENTRY_MAX_BPS, SIGNAL_WINDOW_SEC, MAX_POS_ASSET, ASSET.upper())
    log.info("Stop: %.1fbps  Hold: %.1fs  Cooldown: %.1fs  Spread: %.1f-%.1fbps",
             STOP_LOSS_BPS, HOLD_SEC, COOLDOWN_SEC, SPREAD_MIN_BPS, SPREAD_MAX_BPS)
    log.info("Force close: %.2f%%  Reconcile: %.0fs  Chase: %.1fs  Entry slippage: %d ticks",
             FORCE_CLOSE_SLIPPAGE * 100, RECONCILE_SEC, EXIT_CHASE_SEC, ENTRY_SLIPPAGE_TICKS)
    log.info("Daily limit: $%.2f  Combined: %s  Trade log: %s",
             MAX_DAILY_LOSS_USD, COMBINED_SIGNAL, TRADE_LOG)
    log.info("Circuit breaker: %d consecutive losses → pause %.0fs",
             CONSECUTIVE_LOSS_MAX, CONSECUTIVE_LOSS_PAUSE)
    log.info("=" * 66)

    ok = await startup_checks()
    if not ok:
        return

    state = MarketState()
    risk  = RiskState()
    pnl   = PnLTracker()

    await tg(
        f"Bitso Lead-Lag v4.5.22 [{EXEC_MODE.upper()}] {ASSET.upper()} started\n"
        f"Book: {BITSO_BOOK}  Threshold: {ENTRY_THRESHOLD_BPS}-{ENTRY_MAX_BPS}bps  Window: {SIGNAL_WINDOW_SEC}s\n"
        f"Size: {MAX_POS_ASSET} {ASSET.upper()}  Limit: ${MAX_DAILY_LOSS_USD}\n"
        f"Spread: {SPREAD_MIN_BPS}-{SPREAD_MAX_BPS}bps  Force close: {FORCE_CLOSE_SLIPPAGE*100:.1f}%  Reconciler: {int(RECONCILE_SEC)}s\n"
        f"Circuit breaker: {CONSECUTIVE_LOSS_MAX} losses → {CONSECUTIVE_LOSS_PAUSE/60:.0f}min pause"
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
        # Close persistent HTTP session
        if _http_session and not _http_session.closed:
            await _http_session.close()
        log.info("Shutdown.\n%s", pnl.summary_text(EXEC_MODE, 0))
        _send_telegram_sync(
            f"Bitso Lead-Lag v4.5.22 STOPPED [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
            + pnl.summary_text(EXEC_MODE, 0)
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
