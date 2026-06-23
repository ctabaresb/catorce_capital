#!/usr/bin/env python3
"""
shadow_capture.py — Log the LIVE feature engine's per-minute output, so it can
later be diffed against the TRAINING feature builder (see shadow_diff.py).

WHY: the documented #1 driver of this bot's shadow→live degradation is
train-vs-live feature divergence — tick features approximated from REST trades
(vs S3 quote ticks in training), ~12% of features falling back to medians, and
silent value-divergence on same-named features (dist_ema_*, rsi_14, tick feats).
This captures exactly what the engine feeds the model, minute by minute, so the
gap can be MEASURED instead of guessed.

SAFETY: read-only. It instantiates the bot's HLClient with an EPHEMERAL key (so
the SDK is happy) and only ever calls get_snapshot/get_mid/all_mids — never any
order/sign path. It needs NO SSM secrets and places NO trades. It does not touch
xgb_bot.py's trading code.

RUN (in the xgb_hyperliquid python env, after the bot's deps are installed):
    python3 autonomous_agent/shadow_capture.py --coins SOL --interval 60
Outputs a parquet under data/shadow_capture/. Honors the engine's 130-min
warmup: rows are only emitted once is_warm() for that coin.
"""
import argparse
import glob
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

# This file lives in autonomous_agent/; the bot modules live one level up.
HFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HFT_DIR not in sys.path:
    sys.path.insert(0, HFT_DIR)


def load_union_features(models_dir: str) -> list:
    """Union of every model's features.json under models_dir/{asset}/{cfg}/."""
    paths = glob.glob(os.path.join(models_dir, "*", "*", "features.json"))
    if not paths:
        raise SystemExit(f"No features.json found under {models_dir}")
    feats = []
    seen = set()
    for p in sorted(paths):
        with open(p) as fh:
            for f in json.load(fh):
                if f not in seen:
                    seen.add(f)
                    feats.append(f)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow capture of live engine features")
    ap.add_argument("--models_dir", default=os.path.join(HFT_DIR, "models", "live_v8"),
                    help="Models dir whose features.json define the captured columns")
    ap.add_argument("--coins", default="BTC,ETH,SOL",
                    help="Comma-separated coins to capture (engines run per coin)")
    ap.add_argument("--out_dir", default=os.path.join(HFT_DIR, "data", "shadow_capture"))
    ap.add_argument("--interval", type=float, default=60.0, help="Seconds between ticks")
    ap.add_argument("--warmup", type=int, default=130,
                    help="Warmup minutes before rows are emitted (lower for a smoke test)")
    ap.add_argument("--flush_every", type=int, default=10, help="Flush parquet every N captured rows")
    ap.add_argument("--max_minutes", type=int, default=0, help="Stop after N minutes (0 = run forever)")
    ap.add_argument("--s3_prefix", default="",
                    help="Optional s3://bucket/prefix/ to upload the parquet to on each flush")
    args = ap.parse_args()

    # Heavy imports deferred so --help / syntax check work without the SDK.
    import secrets
    import pandas as pd
    import eth_account
    from xgb_feature_engine import XGBFeatureEngine, COIN_CONFIGS
    from xgb_bot import HLClient

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    for c in coins:
        if c not in COIN_CONFIGS:
            raise SystemExit(f"Unknown coin {c}; known: {list(COIN_CONFIGS)}")

    feats_list = load_union_features(args.models_dir)
    print(f"[shadow_capture] {len(feats_list)} feature columns from {args.models_dir}")
    print(f"[shadow_capture] coins={coins} interval={args.interval}s warmup={args.warmup}min")

    # Read-only client: ephemeral key satisfies HLClient.__init__; we only ever
    # call the public Info paths (get_snapshot/all_mids). It never signs.
    eph_key = "0x" + secrets.token_hex(32)
    eph_addr = eth_account.Account.from_key(eph_key).address
    client = HLClient(private_key=eph_key, wallet_address=eph_addr)

    engines = {c: XGBFeatureEngine(coin=c, buffer_minutes=360, warmup_minutes=args.warmup) for c in coins}

    os.makedirs(args.out_dir, exist_ok=True)
    start_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"shadow_capture_{'-'.join(coins)}_{start_stamp}.parquet")

    rows: list = []
    state = {"stop": False}

    def _stop(*_):
        state["stop"] = True
        print("\n[shadow_capture] stop requested — flushing...")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def flush():
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        if args.s3_prefix:
            try:
                import boto3
                bkt, key = args.s3_prefix.replace("s3://", "").split("/", 1)
                boto3.client("s3").upload_file(out_path, bkt, key.rstrip("/") + "/" + os.path.basename(out_path))
            except Exception as e:
                print(f"[shadow_capture] WARN s3 upload failed: {e}")

    tick = 0
    captured_since_flush = 0
    while not state["stop"]:
        t0 = time.time()
        tick += 1
        try:
            all_mids = client.info.all_mids()
        except Exception:
            all_mids = {}

        hb = []
        for c in coins:
            eng = engines[c]
            try:
                snap = client.get_snapshot(c)
            except Exception as e:
                snap = None
                print(f"[shadow_capture] WARN get_snapshot({c}) failed: {e}")
            missing = snap is None
            if not missing:
                cross = {prefix: float(all_mids.get(other, 0) or 0)
                         for prefix, other in COIN_CONFIGS[c]["cross_assets"]}
                try:
                    eng.tick(snap, cross)
                except Exception as e:
                    print(f"[shadow_capture] WARN tick({c}) failed: {e}")

            warm = eng.is_warm()
            hb.append(f"{c}:warm={int(warm)}/{eng._minutes_ingested}m")

            if warm and not missing:
                try:
                    f = eng.compute_features(feats_list)
                except Exception as e:
                    print(f"[shadow_capture] WARN compute_features({c}) failed: {e}")
                    continue
                ts = time.time()
                rows.append({
                    "ts_unix": ts,
                    "ts_min": pd.Timestamp(ts, unit="s", tz="UTC").floor("min"),
                    "coin": c,
                    "snapshot_missing": False,
                    **{k: float(v) for k, v in f.items()},
                })
                captured_since_flush += 1

        print(f"[shadow_capture] tick={tick} " + " ".join(hb) + f" rows={len(rows)}")

        if captured_since_flush >= args.flush_every:
            flush()
            captured_since_flush = 0
            print(f"[shadow_capture] flushed -> {out_path}")

        if args.max_minutes and tick >= args.max_minutes:
            break

        # sleep to the next interval boundary
        dt = args.interval - (time.time() - t0)
        if dt > 0 and not state["stop"]:
            time.sleep(dt)

    flush()
    print(f"[shadow_capture] done. {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
