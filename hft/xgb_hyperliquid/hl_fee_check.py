#!/usr/bin/env python3
"""
hl_fee_check.py — Fetch and analyze Hyperliquid fee structure for a wallet.

Implements the official fee formula from Hyperliquid docs to compute
actual trading costs after all discounts (staking, referral, aligned quote).

Usage:
  python hl_fee_check.py                              # Uses default wallet
  python hl_fee_check.py --wallet 0xYOUR_ADDRESS      # Specify wallet
  python hl_fee_check.py --wallet 0xYOUR_ADDRESS --coin BTC  # Specific coin
"""

import argparse
import json
import requests
import sys

DEFAULT_WALLET = "YOUR_WALLET_ADDRESS_HERE"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# Native HL perps (not HIP-3 deployed) have deployerFeeScale=0, growthMode=false
# HIP-3 deployed perps would have different values
NATIVE_PERPS = {"BTC", "ETH", "SOL", "DOGE", "AVAX", "ARB", "OP", "SUI", "APT"}

# Coins that settle in USDC (aligned quote token on Hyperliquid)
ALIGNED_QUOTE_COINS = {"BTC", "ETH", "SOL", "DOGE", "AVAX", "ARB", "OP", "SUI",
                        "APT", "LINK", "MATIC", "ADA", "DOT", "NEAR", "ATOM"}


def fetch_user_fees(wallet: str) -> dict:
    """Fetch fee info from Hyperliquid API."""
    resp = requests.post(HL_INFO_URL, json={"type": "userFees", "user": wallet}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def compute_fee_rates(
    maker_rate: float,
    taker_rate: float,
    active_referral_discount: float,
    is_aligned_quote_token: bool,
    deployer_fee_scale: float = 0,
    growth_mode: bool = False,
    is_perp: bool = True,
) -> dict:
    """
    Replicate the official Hyperliquid fee formula.
    Returns maker and taker percentages after all discounts.
    """
    scale_if_hip3 = 1.0
    growth_mode_scale = 1.0
    deployer_share = 0.0

    if is_perp:
        if deployer_fee_scale < 1:
            scale_if_hip3 = deployer_fee_scale + 1
        else:
            scale_if_hip3 = deployer_fee_scale * 2

        if deployer_fee_scale < 1:
            deployer_share = deployer_fee_scale / (1 + deployer_fee_scale)
        else:
            deployer_share = 0.5

        if growth_mode:
            growth_mode_scale = 0.1

    # Maker
    maker_pct = maker_rate * 100 * growth_mode_scale
    if maker_pct > 0:
        maker_pct *= scale_if_hip3 * (1 - active_referral_discount)
    else:
        maker_rebate_scale = (
            (1 - deployer_share) * 1.5 + deployer_share
            if is_aligned_quote_token
            else 1.0
        )
        maker_pct *= maker_rebate_scale

    # Taker
    taker_pct = (
        taker_rate * 100
        * scale_if_hip3
        * growth_mode_scale
        * (1 - active_referral_discount)
    )
    if is_aligned_quote_token:
        taker_scale = (1 - deployer_share) * 0.8 + deployer_share
        taker_pct *= taker_scale

    return {
        "maker_pct": maker_pct,
        "taker_pct": taker_pct,
        "maker_bps": maker_pct * 100,
        "taker_bps": taker_pct * 100,
    }


def main():
    ap = argparse.ArgumentParser(description="Hyperliquid Fee Analyzer")
    ap.add_argument("--wallet", default=DEFAULT_WALLET, help="Wallet address")
    ap.add_argument("--coin", default="BTC", help="Coin to analyze (default: BTC)")
    ap.add_argument("--raw", action="store_true", help="Print raw API response")
    args = ap.parse_args()

    print(f"Fetching fees for {args.wallet[:10]}...{args.wallet[-6:]}\n")

    data = fetch_user_fees(args.wallet)

    if args.raw:
        print(json.dumps(data, indent=2))
        return

    # Extract key fields
    user_cross_rate = float(data["userCrossRate"])
    user_add_rate = float(data["userAddRate"])
    referral_discount = float(data["activeReferralDiscount"])
    staking = data.get("activeStakingDiscount", {})
    staking_discount = float(staking.get("discount", 0)) if staking else 0
    staking_bps = staking.get("bpsOfMaxSupply", "0")

    base_cross = float(data["feeSchedule"]["cross"])
    base_add = float(data["feeSchedule"]["add"])

    print("=" * 60)
    print("  ACCOUNT FEE SUMMARY")
    print("=" * 60)
    print(f"  Base taker rate:      {base_cross * 100:.4f}% ({base_cross * 1e4:.1f} bps)")
    print(f"  Base maker rate:      {base_add * 100:.4f}% ({base_add * 1e4:.1f} bps)")
    print(f"  Staking discount:     {staking_discount:.0%} (bps of supply: {staking_bps})")
    print(f"  Referral discount:    {referral_discount:.0%}")
    print(f"  After staking taker:  {user_cross_rate * 100:.4f}% ({user_cross_rate * 1e4:.2f} bps)")
    print(f"  After staking maker:  {user_add_rate * 100:.4f}% ({user_add_rate * 1e4:.2f} bps)")

    # Compute actual rates for the requested coin
    coin = args.coin.upper()
    is_native = coin in NATIVE_PERPS
    is_aligned = coin in ALIGNED_QUOTE_COINS

    rates = compute_fee_rates(
        maker_rate=user_add_rate,
        taker_rate=user_cross_rate,
        active_referral_discount=referral_discount,
        is_aligned_quote_token=is_aligned,
        deployer_fee_scale=0 if is_native else 0,
        growth_mode=False,
        is_perp=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  ACTUAL FEES FOR {coin} PERP")
    print(f"  Native perp: {is_native}  |  Aligned quote: {is_aligned}")
    print(f"{'=' * 60}")
    print(f"  Taker:  {rates['taker_bps']:.2f} bps")
    print(f"  Maker:  {rates['maker_bps']:.2f} bps")

    # Cost model for our strategy (taker entry + maker exit)
    rt_taker_taker = rates["taker_bps"] * 2
    rt_taker_maker = rates["taker_bps"] + rates["maker_bps"]

    print(f"\n  ROUND-TRIP COST MODELS:")
    print(f"  Taker entry + Maker exit:  {rt_taker_maker:.2f} bps  (our strategy)")
    print(f"  Taker entry + Taker exit:  {rt_taker_taker:.2f} bps  (worst case)")

    # Show what we assumed vs reality
    assumed_rt = 5.4
    actual_rt = rt_taker_maker
    delta = assumed_rt - actual_rt
    print(f"\n  COST MODEL COMPARISON:")
    print(f"  Training assumption:  {assumed_rt:.2f} bps RT")
    print(f"  Actual cost:          {actual_rt:.2f} bps RT")
    print(f"  Hidden edge:          {delta:+.2f} bps per trade {'(conservative)' if delta > 0 else '(UNDERESTIMATED)'}")

    # Available optimizations
    print(f"\n{'=' * 60}")
    print(f"  AVAILABLE OPTIMIZATIONS")
    print(f"{'=' * 60}")

    avail_referral = float(data["feeSchedule"]["referralDiscount"])
    if referral_discount == 0 and avail_referral > 0:
        rates_with_ref = compute_fee_rates(
            maker_rate=user_add_rate,
            taker_rate=user_cross_rate,
            active_referral_discount=avail_referral,
            is_aligned_quote_token=is_aligned,
        )
        rt_with_ref = rates_with_ref["taker_bps"] + rates_with_ref["maker_bps"]
        print(f"  Referral code ({avail_referral:.0%} discount): RT would drop to {rt_with_ref:.2f} bps")
        print(f"    Saving: {actual_rt - rt_with_ref:.2f} bps per trade")

    # Next staking tier
    tiers = data["feeSchedule"]["stakingDiscountTiers"]
    current_discount = staking_discount
    for tier in tiers:
        if float(tier["discount"]) > current_discount:
            print(f"  Next staking tier: {float(tier['discount']):.0%} discount "
                  f"(requires {tier['bpsOfMaxSupply']} bps of max supply)")
            next_rates = compute_fee_rates(
                maker_rate=base_cross * (1 - float(tier["discount"])),
                taker_rate=base_add * (1 - float(tier["discount"])),
                active_referral_discount=referral_discount,
                is_aligned_quote_token=is_aligned,
            )
            break

    # Multi-asset summary
    print(f"\n{'=' * 60}")
    print(f"  MULTI-ASSET FEE TABLE")
    print(f"{'=' * 60}")
    print(f"  {'Coin':<8} {'Aligned':<10} {'Taker bps':<12} {'Maker bps':<12} {'RT bps':<10}")
    print(f"  {'-'*52}")
    for c in ["BTC", "ETH", "SOL"]:
        r = compute_fee_rates(
            maker_rate=user_add_rate,
            taker_rate=user_cross_rate,
            active_referral_discount=referral_discount,
            is_aligned_quote_token=c in ALIGNED_QUOTE_COINS,
        )
        rt = r["taker_bps"] + r["maker_bps"]
        print(f"  {c:<8} {'Yes' if c in ALIGNED_QUOTE_COINS else 'No':<10} "
              f"{r['taker_bps']:<12.2f} {r['maker_bps']:<12.2f} {rt:<10.2f}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
