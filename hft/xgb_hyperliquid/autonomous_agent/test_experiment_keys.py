#!/usr/bin/env python3
"""
test_experiment_keys.py — Read-only sanity check for the $50 HL experiment wallet.

Verifies (WITHOUT placing any order):
  1. The API-wallet private key is valid and derives an agent address.
  2. The main account is funded (perp + spot USDC ~ $50).
  3. The API wallet is an APPROVED agent of the main account (i.e. it can sign
     trades for it).

SECURITY: the private key is read ONLY from the env var HL_AGENT_KEY. It is
never taken as a CLI argument (would show in `ps`/history), never printed, and
never written anywhere. Run it like this and the key stays in your shell memory:

    read -rs HL_AGENT_KEY          # paste the API-wallet private key, press Enter
    export HL_MAIN_ADDRESS=0x...   # the MAIN account address (the Rabby one with $50)
    python3 test_experiment_keys.py
    unset HL_AGENT_KEY

Deps: eth-account, requests  (a scratch venv is fine).
"""

import json
import os
import sys

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def _info(req: dict) -> dict:
    import requests
    r = requests.post(HL_INFO_URL, json=req, timeout=10)
    r.raise_for_status()
    return r.json()


def main() -> int:
    key = os.environ.get("HL_AGENT_KEY", "").strip()
    main_addr = (os.environ.get("HL_MAIN_ADDRESS")
                 or (sys.argv[1] if len(sys.argv) > 1 else "")).strip()

    if not key or not main_addr:
        print("Missing input.\n"
              "  read -rs HL_AGENT_KEY        # paste API-wallet private key\n"
              "  export HL_MAIN_ADDRESS=0x... # main account address ($50 wallet)\n"
              "  python3 test_experiment_keys.py", file=sys.stderr)
        return 2

    # 1) Derive the agent address from the key (never print the key itself).
    try:
        from eth_account import Account
        agent_addr = Account.from_key(key).address
    except Exception:
        print("FAIL: HL_AGENT_KEY is not a valid private key (could not derive an "
              "address). Re-copy the API-wallet private key.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("  HL EXPERIMENT WALLET — read-only sanity check")
    print("=" * 60)
    print(f"  Main account (funds):  {main_addr}")
    print(f"  Agent/API wallet:      {agent_addr}")
    print("    ^ compare this to the API wallet address Hyperliquid showed you.")

    # 2) Funding: perp accountValue + spot USDC (mirrors xgb_bot get_equity).
    try:
        perp = _info({"type": "clearinghouseState", "user": main_addr})
        perp_equity = float(perp.get("marginSummary", {}).get("accountValue", 0) or 0)
    except Exception as e:
        print(f"  WARN: could not read perp state: {e}")
        perp_equity = 0.0

    spot_usdc = 0.0
    try:
        spot = _info({"type": "spotClearinghouseState", "user": main_addr})
        for bal in spot.get("balances", []):
            if bal.get("coin") == "USDC":
                spot_usdc = float(bal.get("total", 0) or 0)
    except Exception as e:
        print(f"  WARN: could not read spot state: {e}")

    total = perp_equity + spot_usdc
    print("-" * 60)
    print(f"  Perp account value:    ${perp_equity:,.2f}")
    print(f"  Spot USDC balance:     ${spot_usdc:,.2f}")
    print(f"  TOTAL equity:          ${total:,.2f}")
    if total < 1:
        print("    !! $0 — did you pass the API-wallet address instead of the MAIN "
              "account address? Funds live in the MAIN account.")

    # 3) Is this agent approved to sign for the main account?
    approved = False
    try:
        agents = _info({"type": "extraAgents", "user": main_addr})
        approved = agent_addr.lower() in json.dumps(agents).lower()
    except Exception as e:
        print(f"  WARN: could not read approved agents: {e}")

    print("-" * 60)
    print(f"  API wallet approved as agent:  {'YES ✓' if approved else 'NO ✗'}")
    if not approved:
        print("    The derived agent address is not in the main account's approved "
              "agents. Re-approve the API wallet in HL (More → API).")
    print("=" * 60)

    ok = (total >= 1) and approved
    print("  RESULT:", "ALL GOOD ✓ — keys work, account funded, agent authorized."
          if ok else "NOT READY ✗ — see notes above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
