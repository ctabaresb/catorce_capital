# =============================================================================
# src/ingestion/universe.py
#
# Asset universe management.
# Defines which assets are in scope, their categories, and risk tiers.
# This is the single source of truth that drives:
#   - Which assets are ingested daily
#   - Which assets enter Conservative / Balanced / Aggressive portfolios
#   - Validation coverage checks in validator.py
#
# =============================================================================
#
# HOW TO ADD A TOKEN (2 steps):
#   1. Find the CoinGecko ID at coingecko.com/en/coins/<token-name>
#      The ID is in the URL: coingecko.com/en/coins/bittensor -> ID is "bittensor"
#   2. Add one line to UNIVERSE_SEED below:
#
#      AssetDefinition("coin-id", "TICK", "Display Name", AssetCategory.CATEGORY, RiskTier.TIER, max_rank)
#
#      RiskTier choices:
#        LOW       -> appears in Conservative + Balanced + Aggressive
#        MEDIUM    -> appears in Balanced + Aggressive only
#        HIGH      -> appears in Aggressive only
#        VERY_HIGH -> appears in Aggressive only
#        EXCLUDED  -> never included (use for stablecoins)
#
#      max_rank = maximum market cap rank to include this asset.
#      Set to 999 if you always want it regardless of rank.
#
# HOW TO REMOVE A TOKEN:
#   Delete its line from UNIVERSE_SEED. Done.
#
# AFTER CHANGES: run a backfill to populate Silver with the new asset history.
#   See README for the backfill command.
#
# =============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums: category and risk tier taxonomies
# ---------------------------------------------------------------------------

class AssetCategory(str, Enum):
    LAYER_1        = "layer_1"
    LAYER_2        = "layer_2"
    DEFI           = "defi"
    INFRASTRUCTURE = "infrastructure"
    AI_TOKEN       = "ai_token"
    EXCHANGE_TOKEN = "exchange_token"
    MEME           = "meme"
    STABLECOIN     = "stablecoin"
    OTHER          = "other"


class RiskTier(str, Enum):
    LOW       = "low"        # Conservative portfolio eligible
    MEDIUM    = "medium"     # Balanced portfolio eligible
    HIGH      = "high"       # Aggressive portfolio eligible
    VERY_HIGH = "very_high"  # Aggressive portfolio only
    EXCLUDED  = "excluded"   # Never included (stablecoins, wrapped assets)


class PortfolioProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    AGGRESSIVE   = "aggressive"


# ---------------------------------------------------------------------------
# Asset definition
# ---------------------------------------------------------------------------

@dataclass
class AssetDefinition:
    coin_id:      str            # CoinGecko canonical ID
    symbol:       str            # ticker
    name:         str            # display name
    category:     AssetCategory
    risk_tier:    RiskTier
    max_mcap_rank: int           # only include if market cap rank <= this


# ---------------------------------------------------------------------------
# Portfolio profile -> eligible risk tiers mapping
# Conservative sees Low only.
# Balanced sees Low + Medium.
# Aggressive sees Low + Medium + High + Very High.
# ---------------------------------------------------------------------------

PROFILE_ELIGIBLE_TIERS: dict[PortfolioProfile, set[RiskTier]] = {
    PortfolioProfile.CONSERVATIVE: {RiskTier.LOW},
    PortfolioProfile.BALANCED:     {RiskTier.LOW, RiskTier.MEDIUM},
    PortfolioProfile.AGGRESSIVE:   {RiskTier.LOW, RiskTier.MEDIUM,
                                    RiskTier.HIGH, RiskTier.VERY_HIGH},
}


# ---------------------------------------------------------------------------
# Asset universe - curated for disruption, AI, and high-conviction DeFi
#
# Philosophy:
#   Conservative = battle-tested L1s with deep liquidity only
#   Balanced     = adds high-quality L2s, DeFi blue chips, and AI infrastructure
#   Aggressive   = adds AI tokens, exchange tokens, and asymmetric bets
#
# Excluded: stablecoins, meme coins, dead gaming tokens, legacy L1s
# ---------------------------------------------------------------------------

UNIVERSE_SEED: list[AssetDefinition] = [

    # -------------------------------------------------------------------------
    # CONSERVATIVE TIER (Low risk)
    # Only the 4 assets with deepest liquidity and institutional adoption.
    # Max drawdown protection > upside capture at this level.
    # -------------------------------------------------------------------------
    AssetDefinition("bitcoin",       "btc",   "Bitcoin",         AssetCategory.LAYER_1,        RiskTier.LOW,       1),
    AssetDefinition("ethereum",      "eth",   "Ethereum",        AssetCategory.LAYER_1,        RiskTier.LOW,       3),
    AssetDefinition("solana",        "sol",   "Solana",          AssetCategory.LAYER_1,        RiskTier.LOW,       10),
    AssetDefinition("binancecoin",   "bnb",   "BNB",             AssetCategory.LAYER_1,        RiskTier.LOW,       10),

    # -------------------------------------------------------------------------
    # BALANCED TIER (Medium risk)
    # Quality L1s, L2s, and DeFi protocols with real revenue and users.
    # Disruption thesis: infrastructure for the next financial system.
    # -------------------------------------------------------------------------

    # High-conviction L1s
    AssetDefinition("ripple",        "xrp",   "XRP",             AssetCategory.LAYER_1,        RiskTier.MEDIUM,    10),
    AssetDefinition("hyperliquid",   "hype",  "Hyperliquid",     AssetCategory.EXCHANGE_TOKEN, RiskTier.MEDIUM,    25),
    AssetDefinition("sui",           "sui",   "Sui",             AssetCategory.LAYER_1,        RiskTier.MEDIUM,    20),
    AssetDefinition("near",          "near",  "NEAR Protocol",   AssetCategory.LAYER_1,        RiskTier.MEDIUM,    30),

    # Layer 2 scaling
    AssetDefinition("arbitrum",      "arb",   "Arbitrum",        AssetCategory.LAYER_2,        RiskTier.MEDIUM,    30),
    AssetDefinition("optimism",      "op",    "Optimism",        AssetCategory.LAYER_2,        RiskTier.MEDIUM,    40),

    # DeFi blue chips (real revenue, real users)
    AssetDefinition("uniswap",       "uni",   "Uniswap",         AssetCategory.DEFI,           RiskTier.MEDIUM,    30),
    AssetDefinition("aave",          "aave",  "Aave",            AssetCategory.DEFI,           RiskTier.MEDIUM,    40),
    AssetDefinition("chainlink",     "link",  "Chainlink",       AssetCategory.DEFI,           RiskTier.MEDIUM,    20),
    AssetDefinition("lido-dao",      "ldo",   "Lido DAO",        AssetCategory.DEFI,           RiskTier.MEDIUM,    30),
    AssetDefinition("jupiter-exchange-solana","jup","Jupiter",   AssetCategory.DEFI,           RiskTier.MEDIUM,    50),
    AssetDefinition("pendle",        "pendle","Pendle",          AssetCategory.DEFI,           RiskTier.MEDIUM,    60),

    # Infrastructure (decentralized compute + data)
    AssetDefinition("render-token",  "rndr",  "Render",          AssetCategory.INFRASTRUCTURE, RiskTier.MEDIUM,    50),
    AssetDefinition("the-graph",     "grt",   "The Graph",       AssetCategory.INFRASTRUCTURE, RiskTier.MEDIUM,    60),

    # -------------------------------------------------------------------------
    # AGGRESSIVE TIER (High risk)
    # Asymmetric bets on AI, modular blockchain, and DeFi disruption.
    # Higher volatility, higher conviction required.
    # -------------------------------------------------------------------------

    # AI infrastructure (the picks-and-shovels of AI)
    AssetDefinition("bittensor",     "tao",   "Bittensor",       AssetCategory.AI_TOKEN,       RiskTier.HIGH,      40),
    AssetDefinition("fetch-ai",      "fet",   "Fetch.ai",        AssetCategory.AI_TOKEN,       RiskTier.HIGH,      80),
    AssetDefinition("singularitynet","agix",  "SingularityNET",  AssetCategory.AI_TOKEN,       RiskTier.HIGH,      80),
    AssetDefinition("ocean-protocol","ocean", "Ocean Protocol",  AssetCategory.AI_TOKEN,       RiskTier.HIGH,      100),
    AssetDefinition("worldcoin-wld", "wld",   "Worldcoin",       AssetCategory.AI_TOKEN,       RiskTier.HIGH,      80),

    # Modular blockchain
    AssetDefinition("celestia",      "tia",   "Celestia",        AssetCategory.LAYER_1,        RiskTier.HIGH,      30),

    # DeFi disruption
    AssetDefinition("injective-protocol","inj","Injective",      AssetCategory.DEFI,           RiskTier.HIGH,      50),
    AssetDefinition("maker",         "mkr",   "MakerDAO",        AssetCategory.DEFI,           RiskTier.HIGH,      40),

    # -------------------------------------------------------------------------
    # MEME TIER (Very high risk, Aggressive only)
    # Included for completeness. Treat as lottery tickets.
    # -------------------------------------------------------------------------
    AssetDefinition("dogecoin",      "doge",  "Dogecoin",        AssetCategory.MEME,           RiskTier.VERY_HIGH, 15),

    # -------------------------------------------------------------------------
    # EXCLUDED (fetched for market data but never in portfolios)
    # -------------------------------------------------------------------------
    AssetDefinition("tether",        "usdt",  "Tether",          AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  1),
    AssetDefinition("usd-coin",      "usdc",  "USD Coin",        AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  1),
    AssetDefinition("dai",           "dai",   "Dai",             AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  1),
]


# ---------------------------------------------------------------------------
# Universe manager class
# ---------------------------------------------------------------------------

class UniverseManager:
    """
    Manages the asset universe: filtering, classification, and enrichment.

    The universe is the contract between:
      - Ingestion (what to fetch)
      - Transform (what to classify)
      - Backtest (what profiles to build)
    """

    def __init__(self, assets: list[AssetDefinition] = UNIVERSE_SEED) -> None:
        self._assets  = assets
        self._by_id   = {a.coin_id: a for a in assets}

    # -------------------------------------------------------------------------
    # Filtering methods
    # -------------------------------------------------------------------------

    def get_all_coin_ids(self, include_excluded: bool = False) -> list[str]:
        """All coin IDs in the universe. Stablecoins excluded by default."""
        return [
            a.coin_id for a in self._assets
            if include_excluded or a.risk_tier != RiskTier.EXCLUDED
        ]

    def get_investable_ids(self) -> list[str]:
        """Assets eligible for at least one portfolio profile."""
        return [
            a.coin_id for a in self._assets
            if a.risk_tier != RiskTier.EXCLUDED
        ]

    def get_ids_for_profile(
        self,
        profile: PortfolioProfile,
        live_ranks: dict[str, int] | None = None,
    ) -> list[str]:
        """
        Assets eligible for a given portfolio profile.

        Args:
            profile:     Conservative / Balanced / Aggressive
            live_ranks:  {coin_id: current_market_cap_rank} from latest ingestion.
                         If provided, assets exceeding their max_mcap_rank are excluded.
        """
        eligible_tiers = PROFILE_ELIGIBLE_TIERS[profile]

        result = []
        for asset in self._assets:
            if asset.risk_tier not in eligible_tiers:
                continue

            if live_ranks and asset.coin_id in live_ranks:
                current_rank = live_ranks[asset.coin_id]
                if current_rank > asset.max_mcap_rank:
                    logger.debug(
                        "Excluding %s from %s: rank=%d > max=%d",
                        asset.coin_id, profile.value,
                        current_rank, asset.max_mcap_rank,
                    )
                    continue

            result.append(asset.coin_id)

        logger.info(
            "Universe for profile=%s: %d assets", profile.value, len(result)
        )
        return result

    def get_ids_for_category(self, category: AssetCategory) -> list[str]:
        """Assets in a specific category."""
        return [a.coin_id for a in self._assets if a.category == category]

    def get_asset(self, coin_id: str) -> AssetDefinition | None:
        """Look up a single asset by CoinGecko ID."""
        return self._by_id.get(coin_id)

    def get_expected_validation_set(self, max_rank: int = 50) -> set[str]:
        """Assets within max_rank for daily validation."""
        return {
            a.coin_id for a in self._assets
            if a.risk_tier != RiskTier.EXCLUDED
            and a.max_mcap_rank <= max_rank
        }

    # -------------------------------------------------------------------------
    # Enrichment: add category + risk tags to ingested records
    # -------------------------------------------------------------------------

    def enrich_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Add category, risk_tier, and profile eligibility flags to raw records.
        Records for unknown assets get category='other', risk_tier='high'.
        Called in ingest_eod.py before writing to Bronze.
        """
        enriched = []
        for record in records:
            coin_id = record.get("id", "")
            asset   = self._by_id.get(coin_id)

            record["category"]  = asset.category.value  if asset else AssetCategory.OTHER.value
            record["risk_tier"] = asset.risk_tier.value if asset else RiskTier.HIGH.value

            record["in_conservative"] = (
                asset is not None and
                asset.risk_tier in PROFILE_ELIGIBLE_TIERS[PortfolioProfile.CONSERVATIVE]
            )
            record["in_balanced"] = (
                asset is not None and
                asset.risk_tier in PROFILE_ELIGIBLE_TIERS[PortfolioProfile.BALANCED]
            )
            record["in_aggressive"] = (
                asset is not None and
                asset.risk_tier != RiskTier.EXCLUDED
            )

            enriched.append(record)

        return enriched

    # -------------------------------------------------------------------------
    # Serialization for S3 storage
    # -------------------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Convert universe to list of dicts for Parquet writing."""
        return [
            {
                "coin_id":         a.coin_id,
                "symbol":          a.symbol,
                "name":            a.name,
                "category":        a.category.value,
                "risk_tier":       a.risk_tier.value,
                "max_mcap_rank":   a.max_mcap_rank,
                "in_conservative": a.risk_tier in PROFILE_ELIGIBLE_TIERS[PortfolioProfile.CONSERVATIVE],
                "in_balanced":     a.risk_tier in PROFILE_ELIGIBLE_TIERS[PortfolioProfile.BALANCED],
                "in_aggressive":   a.risk_tier != RiskTier.EXCLUDED,
            }
            for a in self._assets
        ]

    def summary(self) -> dict[str, Any]:
        """Quick summary for logging and audit."""
        investable = self.get_investable_ids()
        return {
            "total_assets":       len(self._assets),
            "investable_assets":  len(investable),
            "excluded_assets":    len(self._assets) - len(investable),
            "conservative_count": len(self.get_ids_for_profile(PortfolioProfile.CONSERVATIVE)),
            "balanced_count":     len(self.get_ids_for_profile(PortfolioProfile.BALANCED)),
            "aggressive_count":   len(self.get_ids_for_profile(PortfolioProfile.AGGRESSIVE)),
            "categories": {
                cat.value: len(self.get_ids_for_category(cat))
                for cat in AssetCategory
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# Imported directly by other modules: from ingestion.universe import UNIVERSE
# ---------------------------------------------------------------------------
UNIVERSE = UniverseManager()
