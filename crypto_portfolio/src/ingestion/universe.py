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
# The universe is stored in S3 Silver as a versioned Parquet file.
# It is refreshed weekly (or when market cap rankings shift significantly).
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
    GAMING_NFT     = "gaming_nft"
    AI_TOKEN       = "ai_token"
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
# Static universe seed
# This is the baseline classification. It is updated weekly from live data.
# CoinGecko IDs are canonical - they never change even if tickers do.
# ---------------------------------------------------------------------------

UNIVERSE_SEED: list[AssetDefinition] = [

    # -- Layer 1 (Low risk, Conservative eligible) --------------------------
    AssetDefinition("bitcoin",       "btc",   "Bitcoin",         AssetCategory.LAYER_1,        RiskTier.LOW,       1),
    AssetDefinition("ethereum",      "eth",   "Ethereum",        AssetCategory.LAYER_1,        RiskTier.LOW,       3),
    AssetDefinition("solana",        "sol",   "Solana",          AssetCategory.LAYER_1,        RiskTier.MEDIUM,    10),
    AssetDefinition("cardano",       "ada",   "Cardano",         AssetCategory.LAYER_1,        RiskTier.MEDIUM,    15),
    AssetDefinition("avalanche-2",   "avax",  "Avalanche",       AssetCategory.LAYER_1,        RiskTier.MEDIUM,    15),
    AssetDefinition("ripple",        "xrp",   "XRP",             AssetCategory.LAYER_1,        RiskTier.LOW,       10),
    AssetDefinition("binancecoin",   "bnb",   "BNB",             AssetCategory.LAYER_1,        RiskTier.LOW,       10),
    AssetDefinition("near",          "near",  "NEAR Protocol",   AssetCategory.LAYER_1,        RiskTier.MEDIUM,    25),
    AssetDefinition("polkadot",      "dot",   "Polkadot",        AssetCategory.LAYER_1,        RiskTier.MEDIUM,    20),
    AssetDefinition("cosmos",        "atom",  "Cosmos",          AssetCategory.LAYER_1,        RiskTier.MEDIUM,    25),

    # -- Layer 2 (Medium risk, Balanced eligible) ---------------------------
    AssetDefinition("matic-network", "matic", "Polygon",         AssetCategory.LAYER_2,        RiskTier.MEDIUM,    20),
    AssetDefinition("arbitrum",      "arb",   "Arbitrum",        AssetCategory.LAYER_2,        RiskTier.MEDIUM,    30),
    AssetDefinition("optimism",      "op",    "Optimism",        AssetCategory.LAYER_2,        RiskTier.MEDIUM,    40),
    AssetDefinition("starknet",      "strk",  "Starknet",        AssetCategory.LAYER_2,        RiskTier.MEDIUM,    50),
    AssetDefinition("base",          "base",  "Base",            AssetCategory.LAYER_2,        RiskTier.MEDIUM,    50),

    # -- DeFi (Medium risk, Balanced eligible) ------------------------------
    AssetDefinition("uniswap",       "uni",   "Uniswap",         AssetCategory.DEFI,           RiskTier.MEDIUM,    30),
    AssetDefinition("aave",          "aave",  "Aave",            AssetCategory.DEFI,           RiskTier.MEDIUM,    40),
    AssetDefinition("chainlink",     "link",  "Chainlink",       AssetCategory.DEFI,           RiskTier.MEDIUM,    20),
    AssetDefinition("lido-dao",      "ldo",   "Lido DAO",        AssetCategory.DEFI,           RiskTier.MEDIUM,    30),
    AssetDefinition("maker",         "mkr",   "MakerDAO",        AssetCategory.DEFI,           RiskTier.MEDIUM,    40),
    AssetDefinition("curve-dao-token","crv",  "Curve",           AssetCategory.DEFI,           RiskTier.MEDIUM,    50),
    AssetDefinition("jupiter-exchange-solana","jup","Jupiter",   AssetCategory.DEFI,           RiskTier.MEDIUM,    50),

    # -- Infrastructure (Medium risk, Balanced eligible) --------------------
    AssetDefinition("the-graph",     "grt",   "The Graph",       AssetCategory.INFRASTRUCTURE, RiskTier.MEDIUM,    60),
    AssetDefinition("filecoin",      "fil",   "Filecoin",        AssetCategory.INFRASTRUCTURE, RiskTier.MEDIUM,    40),
    AssetDefinition("render-token",  "rndr",  "Render",          AssetCategory.INFRASTRUCTURE, RiskTier.MEDIUM,    50),
    AssetDefinition("injective-protocol","inj","Injective",      AssetCategory.INFRASTRUCTURE, RiskTier.MEDIUM,    50),

    # -- AI Tokens (High risk, Aggressive eligible) -------------------------
    AssetDefinition("fetch-ai",      "fet",   "Fetch.ai",        AssetCategory.AI_TOKEN,       RiskTier.HIGH,      80),
    AssetDefinition("singularitynet","agix",  "SingularityNET",  AssetCategory.AI_TOKEN,       RiskTier.HIGH,      80),
    AssetDefinition("ocean-protocol","ocean", "Ocean Protocol",  AssetCategory.AI_TOKEN,       RiskTier.HIGH,      100),
    AssetDefinition("worldcoin-wld", "wld",   "Worldcoin",       AssetCategory.AI_TOKEN,       RiskTier.HIGH,      80),

    # -- Gaming / NFT (High risk, Aggressive eligible) ----------------------
    AssetDefinition("axie-infinity", "axs",   "Axie Infinity",   AssetCategory.GAMING_NFT,     RiskTier.HIGH,      100),
    AssetDefinition("the-sandbox",   "sand",  "The Sandbox",     AssetCategory.GAMING_NFT,     RiskTier.HIGH,      100),
    AssetDefinition("decentraland",  "mana",  "Decentraland",    AssetCategory.GAMING_NFT,     RiskTier.HIGH,      100),
    AssetDefinition("immutable-x",   "imx",   "Immutable",       AssetCategory.GAMING_NFT,     RiskTier.HIGH,      80),
    AssetDefinition("gala",          "gala",  "Gala",            AssetCategory.GAMING_NFT,     RiskTier.HIGH,      100),

    # -- Meme (Very high risk, Aggressive only) -----------------------------
    AssetDefinition("dogecoin",      "doge",  "Dogecoin",        AssetCategory.MEME,           RiskTier.VERY_HIGH, 15),
    AssetDefinition("shiba-inu",     "shib",  "Shiba Inu",       AssetCategory.MEME,           RiskTier.VERY_HIGH, 20),
    AssetDefinition("pepe",          "pepe",  "Pepe",            AssetCategory.MEME,           RiskTier.VERY_HIGH, 50),

    # -- Stablecoins (Excluded from all portfolios) -------------------------
    AssetDefinition("tether",        "usdt",  "Tether",          AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  1),
    AssetDefinition("usd-coin",      "usdc",  "USD Coin",        AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  1),
    AssetDefinition("dai",           "dai",   "Dai",             AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  1),
    AssetDefinition("true-usd",      "tusd",  "TrueUSD",         AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  1),
    AssetDefinition("pax-gold",      "paxg",  "PAX Gold",        AssetCategory.STABLECOIN,     RiskTier.EXCLUDED,  50),
]


# ---------------------------------------------------------------------------
# Universe manager class
# ---------------------------------------------------------------------------

class UniverseManager:
    """
    Manages the asset universe: filtering, classification, and S3 persistence.

    The universe is the contract between:
      - Ingestion (what to fetch)
      - Backtest engine (what assets are eligible per portfolio profile)
      - Validator (what assets must be present in each daily batch)
    """

    def __init__(self, assets: list[AssetDefinition] | None = None) -> None:
        self._assets = assets or UNIVERSE_SEED
        self._by_id: dict[str, AssetDefinition] = {a.coin_id: a for a in self._assets}

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

        Returns:
            List of coin_ids eligible for this profile.
        """
        eligible_tiers = PROFILE_ELIGIBLE_TIERS[profile]

        result = []
        for asset in self._assets:
            if asset.risk_tier not in eligible_tiers:
                continue

            # Apply live market cap rank filter if rankings available
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
        """
        The set of coin_ids the validator should expect in every daily batch.
        Scoped to assets within max_rank to match the configured universe_size.
        """
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

        Called in prices_transform.py before writing to Silver.
        """
        enriched = []
        for record in records:
            coin_id = record.get("id", "")
            asset   = self._by_id.get(coin_id)

            record["category"]  = asset.category.value  if asset else AssetCategory.OTHER.value
            record["risk_tier"] = asset.risk_tier.value if asset else RiskTier.HIGH.value

            # Add boolean flags for fast portfolio filtering in backtest
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
        """
        Convert universe to list of dicts for Parquet writing.
        Written to s3://bucket/silver/universe/version={v}/universe.parquet
        """
        return [
            {
                "coin_id":        a.coin_id,
                "symbol":         a.symbol,
                "name":           a.name,
                "category":       a.category.value,
                "risk_tier":      a.risk_tier.value,
                "max_mcap_rank":  a.max_mcap_rank,
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
