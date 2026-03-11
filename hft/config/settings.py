"""
config/settings.py
==================
Central configuration for the hft lab.
All settings are read from environment variables or a .env file.
Never hardcode credentials — use .env locally, SSM on EC2.

Usage:
    from config.settings import cfg
    print(cfg.DATA_DIR)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

# Load .env file if present (local Mac dev)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on real env vars


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    return int(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


# ---------------------------------------------------------------------------
# Project root (works on Mac and EC2)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    # ---- Paths ----
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
    )
    RESULTS_DIR: Path = field(
        default_factory=lambda: Path(_env("RESULTS_DIR", str(PROJECT_ROOT / "results")))
    )
    LOG_DIR: Path = field(
        default_factory=lambda: Path(_env("LOG_DIR", str(PROJECT_ROOT / "logs")))
    )

    # ---- Execution mode ----
    EXEC_MODE: str = field(default_factory=lambda: _env("EXEC_MODE", "paper"))
    # "paper"  — no orders submitted, logs only
    # "shadow" — orders submitted to Bitso but zero size (uses order placement endpoint)
    # "live"   — real orders

    # ---- Bitso credentials ----
    BITSO_API_KEY: str = field(default_factory=lambda: _env("BITSO_API_KEY", ""))
    BITSO_API_SECRET: str = field(default_factory=lambda: _env("BITSO_API_SECRET", ""))
    BITSO_WS_URL: str = "wss://ws.bitso.com"
    BITSO_REST_URL: str = "https://api.bitso.com/v3"

    # ---- Lead exchange WebSocket URLs ----
    COINBASE_WS_URL: str = "wss://advanced-trade-ws.coinbase.com"
    BINANCEUS_WS_URL: str = "wss://stream.binance.us:9443/ws"

    # ---- Telegram alerts ----
    TELEGRAM_TOKEN: str = field(default_factory=lambda: _env("TELEGRAM_TOKEN", ""))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))

    # ---- Research defaults ----
    TRAIN_FRAC: float = 0.60
    OBI_LEVELS: list = field(default_factory=lambda: [1, 2, 3])
    TFI_WINDOWS_SEC: list = field(default_factory=lambda: [10, 30, 60])
    FWD_HORIZONS_SEC: list = field(default_factory=lambda: [1, 3, 5, 10])
    MIN_SPREAD_BPS: float = 0.5
    MAX_SPREAD_BPS: float = 15.0
    MIN_IC_TO_TRADE: float = 0.05
    MIN_TRADES_FOR_STATS: int = 30
    COOLDOWN_SEC: float = 5.0

    # ---- Risk limits (live trading) ----
    MAX_POS_ASSET: float = field(default_factory=lambda: _env_float("MAX_POS_ASSET", 0.001))
    MAX_DAILY_LOSS_USD: float = field(
        default_factory=lambda: _env_float("MAX_DAILY_LOSS_USD", 15.0)
    )
    SPREAD_MAX_BPS: float = field(default_factory=lambda: _env_float("SPREAD_MAX_BPS", 5.0))

    def __post_init__(self):
        # Ensure directories exist on import
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_live(self) -> bool:
        return self.EXEC_MODE == "live"

    @property
    def is_paper(self) -> bool:
        return self.EXEC_MODE == "paper"

    def validate_live_credentials(self):
        if not self.BITSO_API_KEY or not self.BITSO_API_SECRET:
            raise EnvironmentError(
                "BITSO_API_KEY and BITSO_API_SECRET must be set for live/shadow mode. "
                "Add them to your .env file or export them as environment variables."
            )

    def __repr__(self) -> str:
        return (
            f"Config(mode={self.EXEC_MODE}, data={self.DATA_DIR}, "
            f"results={self.RESULTS_DIR})"
        )


# Singleton
cfg = Config()
