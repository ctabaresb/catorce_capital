# config/settings.py
"""
All tunable parameters in one place.
Load with: from config.settings import CONFIG
Override via env vars for EC2 deployment.
"""
import os

CONFIG = {
    # Feed
    "book": os.environ.get("BITSO_BOOK", "btc_usd"),
    "subscriptions": ["orders", "trades"],

    # Execution
    "execution_mode": os.environ.get("EXEC_MODE", "shadow"),   # shadow | paper | live
    "api_key": os.environ.get("BITSO_API_KEY", ""),
    "api_secret": os.environ.get("BITSO_API_SECRET", ""),
    "cancel_timeout_sec": float(os.environ.get("CANCEL_TIMEOUT", "3.0")),

    # Signal
    "strategy": os.environ.get("STRATEGY", "passive_mm"),      # passive_mm | obi_momentum | flow_momentum
    "obi_threshold": float(os.environ.get("OBI_THRESHOLD", "0.3")),
    "flow_threshold": float(os.environ.get("FLOW_THRESHOLD", "0.4")),
    "spread_bps_max": float(os.environ.get("SPREAD_MAX_BPS", "20.0")),
    "spread_bps_min": float(os.environ.get("SPREAD_MIN_BPS", "1.0")),
    "cooldown_sec": float(os.environ.get("SIGNAL_COOLDOWN", "1.0")),

    # Risk
    "max_position_btc": float(os.environ.get("MAX_POS_BTC", "0.05")),
    "max_order_size_btc": float(os.environ.get("MAX_ORDER_BTC", "0.01")),
    "max_daily_loss_usd": float(os.environ.get("MAX_DAILY_LOSS", "200.0")),
    "max_orders_per_minute": int(os.environ.get("MAX_OPM", "30")),
    "max_signal_age_sec": float(os.environ.get("MAX_SIGNAL_AGE", "2.0")),
    "shadow_mode": os.environ.get("EXEC_MODE", "shadow") == "shadow",

    # Logging
    "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    "log_dir": os.environ.get("LOG_DIR", "./logs"),

    # Feature computation interval
    "feature_interval_sec": float(os.environ.get("FEATURE_INTERVAL", "0.5")),

    # Staleness thresholds
    "book_stale_sec": float(os.environ.get("BOOK_STALE_SEC", "5.0")),

    # Trade tape
    "trade_tape_maxlen": int(os.environ.get("TAPE_MAXLEN", "5000")),
}
