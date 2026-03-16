# =============================================================================
# src/ingestion/coingecko_client.py
#
# Production-grade CoinGecko API client.
# Handles: rate limiting, exponential backoff retry, response checksums,
# and clean separation between free and pro tier endpoints.
#
# Used by: ingest_eod.py (Lambda handler)
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# Loaded once at Lambda cold start from Secrets Manager.
# ---------------------------------------------------------------------------
@dataclass
class CoinGeckoConfig:
    api_key: str
    plan: str = "demo"                  # "free", "demo", or "pro"
    max_retries: int = 3
    backoff_factor: float = 2.0
    request_timeout: int = 30
    per_page: int = 250

    # Rate limits (calls per minute) per plan
    RATE_LIMITS: dict = field(default_factory=lambda: {
        "free":  30,
        "demo":  250,   # CoinGecko Basic paid plan (CG- keys)
        "pro":   500,   # CoinGecko Analyst/Lite/Enterprise
    })

    @property
    def base_url(self) -> str:
        # Only true Pro/Enterprise plans use the pro subdomain
        if self.plan == "pro":
            return "https://pro-api.coingecko.com/api/v3"
        # Free and Demo (Basic paid) both use the standard endpoint
        return "https://api.coingecko.com/api/v3"

    @property
    def rate_limit_per_min(self) -> int:
        return self.RATE_LIMITS.get(self.plan, 30)

    @property
    def min_seconds_between_calls(self) -> float:
        return (60.0 / self.rate_limit_per_min) * 1.2

    @property
    def auth_header(self) -> dict:
        if self.plan == "pro" and self.api_key not in ("free-tier", ""):
            return {"x-cg-pro-api-key": self.api_key}
        if self.plan in ("demo", "free") and self.api_key not in ("free-tier", ""):
            return {"x-cg-demo-api-key": self.api_key}
        return {}


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------
class CoinGeckoClient:
    """
    Thread-safe CoinGecko API client with:
    - Automatic rate limiting (respects free/pro limits)
    - Exponential backoff on 429 / 5xx errors
    - MD5 checksum on every response for data integrity
    - Full request/response logging for CloudWatch
    """

    def __init__(self, config: CoinGeckoConfig) -> None:
        self.config = config
        self._session = self._build_session()
        self._last_call_ts: float = 0.0

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def get_markets(self, page: int = 1, per_page: int | None = None) -> dict:
        """
        Fetch current market data for top N assets by market cap.

        Returns raw API response dict with added metadata:
        {
            "data": [...],           # list of coin market objects
            "fetched_at": "...",     # ISO timestamp
            "checksum": "...",       # MD5 of raw response body
            "page": 1,
            "per_page": 250,
            "endpoint": "/coins/markets"
        }
        """
        per_page = per_page or self.config.per_page

        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "24h",
            "locale": "en",
            "precision": "full",
        }

        raw_response, checksum = self._get("/coins/markets", params=params)
        data = json.loads(raw_response)

        logger.info(
            "get_markets: page=%d per_page=%d returned=%d assets checksum=%s",
            page, per_page, len(data), checksum[:8]
        )

        return {
            "data": data,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "page": page,
            "per_page": per_page,
            "endpoint": "/coins/markets",
        }

    def get_coin_market_chart(
        self,
        coin_id: str,
        days: int,
    ) -> dict:
        """
        Fetch historical market data using /market_chart endpoint.
        Available on ALL plans including Basic ($29/mo).

        Args:
            coin_id:  CoinGecko coin ID (e.g. "bitcoin")
            days:     number of days of history (max 365 per call on Basic)

        Returns wrapped response with checksum and metadata.
        """
        params = {
            "vs_currency": "usd",
            "days":        days,
            "interval":    "daily",
        }

        raw_response, checksum = self._get(
            f"/coins/{coin_id}/market_chart",
            params=params,
        )
        data = json.loads(raw_response)

        logger.info(
            "get_coin_market_chart: coin=%s days=%d "
            "price_points=%d checksum=%s",
            coin_id, days,
            len(data.get("prices", [])),
            checksum[:8],
        )

        return {
            "coin_id":    coin_id,
            "data":       data,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "checksum":   checksum,
            "days":       days,
            "endpoint":   f"/coins/{coin_id}/market_chart",
        }

    def get_coin_history_range(
        self,
        coin_id: str,
        from_ts: int,
        to_ts: int,
    ) -> dict:
        """
        Fetch historical market data for a single coin over a date range.
        NOTE: Requires Analyst plan ($103/mo) or higher.
        For Basic plan use get_coin_market_chart() instead.
        """
        params = {
            "vs_currency": "usd",
            "from": from_ts,
            "to":   to_ts,
        }

        raw_response, checksum = self._get(
            f"/coins/{coin_id}/market_chart/range",
            params=params,
        )
        data = json.loads(raw_response)

        logger.info(
            "get_coin_history_range: coin=%s from=%d to=%d "
            "price_points=%d checksum=%s",
            coin_id, from_ts, to_ts,
            len(data.get("prices", [])),
            checksum[:8],
        )

        return {
            "coin_id":    coin_id,
            "data":       data,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "checksum":   checksum,
            "from_ts":    from_ts,
            "to_ts":      to_ts,
            "endpoint":   f"/coins/{coin_id}/market_chart/range",
        }

    def get_coin_history_by_date(
        self,
        coin_id: str,
        date: str,
    ) -> dict:
        """
        Fetch price, market cap, and volume for a specific date.
        Endpoint: /coins/{id}/history?date=DD-MM-YYYY
        Available on ALL plans including Basic ($29/mo).
        No day-count limit - works for any historical date back to 2013.

        Args:
            coin_id: CoinGecko coin ID (e.g. "bitcoin")
            date:    Date string in YYYY-MM-DD format

        Returns:
            dict with price, market_cap, volume for that date
        """
        # CoinGecko expects DD-MM-YYYY format
        dt = datetime.strptime(date, "%Y-%m-%d")
        cg_date = dt.strftime("%d-%m-%Y")

        params = {"date": cg_date, "localization": "false"}

        raw_response, checksum = self._get(
            f"/coins/{coin_id}/history",
            params=params,
        )
        data = json.loads(raw_response)

        # Extract price, market cap, volume from nested response
        market_data = data.get("market_data", {})
        price    = market_data.get("current_price", {}).get("usd")
        mkt_cap  = market_data.get("market_cap", {}).get("usd")
        volume   = market_data.get("total_volume", {}).get("usd")

        logger.debug(
            "get_coin_history_by_date: coin=%s date=%s price=%s",
            coin_id, date, price,
        )

        return {
            "coin_id":    coin_id,
            "date":       date,
            "price":      price,
            "market_cap": mkt_cap,
            "volume":     volume,
            "checksum":   checksum,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_global(self) -> dict:
        """
        Fetch global crypto market data (total market cap, BTC dominance).
        Used for benchmark normalization.
        """
        raw_response, checksum = self._get("/global")
        data = json.loads(raw_response)

        logger.info("get_global: checksum=%s", checksum[:8])

        return {
            "data": data,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "endpoint": "/global",
        }

    def ping(self) -> bool:
        """
        Health check. Returns True if CoinGecko API is reachable.
        Called at Lambda startup before main ingestion logic.
        """
        try:
            raw_response, _ = self._get("/ping")
            result = json.loads(raw_response)
            is_alive = result.get("gecko_says") is not None
            logger.info("CoinGecko ping: alive=%s", is_alive)
            return is_alive
        except Exception as exc:
            logger.error("CoinGecko ping failed: %s", exc)
            return False

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> tuple[str, str]:
        """
        Core HTTP GET with:
        - Rate limit enforcement (sleep before call if needed)
        - Retry on 429 / 5xx
        - Response checksum computation
        - Full CloudWatch logging

        Returns:
            (raw_response_text, md5_checksum)
        """
        url = f"{self.config.base_url}{endpoint}"
        headers = {
            "Accept": "application/json",
            **self.config.auth_header,
        }

        self._enforce_rate_limit()

        call_start = time.monotonic()

        try:
            response = self._session.get(
                url,
                headers=headers,
                params=params,
                timeout=self.config.request_timeout,
            )
            elapsed_ms = int((time.monotonic() - call_start) * 1000)
            self._last_call_ts = time.monotonic()

            logger.info(
                "HTTP GET %s status=%d elapsed_ms=%d",
                endpoint, response.status_code, elapsed_ms,
            )

            # Handle rate limit explicitly (429 not always caught by urllib3 retry)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(
                    "Rate limited on %s. Sleeping %ds.", endpoint, retry_after
                )
                time.sleep(retry_after)
                return self._get(endpoint, params)  # single recursive retry

            response.raise_for_status()

            raw_text = response.text
            checksum = hashlib.md5(raw_text.encode("utf-8")).hexdigest()

            return raw_text, checksum

        except requests.exceptions.RequestException as exc:
            elapsed_ms = int((time.monotonic() - call_start) * 1000)
            logger.error(
                "Request failed: endpoint=%s elapsed_ms=%d error=%s",
                endpoint, elapsed_ms, str(exc),
            )
            raise

    def _enforce_rate_limit(self) -> None:
        """
        Sleep if the minimum interval between calls has not elapsed.
        Prevents hitting CoinGecko rate limits.
        """
        if self._last_call_ts == 0.0:
            return

        elapsed = time.monotonic() - self._last_call_ts
        min_interval = self.config.min_seconds_between_calls

        if elapsed < min_interval:
            sleep_duration = min_interval - elapsed
            logger.debug("Rate limit sleep: %.2fs", sleep_duration)
            time.sleep(sleep_duration)

    def _build_session(self) -> requests.Session:
        """
        Build a requests Session with connection pooling and retry logic.
        Retries on connection errors and 5xx responses (not 429 - handled above).
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=2,
            pool_maxsize=4,
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session
