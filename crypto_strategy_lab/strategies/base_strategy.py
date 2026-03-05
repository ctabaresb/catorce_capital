# strategies/base_strategy.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """
    Contract every strategy module must satisfy.

    generate_signal(df) -> boolean Series
      - True  = enter long at bar close
      - False = do not trade
      - Must apply regime gate internally
      - Must not look ahead (no use of fwd_ret_* columns)
      - Index must match df.index exactly
    """

    def __init__(self, params: dict = None):
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        pass

    def _regime_gate(self, df: pd.DataFrame) -> pd.Series:
        """
        Mandatory uptrend gate shared by all strategies.
        Returns boolean Series: True = regime acceptable for long entry.

        Two conditions must both hold:
          1. EMA 120m slope is positive (trend direction)
          2. Price is above EMA 120m (price confirmation)

        Uses exact column names from feature_list_decision_15m_btc_usd.json.
        """
        slope_ok = pd.to_numeric(
            df.get("ema_120m_slope_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        ) > self.params.get("min_slope_bps", 0.0)

        price_above_ema = (
            pd.to_numeric(df.get("mid", pd.Series(np.nan, index=df.index)), errors="coerce") >
            pd.to_numeric(df.get("ema_120m_last", pd.Series(np.nan, index=df.index)), errors="coerce")
        )

        return slope_ok & price_above_ema

    def _can_trade_gate(self, df: pd.DataFrame) -> pd.Series:
        """
        Hard gate: exclude bars flagged as missing or non-tradable.
        Always applied on top of strategy signal.
        """
        return pd.to_numeric(
            df.get("can_trade", pd.Series(1, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(int) == 1