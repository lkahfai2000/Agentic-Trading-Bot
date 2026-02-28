from abc import ABC, abstractmethod

import polars as pl


class Strategy(ABC):
    """Base class all LLM-generated strategies must inherit from.

    Contract:
    - Receive a Polars DataFrame with columns: timestamp, open, high, low, close, volume
    - Return a Polars Series (Float64 or Int32) of the same length
    - Signal values: continuous in [-1.0, 1.0]
      - 1.0 = full long, -1.0 = full short, 0.0 = flat
      - Fractional values (e.g. 0.5) = partial position sizing
      - Discrete {-1, 0, 1} Int32 signals remain valid (backward compatible)
    - Signals are evaluated at the CLOSE of each candle
    - Execution happens at the OPEN of the NEXT candle (handled by engine)
    - No for-loops over time-series data — use Polars expressions only
    """

    @abstractmethod
    def generate_signals(self, df: pl.DataFrame) -> pl.Series:
        """Accept OHLCV DataFrame, return signal Series.

        Args:
            df: Polars DataFrame with columns
                [timestamp, open, high, low, close, volume].

        Returns:
            pl.Series in [-1.0, 1.0]: 1.0 = full long, -1.0 = full short,
            0.0 = flat. Fractional values for partial sizing.
            Must have the same length as df.
        """
        ...
