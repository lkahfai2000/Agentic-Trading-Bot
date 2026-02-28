from abc import ABC, abstractmethod

import polars as pl


class Strategy(ABC):
    """Base class all LLM-generated strategies must inherit from.

    Contract:
    - Receive a Polars DataFrame with columns: timestamp, open, high, low, close, volume
    - Return a Polars Series (Int32) of the same length
    - Signal values: 1 = Long, -1 = Short, 0 = Flat
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
            pl.Series of Int32: 1 = Long, -1 = Short, 0 = Flat.
            Must have the same length as df.
        """
        ...
