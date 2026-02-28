import polars as pl

from backtester.strategy import Strategy


class SmaCrossover(Strategy):
    """Simple Moving Average crossover strategy.

    Goes long when the fast MA crosses above the slow MA,
    goes short when the fast MA crosses below the slow MA.

    This serves as the reference implementation for the Strategy interface.
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pl.DataFrame) -> pl.Series:
        result = df.select(
            pl.when(
                pl.col("close").rolling_mean(self.fast)
                > pl.col("close").rolling_mean(self.slow)
            )
            .then(1)
            .when(
                pl.col("close").rolling_mean(self.fast)
                < pl.col("close").rolling_mean(self.slow)
            )
            .then(-1)
            .otherwise(0)
            .fill_null(0)
            .cast(pl.Int32)
            .alias("signal")
        )

        return result["signal"]
