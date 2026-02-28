from datetime import datetime, timedelta

import numpy as np
import polars as pl

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_ohlcv(filepath: str) -> pl.DataFrame:
    """Load OHLCV data from a .parquet file.

    Returns a Polars DataFrame with columns:
        timestamp (Datetime), open, high, low, close, volume (Float64).
    Sorted by timestamp ascending.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pl.read_parquet(filepath)

    # Normalize column names to lowercase
    df = df.rename({c: c.lower() for c in df.columns})

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Cast numeric columns to Float64
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in numeric_cols])

    # Ensure timestamp is Datetime
    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

    return df.sort("timestamp")


def generate_mock_ohlcv(
    symbol: str = "BTC/USDT",
    hours: int = 8760,
    start_price: float | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic 1h OHLCV data using geometric Brownian motion.

    Produces crypto-like price action with realistic volatility:
    - BTC: ~60% annualized vol, default start $40,000
    - ETH: ~75% annualized vol, default start $2,500

    Args:
        symbol: Trading pair name (used to set default price/vol).
        hours: Number of hourly candles to generate.
        start_price: Override starting price. Auto-detected from symbol if None.
        seed: Random seed for reproducibility.

    Returns:
        Polars DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)

    # Symbol-aware defaults
    if start_price is None:
        if "BTC" in symbol.upper():
            start_price = 40_000.0
        elif "ETH" in symbol.upper():
            start_price = 2_500.0
        else:
            start_price = 100.0

    if "ETH" in symbol.upper():
        annual_vol = 0.75
    elif "BTC" in symbol.upper():
        annual_vol = 0.60
    else:
        annual_vol = 0.50

    # GBM parameters (hourly)
    dt = 1.0 / 8760.0  # fraction of a year per hour
    mu = 0.0  # drift-neutral for unbiased mock data
    sigma = annual_vol

    # Generate log-returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(hours)

    # Build close prices from cumulative returns
    close_prices = start_price * np.exp(np.cumsum(log_returns))

    # Open = previous close (first open = start_price)
    open_prices = np.empty(hours)
    open_prices[0] = start_price
    open_prices[1:] = close_prices[:-1]

    # High/Low: extend beyond open-close range with intra-candle noise
    candle_body_high = np.maximum(open_prices, close_prices)
    candle_body_low = np.minimum(open_prices, close_prices)
    body_range = candle_body_high - candle_body_low + 1e-8  # avoid zero range

    high_prices = candle_body_high + rng.exponential(0.3 * body_range)
    low_prices = candle_body_low - rng.exponential(0.3 * body_range)
    low_prices = np.maximum(low_prices, 1e-2)  # floor at near-zero

    # Volume: log-normal, correlated with absolute returns
    abs_returns = np.abs(log_returns)
    base_volume = 1000.0 * start_price  # notional-scaled
    volume = base_volume * np.exp(
        rng.normal(0, 0.5, hours) + 2.0 * abs_returns / abs_returns.mean()
    )

    # Timestamps: hourly from 2024-01-01
    start_ts = datetime(2024, 1, 1)
    timestamps = [start_ts + timedelta(hours=i) for i in range(hours)]

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    ])
