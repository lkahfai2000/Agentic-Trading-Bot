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
    timeframe_minutes: int = 15,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data using geometric Brownian motion.

    Produces crypto-like price action with realistic volatility:
    - BTC: ~60% annualized vol, default start $40,000
    - ETH: ~75% annualized vol, default start $2,500

    Args:
        symbol: Trading pair name (used to set default price/vol).
        hours: Duration of the generated dataset in hours.
        start_price: Override starting price. Auto-detected from symbol if None.
        seed: Random seed for reproducibility.
        timeframe_minutes: Candle interval in minutes (15 for 15m, 60 for 1h).

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

    # GBM parameters (timeframe-aware)
    n_bars = hours * 60 // timeframe_minutes
    dt = timeframe_minutes / (8760.0 * 60.0)  # fraction of a year per bar
    mu = 0.0  # drift-neutral for unbiased mock data
    sigma = annual_vol

    # Generate log-returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_bars)

    # Build close prices from cumulative returns
    close_prices = start_price * np.exp(np.cumsum(log_returns))

    # Open = previous close (first open = start_price)
    open_prices = np.empty(n_bars)
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
        rng.normal(0, 0.5, n_bars) + 2.0 * abs_returns / abs_returns.mean()
    )

    # Timestamps
    start_ts = datetime(2024, 1, 1)
    timestamps = [start_ts + timedelta(minutes=timeframe_minutes * i) for i in range(n_bars)]

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


def generate_mock_ohlcv_mtf(
    symbol: str = "BTC/USDT",
    hours: int = 8760,
    start_price: float | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate consistent 15m and 1h synthetic OHLCV data.

    Generates 15m bars via GBM first, then resamples to 1h using proper
    OHLCV aggregation (first open, max high, min low, last close, sum volume).
    This ensures the two timeframes are physically consistent — 1h candles
    are mathematically derived from the 15m candles.

    Uses an independent RNG sequence (does NOT share state with
    generate_mock_ohlcv) to preserve seed=42 determinism for existing code.

    Args:
        symbol: Trading pair name (used to set default price/vol).
        hours: Number of 1h candles in the output. 15m output has 4x bars.
        start_price: Override starting price. Auto-detected from symbol if None.
        seed: Random seed for reproducibility.

    Returns:
        (df_1h, df_15m): Tuple of Polars DataFrames with matching price paths.
    """
    # Use a derived seed to avoid collisions with the 1h-only generator
    rng = np.random.default_rng(seed + 1_000_000)

    # Symbol-aware defaults (same as generate_mock_ohlcv)
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

    # GBM parameters (15-minute bars)
    bars_15m = hours * 4
    dt = 15.0 / (8760.0 * 60.0)  # fraction of a year per 15m bar
    mu = 0.0
    sigma = annual_vol

    # Generate 15m log-returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(bars_15m)
    close_prices = start_price * np.exp(np.cumsum(log_returns))

    open_prices = np.empty(bars_15m)
    open_prices[0] = start_price
    open_prices[1:] = close_prices[:-1]

    candle_body_high = np.maximum(open_prices, close_prices)
    candle_body_low = np.minimum(open_prices, close_prices)
    body_range = candle_body_high - candle_body_low + 1e-8

    high_prices = candle_body_high + rng.exponential(0.3 * body_range)
    low_prices = candle_body_low - rng.exponential(0.3 * body_range)
    low_prices = np.maximum(low_prices, 1e-2)

    abs_returns = np.abs(log_returns)
    base_volume = 1000.0 * start_price
    volume = base_volume * np.exp(
        rng.normal(0, 0.5, bars_15m) + 2.0 * abs_returns / abs_returns.mean()
    )

    # Timestamps: 15-minute intervals from 2024-01-01
    start_ts = datetime(2024, 1, 1)
    timestamps = [start_ts + timedelta(minutes=15 * i) for i in range(bars_15m)]

    df_15m = pl.DataFrame({
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

    # Resample 15m → 1h via proper OHLCV aggregation
    df_1h = df_15m.group_by_dynamic("timestamp", every="1h").agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]).sort("timestamp")

    return df_1h, df_15m
