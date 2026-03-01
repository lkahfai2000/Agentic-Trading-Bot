"""
Shared utility functions for the LDR Pilot Study.

All rolling window calculations use ONLY data available at the time of
calculation (no lookahead bias). Timestamps are UTC throughout.
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SIGNALS = PROJECT_ROOT / "data" / "signals"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

# ── Study parameters (locked before data collection) ──────────────────────────

STUDY_START = pd.Timestamp("2025-09-01", tz="UTC")
STUDY_END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

# Rolling window sizes
ROLLING_30D_HOURS = 30 * 24  # 720 hours
ROLLING_90D_HOURS = 90 * 24  # 2160 hours
ROLLING_4H = 4

# Signal thresholds (locked — not to be modified after data collection)
LDR_PERCENTILE = 90            # signal fires above this rolling percentile
OUTCOME_WINDOW_HOURS = 24      # forward window for outcome measurement
HIT_THRESHOLD_PCT = 3.0        # minimum move to count as a HIT
CASCADE_MULTIPLIER = 3.0       # liquidation must exceed Nx average for cascade flag
FUNDING_EXTREME_ZSCORE = 2.0   # |z| > 2 is considered extreme funding

# Kill criteria thresholds
KILL_ACCURACY = 55.0           # hit rate must exceed this (%)
KILL_MAGNITUDE = 3.0           # average move must exceed this (%)
KILL_TEMPORAL_SPREAD = 20.0    # rolling hit rate must not vary by more than this (pp)

# Transaction cost assumptions
MAKER_FEE = 0.0005    # 0.05%
TAKER_FEE = 0.0007    # 0.07%
SLIPPAGE_ENTRY = 0.005 # 0.5% during cascade conditions
SLIPPAGE_EXIT = 0.003  # 0.3%

# Instruments
INSTRUMENTS = {
    "primary": {"symbol": "BTCUSDT", "name": "BTC-USDT"},
    "secondary": {"symbol": "ETHUSDT", "name": "ETH-USDT"},
}

# ── API helpers ────────────────────────────────────────────────────────────────

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
COINGLASS_BASE = "https://open-api.coinglass.com"
DERIBIT_BASE = "https://www.deribit.com/api/v2"


def load_api_keys():
    """Load API keys from environment or .env file."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "coinglass": os.getenv("COINGLASS_API_KEY", ""),
        "deribit_id": os.getenv("DERIBIT_CLIENT_ID", ""),
        "deribit_secret": os.getenv("DERIBIT_CLIENT_SECRET", ""),
    }


def api_request(url, params=None, headers=None, max_retries=5, base_delay=1.0):
    """
    Make an API request with exponential backoff retry on rate limits and errors.

    Returns the parsed JSON response or raises after exhausting retries.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)

            if resp.status_code == 429:
                wait = base_delay * (2 ** attempt)
                logger.warning("Rate limited on %s, waiting %.1fs", url, wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            logger.warning("Request error on %s: %s — retrying in %.1fs", url, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def ts_to_ms(dt):
    """Convert a datetime or pd.Timestamp to milliseconds since epoch."""
    if isinstance(dt, pd.Timestamp):
        return int(dt.timestamp() * 1000)
    if isinstance(dt, datetime):
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return int(dt)


def ms_to_ts(ms):
    """Convert milliseconds since epoch to a UTC pd.Timestamp."""
    return pd.Timestamp(ms, unit="ms", tz="UTC")


# ── Data processing helpers ───────────────────────────────────────────────────

def rolling_mean(series, window):
    """Backward-looking rolling mean (no lookahead)."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def rolling_std(series, window):
    """Backward-looking rolling standard deviation (no lookahead)."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).std()


def rolling_percentile(series, window, percentile):
    """
    Backward-looking rolling percentile (no lookahead).

    Uses only data in the window ending at the current observation.
    """
    return series.rolling(window=window, min_periods=max(1, window // 2)).quantile(
        percentile / 100.0
    )


def zscore(value, mean, std):
    """Compute z-score, returning 0 where std is 0 or NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (value - mean) / std
    return np.where(np.isfinite(z), z, 0.0)


def forward_fill_to_hourly(df, timestamp_col="timestamp"):
    """
    Forward-fill a lower-frequency DataFrame to hourly resolution.

    The DataFrame must have a UTC-aware timestamp column.
    """
    df = df.set_index(timestamp_col).sort_index()
    hourly_index = pd.date_range(
        start=df.index.min().floor("h"),
        end=df.index.max().ceil("h"),
        freq="h",
        tz="UTC",
    )
    df = df.reindex(hourly_index, method="ffill")
    df.index.name = "timestamp"
    return df.reset_index()


def save_parquet(df, path):
    """Save DataFrame to parquet, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)


def load_parquet(path):
    """Load a parquet file into a DataFrame."""
    return pd.read_parquet(path)


def save_json(data, path):
    """Save data to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON to %s", path)


def setup_logging(level=logging.INFO):
    """Configure logging for the study scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
