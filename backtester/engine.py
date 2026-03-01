import math

import numpy as np
import polars as pl

from .feedback import generate_failure_narrative
from .schema import GradingReport


def _validate_signals(signals: pl.Series, expected_len: int) -> None:
    """Validate signal Series values are in [-1.0, 1.0] and has correct length."""
    if len(signals) != expected_len:
        raise ValueError(
            f"Signal length ({len(signals)}) != DataFrame length ({expected_len})"
        )
    non_null = signals.drop_nulls()
    if len(non_null) > 0:
        min_val = float(non_null.min())
        max_val = float(non_null.max())
        if min_val < -1.0 or max_val > 1.0:
            raise ValueError(
                f"Signals out of range: min={min_val}, max={max_val}. "
                f"Must be in [-1.0, 1.0]."
            )


def run_backtest(
    df: pl.DataFrame,
    signals: pl.Series,
    init_cash: float = 10_000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 10.0,
    symbol: str = "BTC/USDT",
    strategy_name: str = "Unknown",
    candles_per_year: int = 35_040,
) -> GradingReport:
    """Execute a fully vectorized backtest using Polars. No for-loops.

    Execution model:
    - Signal decided at Close of candle T
    - Order filled at Open of candle T+1
    - Fee and slippage applied at fill time

    Args:
        df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume].
        signals: Series of signals in [-1.0, 1.0] (1=Long, -1=Short, 0=Flat, fractional=partial size).
        init_cash: Starting capital.
        fee_bps: Trading fee in basis points (10 = 0.1%).
        slippage_bps: Slippage in basis points (10 = 0.1%).
        symbol: Trading pair name for the report.
        strategy_name: Strategy class name for the report.
        candles_per_year: Number of candles per year for annualization
            (35040 for 15m, 8760 for 1h).

    Returns:
        GradingReport with all metrics and failure narrative.
    """
    _validate_signals(signals, len(df))

    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0

    # --- Build backtest DataFrame ---
    # T+1 execution: signal at row T becomes position at row T+1
    position = signals.cast(pl.Float64).shift(1).fill_null(0.0).alias("position")

    bt = df.select([
        pl.col("timestamp"),
        pl.col("open"),
        pl.col("close"),
    ]).with_columns([
        signals.alias("signal"),
        position,
    ])

    # Trade delta: change in position (where trades occur)
    bt = bt.with_columns(
        (pl.col("position") - pl.col("position").shift(1).fill_null(0))
        .alias("trade_delta")
    )

    # Slippage-adjusted fill price + fee cost
    bt = bt.with_columns([
        pl.when(pl.col("trade_delta") > 0)
          .then(pl.col("open") * (1.0 + slip_rate))
          .when(pl.col("trade_delta") < 0)
          .then(pl.col("open") * (1.0 - slip_rate))
          .otherwise(pl.col("open"))
          .alias("fill_price"),
        (pl.col("trade_delta").abs().cast(pl.Float64) * pl.col("open") * fee_rate)
          .alias("fee_cost"),
    ])

    # Market return per candle
    bt = bt.with_columns(
        pl.col("close").pct_change().fill_null(0.0).alias("market_return")
    )

    # Strategy return = position * market_return - fees (normalized)
    bt = bt.with_columns(
        (pl.col("position").cast(pl.Float64) * pl.col("market_return"))
        .alias("strategy_return_gross")
    )

    bt = bt.with_columns(
        (pl.col("strategy_return_gross") - pl.col("fee_cost") / init_cash)
        .alias("strategy_return")
    )

    # Cumulative equity curve
    bt = bt.with_columns(
        (1.0 + pl.col("strategy_return")).cum_prod().alias("equity_curve")
    )

    # === METRIC EXTRACTION ===

    equity = bt["equity_curve"]
    returns = bt["strategy_return"]
    n_candles = len(bt)

    # Total return
    final_equity = float(equity[-1]) if n_candles > 0 else 1.0
    total_return = final_equity - 1.0

    # CAGR
    years = n_candles / float(candles_per_year)
    if years > 0 and final_equity > 0:
        cagr = final_equity ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Max Drawdown (magnitude)
    running_max = equity.cum_max()
    drawdown = (equity - running_max) / running_max
    max_dd_magnitude = float(drawdown.min()) if n_candles > 0 else 0.0

    # Max Drawdown duration (longest consecutive underwater streak)
    is_in_dd = (equity < running_max).cast(pl.Int32)
    dd_group_boundaries = (is_in_dd != is_in_dd.shift(1).fill_null(0)).cum_sum()

    dd_df = pl.DataFrame({
        "in_dd": is_in_dd,
        "dd_group": dd_group_boundaries,
    }).filter(pl.col("in_dd") == 1)

    if len(dd_df) > 0:
        dd_durations = dd_df.group_by("dd_group").agg(pl.len().alias("duration"))
        max_dd_duration = int(dd_durations["duration"].max())
    else:
        max_dd_duration = 0

    # Sharpe Ratio (annualized)
    ret_mean = float(returns.mean()) if n_candles > 0 else 0.0
    ret_std = float(returns.std()) if n_candles > 1 else 0.0
    sharpe = (ret_mean / ret_std * math.sqrt(candles_per_year)) if ret_std > 0 else 0.0

    # Sortino Ratio (annualized, downside deviation)
    downside = returns.filter(returns < 0)
    if len(downside) > 1:
        downside_std = float(downside.std())
        sortino = (ret_mean / downside_std * math.sqrt(candles_per_year)) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Win Rate + Trade Count (per round-trip)
    trade_deltas = bt["trade_delta"]
    # A trade entry occurs when trade_delta != 0
    # Group consecutive positions into round-trip trades
    trade_entries = bt.filter(pl.col("trade_delta") != 0)
    total_trades = len(trade_entries) // 2  # entry + exit = 1 round trip

    # Per-trade PnL via position segments
    win_rate = _compute_win_rate(bt)

    # Failure Narrative
    narrative = generate_failure_narrative(df, bt, {
        "cagr": cagr,
        "max_drawdown": max_dd_magnitude,
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
    }, candles_per_year=candles_per_year)

    return GradingReport(
        symbol=symbol,
        strategy_name=strategy_name,
        cagr=round(cagr, 6),
        max_drawdown=round(max_dd_magnitude, 6),
        max_drawdown_duration_candles=max_dd_duration,
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        win_rate=round(win_rate, 4),
        failure_narrative=narrative,
        total_trades=total_trades,
        total_return=round(total_return, 6),
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        candles_evaluated=n_candles,
    )


def _compute_win_rate(bt: pl.DataFrame) -> float:
    """Compute win rate from round-trip trades using vectorized operations.

    Each trade segment is a contiguous block of non-zero position.
    PnL for each segment is the product of (1 + strategy_return) across
    the segment, minus 1.
    """
    positions = bt["position"]
    if positions.n_unique() <= 1:
        return 0.0

    # Mark the start of each new trade segment by direction flips (sign changes),
    # not magnitude changes — supports fractional position sizing
    pos_sign = positions.sign().cast(pl.Int32)
    sign_changed = (pos_sign != pos_sign.shift(1).fill_null(0)).cast(pl.Int32)
    trade_id = sign_changed.cum_sum().alias("trade_id")

    trades_df = bt.with_columns(trade_id).filter(pl.col("position") != 0)

    if len(trades_df) == 0:
        return 0.0

    # Compute per-trade cumulative return
    trade_pnl = trades_df.group_by("trade_id").agg([
        (pl.col("strategy_return") + 1.0).product().alias("gross_return"),
    ]).with_columns(
        (pl.col("gross_return") - 1.0).alias("pnl")
    )

    if len(trade_pnl) == 0:
        return 0.0

    n_winning = int(trade_pnl.filter(pl.col("pnl") > 0).height)
    n_total = int(trade_pnl.height)

    return n_winning / n_total if n_total > 0 else 0.0
