import math

import polars as pl


def generate_failure_narrative(
    df: pl.DataFrame,
    bt: pl.DataFrame,
    metrics: dict,
) -> str:
    """Generate a concise 2-sentence Failure Narrative analyzing strategy
    performance relative to market regimes.

    Uses vectorized regime classification (no for-loops):
    - Trending: strong directional bias over rolling windows
    - Choppy/Mean-reverting: low directional consistency, high volatility
    - Low-vol consolidation: low volatility, no clear direction

    Args:
        df: Original OHLCV DataFrame.
        bt: Backtest DataFrame with equity_curve, strategy_return, position columns.
        metrics: Dict with keys: cagr, max_drawdown, sharpe_ratio,
                 total_return, total_trades, win_rate.

    Returns:
        2-sentence string suitable for LLM consumption.
    """
    cagr = metrics.get("cagr", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    total_return = metrics.get("total_return", 0.0)
    total_trades = metrics.get("total_trades", 0)
    win_rate = metrics.get("win_rate", 0.0)
    n_candles = len(df)

    # Handle degenerate cases
    if total_trades == 0:
        return (
            f"Strategy generated zero trades across {n_candles:,} candles, "
            f"producing no returns. "
            f"The signal logic may be too restrictive or contain a bug "
            f"preventing any entry conditions from being met."
        )

    # --- Regime classification (vectorized) ---
    regime_info = _classify_market_regime(df)
    dominant_regime = regime_info["dominant_regime"]
    regime_pcts = regime_info["regime_pcts"]

    # --- Identify worst-performing regime for the strategy ---
    worst_regime = _find_worst_regime(df, bt, regime_info)

    # --- Build narrative ---
    # Sentence 1: What happened
    cagr_pct = cagr * 100
    dd_pct = abs(max_dd) * 100
    ret_pct = total_return * 100

    if cagr >= 0.05:
        perf_desc = f"achieved {cagr_pct:+.1f}% CAGR with a Sharpe of {sharpe:.2f}"
    elif cagr >= -0.05:
        perf_desc = f"returned a near-flat {cagr_pct:+.1f}% CAGR with a Sharpe of {sharpe:.2f}"
    else:
        perf_desc = f"returned {cagr_pct:+.1f}% CAGR with a {dd_pct:.0f}% max drawdown"

    sentence1 = (
        f"Strategy {perf_desc} across {n_candles:,} candles "
        f"({total_trades} trades, {win_rate:.0%} win rate)."
    )

    # Sentence 2: Why — regime analysis
    if cagr < -0.05:
        # Poor performance — explain failure
        sentence2 = _explain_failure(
            dominant_regime, worst_regime, regime_pcts, dd_pct
        )
    elif cagr < 0.05:
        # Mediocre performance
        sentence2 = _explain_mediocre(dominant_regime, regime_pcts, sharpe)
    else:
        # Good performance
        sentence2 = _explain_success(dominant_regime, regime_pcts, sharpe)

    return f"{sentence1} {sentence2}"


def _classify_market_regime(df: pl.DataFrame) -> dict:
    """Classify market into regime buckets using rolling metrics.

    Returns dict with:
    - dominant_regime: str (the most common regime)
    - regime_pcts: dict[str, float] (fraction of time in each regime)
    - regime_series: pl.Series (per-candle regime label)
    """
    window = 168  # 1 week in hours

    # Compute all regime features and classification in a single select
    regime_df = df.select([
        # Rolling return (1-week directional bias)
        (pl.col("close") / pl.col("close").shift(window) - 1.0)
            .fill_null(0.0)
            .alias("rolling_ret"),
        # Rolling realized volatility (annualized)
        pl.col("close").pct_change()
            .fill_null(0.0)
            .rolling_std(window)
            .fill_null(0.0)
            .mul(math.sqrt(8760))
            .alias("rolling_vol"),
        # Directional consistency
        pl.col("close").pct_change()
            .fill_null(0.0)
            .sign()
            .rolling_mean(window)
            .fill_null(0.5)
            .alias("pos_frac"),
    ]).select(
        pl.when(
            (pl.col("rolling_ret").abs() > 0.05) & (pl.col("pos_frac").abs() > 0.05)
        )
        .then(pl.lit("trending"))
        .when(pl.col("rolling_vol") > 0.80)
        .then(pl.lit("high_vol_chop"))
        .otherwise(pl.lit("consolidation"))
        .alias("regime")
    )

    regime_series = regime_df["regime"]

    # Count regime proportions
    counts = regime_df.group_by("regime").agg(pl.len().alias("count"))
    total = len(df)

    regime_pcts = {}
    for row in counts.iter_rows():
        regime_pcts[row[0]] = row[1] / total

    dominant_regime = max(regime_pcts, key=regime_pcts.get)

    return {
        "dominant_regime": dominant_regime,
        "regime_pcts": regime_pcts,
        "regime_series": regime_series,
    }


def _find_worst_regime(
    df: pl.DataFrame,
    bt: pl.DataFrame,
    regime_info: dict,
) -> str:
    """Find the regime where the strategy performed worst."""
    regime_series = regime_info["regime_series"]
    strategy_returns = bt["strategy_return"]

    analysis_df = pl.DataFrame({
        "regime": regime_series,
        "strategy_return": strategy_returns,
    })

    regime_perf = analysis_df.group_by("regime").agg(
        pl.col("strategy_return").sum().alias("total_ret")
    )

    if len(regime_perf) == 0:
        return "unknown"

    worst_row = regime_perf.sort("total_ret").row(0)
    return worst_row[0]


def _explain_failure(
    dominant_regime: str,
    worst_regime: str,
    regime_pcts: dict,
    dd_pct: float,
) -> str:
    regime_labels = {
        "trending": "sustained trending",
        "high_vol_chop": "high-volatility choppy",
        "consolidation": "low-volatility consolidation",
    }

    worst_label = regime_labels.get(worst_regime, worst_regime)
    worst_pct = regime_pcts.get(worst_regime, 0) * 100

    if worst_regime == "high_vol_chop":
        mechanism = "where frequent reversals generated excessive whipsaws and fee drag"
    elif worst_regime == "consolidation":
        mechanism = "where the lack of directional movement starved the strategy of profitable entries"
    else:
        mechanism = "suggesting the signal logic failed to capture the prevailing directional moves"

    return (
        f"Performance degraded primarily during the {worst_label} regime "
        f"({worst_pct:.0f}% of candles), {mechanism}."
    )


def _explain_mediocre(
    dominant_regime: str,
    regime_pcts: dict,
    sharpe: float,
) -> str:
    regime_labels = {
        "trending": "trending",
        "high_vol_chop": "choppy, high-volatility",
        "consolidation": "range-bound consolidation",
    }
    dom_label = regime_labels.get(dominant_regime, dominant_regime)
    dom_pct = regime_pcts.get(dominant_regime, 0) * 100

    return (
        f"The market spent {dom_pct:.0f}% of the period in a {dom_label} regime, "
        f"and the strategy's near-zero Sharpe ({sharpe:.2f}) indicates it lacked "
        f"a clear edge in any regime."
    )


def _explain_success(
    dominant_regime: str,
    regime_pcts: dict,
    sharpe: float,
) -> str:
    regime_labels = {
        "trending": "trending",
        "high_vol_chop": "volatile",
        "consolidation": "consolidating",
    }
    dom_label = regime_labels.get(dominant_regime, dominant_regime)
    dom_pct = regime_pcts.get(dominant_regime, 0) * 100

    return (
        f"The strategy capitalized on the dominant {dom_label} regime "
        f"({dom_pct:.0f}% of candles) with a Sharpe of {sharpe:.2f}, "
        f"indicating a genuine edge in current market conditions."
    )
