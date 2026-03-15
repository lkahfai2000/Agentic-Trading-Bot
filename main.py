"""Headless backtesting pipeline for the Agentic Trading System.

Usage:
    python main.py                    # Run both strategies, compare
    python main.py sma                # Run SMA crossover only
    python main.py squeeze            # Run volatility squeeze only
    python main.py mtf                # Run MTF squeeze (15m sniper filter)
    python main.py walkforward        # Walk-forward validation
"""

import json
import sys
import time

from backtester.data import generate_mock_ohlcv, generate_mock_ohlcv_mtf
from backtester.engine import run_backtest
from backtester.strategy import Strategy
from strategies.sma_crossover import SmaCrossover
from strategies.volatility_squeeze import VolatilitySqueezeBreakout

import polars as pl


def run_strategy(
    strategy: Strategy,
    df,
    symbol: str = "BTC/USDT",
    df_fast: pl.DataFrame | None = None,
    candles_per_year: int = 35_040,
) -> dict:
    """Run a single strategy through the Proving Ground and return results."""
    t0 = time.perf_counter()
    signals = strategy.generate_signals(df, df_fast=df_fast)
    report = run_backtest(
        df=df,
        signals=signals,
        init_cash=10_000.0,
        fee_bps=10.0,
        slippage_bps=10.0,
        symbol=symbol,
        strategy_name=strategy.__class__.__name__,
        candles_per_year=candles_per_year,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"report": report, "elapsed_ms": elapsed_ms}


def run_walkforward(
    df,
    is_ratio: float = 0.70,
) -> list[dict]:
    """Walk-forward validation: In-Sample (training) vs Out-of-Sample (testing).

    Splits data by row count, runs the same strategy on both splits,
    and reports metrics side-by-side to validate that BBW percentiles
    and ADX thresholds aren't overfit "lucky numbers."
    """
    n = len(df)
    split_idx = int(n * is_ratio)

    df_is = df.head(split_idx)
    df_oos = df.tail(n - split_idx)

    print(f"=== Walk-Forward Validation ===", file=sys.stderr)
    print(f"    Split: {split_idx} IS / {n - split_idx} OOS candles", file=sys.stderr)

    results = []
    for label, data in [("[IS] VolatilitySqueezeBreakout", df_is),
                        ("[OOS] VolatilitySqueezeBreakout", df_oos)]:
        strategy = VolatilitySqueezeBreakout()
        result = run_strategy(strategy, data, symbol="BTC/USDT")
        result["report"] = result["report"].model_copy(
            update={"strategy_name": label}
        )
        results.append(result)

        r = result["report"]
        print(f"\n  {label}:", file=sys.stderr)
        print(f"    CAGR:     {r.cagr*100:+.1f}%", file=sys.stderr)
        print(f"    Max DD:   {r.max_drawdown*100:.1f}%", file=sys.stderr)
        print(f"    Sharpe:   {r.sharpe_ratio:+.2f}", file=sys.stderr)
        print(f"    Trades:   {r.total_trades}", file=sys.stderr)
        print(f"    Win Rate: {r.win_rate:.0%}", file=sys.stderr)
        print(f"    Time:     {result['elapsed_ms']:.1f}ms", file=sys.stderr)

    return results


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Generate shared dataset (15m candles, same seed = same market for fair comparison)
    df = generate_mock_ohlcv(symbol="BTC/USDT", hours=8760, seed=42, timeframe_minutes=15)

    if target == "walkforward":
        results = run_walkforward(df)
        output = [r["report"].model_dump() for r in results]
        print(json.dumps(output, indent=2))
        return

    if target == "mtf":
        # Multi-timeframe comparison: 15m standalone vs 15m + 1h sniper filter
        df_1h, df_15m = generate_mock_ohlcv_mtf(
            symbol="BTC/USDT", hours=8760, seed=42
        )
        strategy = VolatilitySqueezeBreakout()

        print("=== MTF Comparison (seed=42) ===", file=sys.stderr)

        # Baseline: 15m only (no sniper filter)
        r_15m = run_strategy(strategy, df_15m)
        rep = r_15m["report"]
        print(f"\n  [15m Only]:", file=sys.stderr)
        print(f"    CAGR:     {rep.cagr*100:+.1f}%", file=sys.stderr)
        print(f"    Max DD:   {rep.max_drawdown*100:.1f}%", file=sys.stderr)
        print(f"    Sharpe:   {rep.sharpe_ratio:+.2f}", file=sys.stderr)
        print(f"    Trades:   {rep.total_trades}", file=sys.stderr)
        print(f"    Win Rate: {rep.win_rate:.0%}", file=sys.stderr)
        print(f"    Time:     {r_15m['elapsed_ms']:.1f}ms", file=sys.stderr)
        print(f"    Sniper:   OFF (15m only)", file=sys.stderr)

        # MTF: 15m + 1h sniper filter
        r_mtf = run_strategy(strategy, df_15m, df_fast=df_1h)
        rep = r_mtf["report"]
        sniper = getattr(strategy, "_sniper_meta", {})
        print(f"\n  [15m + 1h Sniper]:", file=sys.stderr)
        print(f"    CAGR:     {rep.cagr*100:+.1f}%", file=sys.stderr)
        print(f"    Max DD:   {rep.max_drawdown*100:.1f}%", file=sys.stderr)
        print(f"    Sharpe:   {rep.sharpe_ratio:+.2f}", file=sys.stderr)
        print(f"    Trades:   {rep.total_trades}", file=sys.stderr)
        print(f"    Win Rate: {rep.win_rate:.0%}", file=sys.stderr)
        print(f"    Time:     {r_mtf['elapsed_ms']:.1f}ms", file=sys.stderr)
        print(f"    Sniper:   {sniper.get('entries_gated', 0)} entries gated by 1h filter", file=sys.stderr)

        output = [
            {**r_15m["report"].model_dump(), "strategy_name": "Squeeze_15m_Only"},
            {**r_mtf["report"].model_dump(), "strategy_name": "Squeeze_15m+1h_Sniper"},
        ]
        print(json.dumps(output, indent=2))
        return

    strategies = {}
    if target in ("all", "sma"):
        strategies["SmaCrossover"] = SmaCrossover(fast=10, slow=30)
    if target in ("all", "squeeze"):
        strategies["VolatilitySqueezeBreakout"] = VolatilitySqueezeBreakout()

    results = {}
    for name, strategy in strategies.items():
        result = run_strategy(strategy, df)
        results[name] = result
        print(f"=== {name} ===", file=sys.stderr)
        print(f"    Time: {result['elapsed_ms']:.1f}ms", file=sys.stderr)

    # Output JSON array of all reports
    output = [r["report"].model_dump() for r in results.values()]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
