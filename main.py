"""Headless backtesting pipeline for the Agentic Trading System.

Usage:
    python main.py                    # Run both strategies, compare
    python main.py sma                # Run SMA crossover only
    python main.py squeeze            # Run volatility squeeze only
"""

import json
import sys
import time

from backtester.data import generate_mock_ohlcv
from backtester.engine import run_backtest
from backtester.strategy import Strategy
from strategies.sma_crossover import SmaCrossover
from strategies.volatility_squeeze import VolatilitySqueezeBreakout


def run_strategy(
    strategy: Strategy,
    df,
    symbol: str = "BTC/USDT",
) -> dict:
    """Run a single strategy through the Proving Ground and return results."""
    t0 = time.perf_counter()
    signals = strategy.generate_signals(df)
    report = run_backtest(
        df=df,
        signals=signals,
        init_cash=10_000.0,
        fee_bps=10.0,
        slippage_bps=10.0,
        symbol=symbol,
        strategy_name=strategy.__class__.__name__,
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

    # Generate shared dataset (same seed = same market for fair comparison)
    df = generate_mock_ohlcv(symbol="BTC/USDT", hours=8760, seed=42)

    if target == "walkforward":
        results = run_walkforward(df)
        output = [r["report"].model_dump() for r in results]
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
