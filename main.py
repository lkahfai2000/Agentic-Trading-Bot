"""Headless backtesting pipeline for the Agentic Trading System.

Usage:
    python main.py

Runs the full pipeline: mock data generation -> signal generation -> backtest -> JSON report.
"""

import sys
import time

from backtester.data import generate_mock_ohlcv
from backtester.engine import run_backtest
from strategies.sma_crossover import SmaCrossover


def main() -> None:
    t0 = time.perf_counter()

    # 1. Generate mock BTC 1h data (1 year)
    df = generate_mock_ohlcv(symbol="BTC/USDT", hours=8760)

    # 2. Instantiate strategy and generate signals
    strategy = SmaCrossover(fast=10, slow=30)
    signals = strategy.generate_signals(df)

    # 3. Run backtest with 10bps fee + 10bps slippage
    report = run_backtest(
        df=df,
        signals=signals,
        init_cash=10_000.0,
        fee_bps=10.0,
        slippage_bps=10.0,
        symbol="BTC/USDT",
        strategy_name=strategy.__class__.__name__,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 4. Output JSON to stdout
    print(report.model_dump_json(indent=2))
    print(f"\n--- Pipeline completed in {elapsed_ms:.1f}ms ---", file=sys.stderr)


if __name__ == "__main__":
    main()
