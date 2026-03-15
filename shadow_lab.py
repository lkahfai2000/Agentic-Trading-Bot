"""Shadow Lab -- Strategy R&D Sandbox.

Runs unrestricted mutation experiments against live market data.
Does NOT modify production strategy files. Pure virtual PnL tracking.

Mutations include:
  - Deterministic parameter mutations (reused from meta_loop)
  - Unrestricted LLM structural mutations (code rewrite, 5-gate validated)

Alerts via Telegram when a variant achieves exceptional performance.

Usage:
    python shadow_lab.py                     # continuous R&D loop
    python shadow_lab.py --once              # single cycle then exit
    python shadow_lab.py --sharpe-threshold 2.0  # custom alert threshold
    python shadow_lab.py --dry-run           # no Telegram alerts

Exit codes:
    0 = normal completion (--once) or clean shutdown
    1 = fatal error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import polars as pl

from alerts import TelegramAlerter
from backtester.data import generate_mock_ohlcv
from backtester.engine import run_backtest
from backtester.schema import GradingReport
from bridge import BinanceTestnetClient
from meta_loop import (
    AuditMetrics,
    BASELINE_PARAMS,
    FailureStats,
    MutationCandidate,
    STRATEGY_FILE,
    _call_anthropic_api,
    _exec_and_generate,
    _extract_generate_signals,
    _read_current_params,
    _SHADOW_LLM_SYSTEM_PROMPT,
    _SHADOW_LLM_USER_TEMPLATE,
    _splice_method,
    build_failure_stats,
    ingest_audit,
    propose_mutations,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHADOW_SHARPE_THRESHOLD = 1.8
SHADOW_MIN_TRADES = 100
SHADOW_LOG_FILE = Path("logs/shadow_lab.jsonl")
DEFAULT_CYCLE_MINUTES = 15


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


@dataclass
class ShadowVariant:
    """Tracks a single mutation variant's cumulative virtual performance."""

    name: str
    reports: list[GradingReport] = field(default_factory=list)
    cumulative_trades: int = 0
    latest_sharpe: float = 0.0
    alerted: bool = False

    def update(self, report: GradingReport) -> None:
        self.reports.append(report)
        self.cumulative_trades += report.total_trades
        self.latest_sharpe = report.sharpe_ratio

    @property
    def meets_threshold(self) -> bool:
        return (
            self.latest_sharpe >= SHADOW_SHARPE_THRESHOLD
            and self.cumulative_trades >= SHADOW_MIN_TRADES
            and not self.alerted
        )


# ---------------------------------------------------------------------------
# Unrestricted LLM structural mutation
# ---------------------------------------------------------------------------


def _get_shadow_llm_mutation(
    metrics: AuditMetrics,
    stats: FailureStats,
    model: str = "claude-sonnet-4-20250514",
) -> Optional[MutationCandidate]:
    """Generate an unrestricted LLM structural mutation for the shadow lab.

    Uses the original code-rewrite prompt (preserved in meta_loop as
    _SHADOW_LLM_SYSTEM_PROMPT / _SHADOW_LLM_USER_TEMPLATE).
    Validates through the 5-gate Safe-Solder Protocol before accepting.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("  ANTHROPIC_API_KEY not set -- skipping LLM structural mutation.")
        return None

    strategy_source = STRATEGY_FILE.read_text(encoding="utf-8")
    try:
        method_source = _extract_generate_signals(strategy_source)
    except ValueError:
        print("  Cannot extract generate_signals from strategy file.")
        return None

    user_prompt = _SHADOW_LLM_USER_TEMPLATE.format(
        mean_slip=metrics.mean_alpha_leak_bps,
        median_slip=metrics.median_alpha_leak_bps,
        filled=metrics.filled_count,
        cancelled=metrics.cancelled_count,
        narrative=stats.narrative,
        method_source=method_source,
    )

    print(f"  Calling LLM ({model}) for structural mutation...")
    response = _call_anthropic_api(
        api_key, _SHADOW_LLM_SYSTEM_PROMPT, user_prompt, model=model,
    )
    if not response:
        return None

    # Parse code block
    code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if not code_match or "def generate_signals" not in code_match.group(1):
        print("  No valid Python code block in LLM response.")
        return None

    code = code_match.group(1)
    diag_match = re.search(
        r"DIAGNOSIS:\s*(.+?)(?=\nCHANGE:|\n```)", response, re.DOTALL,
    )
    diagnosis = diag_match.group(1).strip() if diag_match else "N/A"

    # Splice and validate (5-gate protocol)
    try:
        full_source = _splice_method(strategy_source, code)
        compile(full_source, "shadow_llm.py", "exec")
    except (ValueError, SyntaxError) as e:
        print(f"  Gate 1 FAIL (compile/splice): {e}")
        return None

    smoke_df = generate_mock_ohlcv(hours=8760, seed=42, timeframe_minutes=15)
    try:
        _exec_and_generate(full_source, smoke_df)
    except Exception as e:
        print(f"  Gate 2-5 FAIL (exec/generate): {e}")
        return None

    print("  LLM structural mutation validated (5 gates passed).")
    return MutationCandidate(
        name="SHADOW_LLM_STRUCTURAL",
        rationale=diagnosis,
        param_changes={},
        full_params={},
        is_llm=True,
        llm_source=full_source,
    )


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


def _log_result(
    candidate: MutationCandidate,
    report: GradingReport,
    variant: ShadowVariant,
) -> None:
    """Append one result record to the shadow lab JSONL log."""
    SHADOW_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "variant": candidate.name,
        "is_llm": candidate.is_llm,
        "sharpe": report.sharpe_ratio,
        "cagr": report.cagr,
        "max_dd": report.max_drawdown,
        "trades": report.total_trades,
        "cumulative_trades": variant.cumulative_trades,
        "param_changes": candidate.param_changes or None,
        "rationale": candidate.rationale[:200],
    }
    with open(SHADOW_LOG_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Core cycle
# ---------------------------------------------------------------------------


def run_shadow_cycle(
    client: BinanceTestnetClient,
    registry: dict[str, ShadowVariant],
    alerter: TelegramAlerter,
    log_dir: str,
    sharpe_threshold: float = SHADOW_SHARPE_THRESHOLD,
    dry_run: bool = False,
) -> None:
    """Run one shadow lab cycle: fetch data, mutate, backtest, track."""
    now = datetime.now(timezone.utc)
    print(f"\n{'=' * 60}")
    print(f"  SHADOW LAB CYCLE -- {now.strftime('%Y-%m-%dT%H:%M:%S UTC')}")
    print(f"{'=' * 60}")

    # 1. Fetch live OHLCV
    print("  [1] Fetching live OHLCV (15m, 4400 candles)...")
    try:
        raw = client.fetch_ohlcv("BTC/USDT", "15m", limit=4400)
    except Exception as e:
        print(f"  OHLCV fetch failed: {e}")
        return

    df = (
        pl.DataFrame(
            raw,
            schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        .with_columns(
            (pl.col("timestamp_ms") * 1_000)
            .cast(pl.Datetime("us"))
            .alias("timestamp"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        .drop("timestamp_ms")
        .sort("timestamp")
    )
    print(f"  Loaded {len(df)} candles ({df['timestamp'][0]} to {df['timestamp'][-1]})")

    # 2. Gather failure context
    print("  [2] Ingesting audit metrics and failure stats...")
    metrics = ingest_audit(log_dir)
    stats = build_failure_stats(log_dir, lookback_hours=4320)
    current_params = _read_current_params()

    # 3. Generate mutation candidates
    print("  [3] Generating mutations...")
    candidates: list[MutationCandidate] = []

    # 3a. Deterministic parameter mutations
    det_mutations = propose_mutations(stats, current_params=current_params)
    candidates.extend(det_mutations)
    print(f"  Deterministic: {len(det_mutations)} candidates")

    # 3b. Unrestricted LLM structural mutation
    llm_candidate = _get_shadow_llm_mutation(metrics, stats)
    if llm_candidate:
        candidates.append(llm_candidate)
        print("  LLM structural: 1 candidate")
    else:
        print("  LLM structural: none (skipped or failed)")

    if not candidates:
        print("  No candidates generated -- cycle complete.")
        return

    # 4. Backtest each candidate on live data
    print(f"\n  [4] Backtesting {len(candidates)} candidates...")
    for c in candidates:
        variant_key = c.name

        if c.is_llm and c.llm_source:
            # LLM structural: use exec'd source
            try:
                signals = _exec_and_generate(c.llm_source, df)
            except Exception as e:
                print(f"  {c.name}: exec failed -- {e}")
                continue
        else:
            # Parameter mutation: instantiate with full_params
            try:
                from strategies.volatility_squeeze import VolatilitySqueezeBreakout
                strategy = VolatilitySqueezeBreakout(**c.full_params)
                signals = strategy.generate_signals(df)
            except Exception as e:
                print(f"  {c.name}: signal generation failed -- {e}")
                continue

        report = run_backtest(
            df=df,
            signals=signals,
            init_cash=10_000.0,
            fee_bps=10.0,
            slippage_bps=15.0,
            symbol="BTC/USDT",
            strategy_name=c.name,
            candles_per_year=35_040,
        )

        # Update registry
        if variant_key not in registry:
            registry[variant_key] = ShadowVariant(name=c.name)
        registry[variant_key].update(report)
        variant = registry[variant_key]

        print(
            f"  {c.name:<30}  Sharpe={report.sharpe_ratio:+.3f}  "
            f"CAGR={report.cagr * 100:+.1f}%  Trades={report.total_trades}  "
            f"Cumulative={variant.cumulative_trades}"
        )

        # Log to JSONL
        _log_result(c, report, variant)

        # 5. Alert if threshold met
        if variant.meets_threshold and not dry_run:
            variant.alerted = True
            alerter._send(
                f"<b>SHADOW LAB -- Promising Variant</b>\n\n"
                f"Variant: <code>{c.name}</code>\n"
                f"Sharpe: <code>{report.sharpe_ratio:+.4f}</code> "
                f"(threshold: {sharpe_threshold})\n"
                f"Cumulative trades: {variant.cumulative_trades}\n"
                f"CAGR: <code>{report.cagr * 100:+.1f}%</code>\n"
                f"Rationale: {c.rationale[:200]}\n\n"
                f"Review and manually promote if appropriate."
            )
            print(
                f"    ** ALERT SENT -- Sharpe {report.sharpe_ratio:.3f} "
                f"over {variant.cumulative_trades} trades"
            )

    print(f"\n  Cycle complete. {len(registry)} variants tracked.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow Lab -- R&D mutation sandbox",
    )
    parser.add_argument("--once", action="store_true", help="Single cycle then exit")
    parser.add_argument(
        "--sharpe-threshold",
        type=float,
        default=SHADOW_SHARPE_THRESHOLD,
        help=f"Alert when Sharpe exceeds this (default: {SHADOW_SHARPE_THRESHOLD})",
    )
    parser.add_argument(
        "--cycle-minutes",
        type=int,
        default=DEFAULT_CYCLE_MINUTES,
        help=f"Minutes between cycles (default: {DEFAULT_CYCLE_MINUTES})",
    )
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--dry-run", action="store_true", help="No Telegram alerts")
    args = parser.parse_args()

    # Initialize Binance client
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    if not api_key or not api_secret:
        print(
            "FATAL: BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET not set",
            file=sys.stderr,
        )
        sys.exit(1)

    client = BinanceTestnetClient(api_key, api_secret)
    alerter = TelegramAlerter()
    registry: dict[str, ShadowVariant] = {}

    print("=" * 60)
    print("  SHADOW LAB STARTED")
    print(f"  Cycle interval: {args.cycle_minutes}m")
    print(f"  Alert threshold: Sharpe > {args.sharpe_threshold} over {SHADOW_MIN_TRADES} trades")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)

    if args.once:
        run_shadow_cycle(
            client=client,
            registry=registry,
            alerter=alerter,
            log_dir=args.log_dir,
            sharpe_threshold=args.sharpe_threshold,
            dry_run=args.dry_run,
        )
        return

    while True:
        try:
            run_shadow_cycle(
                client=client,
                registry=registry,
                alerter=alerter,
                log_dir=args.log_dir,
                sharpe_threshold=args.sharpe_threshold,
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            print("\nShadow Lab shutting down (Ctrl+C).")
            break
        except Exception:
            traceback.print_exc()
            print("  Shadow cycle error -- will retry next cycle.")

        time.sleep(args.cycle_minutes * 60)


if __name__ == "__main__":
    main()
