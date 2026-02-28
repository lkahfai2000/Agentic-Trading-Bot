"""Meta Loop — Closed-Loop Strategy Optimization.

Connects the execution audit, live failure diagnosis, parameter mutation,
and Proving Ground backtester into a single automated pipeline.

Pipeline:
  1. Ingest audit.py results → measure real slippage vs 10bps assumption
  2. Update bridge_override.json if alpha leakage exceeds threshold
  3. Parse last 24h of JSONL logs → generate failure narrative
  4. Propose 3 parameter mutations targeted at the top failure modes
  5. Run baseline + 3 mutations through the Proving Ground at 15bps friction
  6. Hot-swap strategies/volatility_squeeze.py if best mutation wins by >10% Sharpe

Usage:
    python meta_loop.py                     # full run, reads ./logs/
    python meta_loop.py --log-dir logs/     # explicit log directory
    python meta_loop.py --dry-run           # analyse only, no file writes
    python meta_loop.py --friction-bps 20  # override slippage floor

Cron example (daily at 06:00 UTC):
    0 6 * * * cd /home/user/Agentic-Trading-Bot && python meta_loop.py >> meta_loop.log 2>&1

Exit codes:
    0 = success, no swap executed
    1 = unhandled error
    2 = hot-swap executed (trigger bridge restart if desired)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal project imports
# ---------------------------------------------------------------------------
import audit as _audit  # import as module to avoid name collisions
from alerts import TelegramAlerter
from backtester.data import generate_mock_ohlcv
from backtester.engine import run_backtest
from backtester.schema import GradingReport
from strategies.volatility_squeeze import VolatilitySqueezeBreakout

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVING_GROUND_SLIPPAGE_BPS: float = 15.0   # new friction floor
ALPHA_LEAK_THRESHOLD_BPS: float = 10.0       # trigger bridge override
SHARPE_IMPROVEMENT_THRESHOLD: float = 1.10   # 10% better Sharpe to swap
MIN_TRADE_COUNT: int = 5                     # minimum trades for a valid backtest
BRIDGE_OVERRIDE_PATH = Path("bridge_override.json")
STRATEGY_FILE = Path("strategies/volatility_squeeze.py")
STRATEGY_BACKUP = Path("strategies/volatility_squeeze.py.bak")

# Exact literal strings from the volatility_squeeze.py __init__ signature.
# Used by str.replace() to avoid float formatting ambiguity.
PARAM_SOURCE_STRINGS: dict[str, str] = {
    "bb_period": "20",
    "bb_std": "2.0",
    "atr_period": "14",
    "squeeze_lookback": "240",
    "squeeze_pctile": "0.10",
    "atr_stop_mult": "3.5",
    "release_window": "3",
    "candle_body_threshold": "0.30",
    "adx_period": "14",
    "adx_threshold": "25.0",
    "adx_weak_factor": "0.5",
    "be_atr_threshold": "999.0",
    "be_stop_tighten": "0.5",
    "bear_ema_span": "800",
    "bear_size_factor": "0.08",
    "bear_stop_factor": "1.0",
    "cb_atr_mult": "3.0",
    "cb_atr_lookback": "168",
}

# Canonical production defaults — used to build full_params for each mutation
BASELINE_PARAMS: dict[str, Any] = {
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "squeeze_lookback": 240,
    "squeeze_pctile": 0.10,
    "atr_stop_mult": 3.5,
    "release_window": 3,
    "candle_body_threshold": 0.30,
    "adx_period": 14,
    "adx_threshold": 25.0,
    "adx_weak_factor": 0.5,
    "be_atr_threshold": 999.0,
    "be_stop_tighten": 0.5,
    "bear_ema_span": 800,
    "bear_size_factor": 0.08,
    "bear_stop_factor": 1.0,
    "cb_atr_mult": 3.0,
    "cb_atr_lookback": 168,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuditMetrics:
    has_data: bool
    filled_count: int
    cancelled_count: int
    mean_alpha_leak_bps: float
    median_alpha_leak_bps: float
    mean_fill_time_s: float
    p95_fill_time_s: float
    stuck_count: int


@dataclass
class FailureStats:
    total_cycles: int
    cb_active_count: int
    bear_regime_count: int
    cancelled_count: int
    placed_count: int
    network_error_count: int
    other_error_count: int
    narrative: str
    ranked_modes: list[str]


@dataclass
class MutationCandidate:
    name: str
    rationale: str
    param_changes: dict[str, Any]       # delta only — e.g. {"cb_atr_mult": 3.5}
    full_params: dict[str, Any]         # complete kwargs for strategy constructor


@dataclass
class MutationResult:
    candidate: Optional[MutationCandidate]   # None for baseline
    label: str
    report: GradingReport
    sharpe_improvement_pct: float            # vs baseline; 0.0 for baseline itself
    meets_threshold: bool
    is_valid: bool                           # total_trades >= MIN_TRADE_COUNT


# ---------------------------------------------------------------------------
# Mutation lookup table
# ---------------------------------------------------------------------------
# Each entry: (name, rationale, param_changes)
# Evaluated top-to-bottom; first 3 triggered are selected.
# "Pad" entries always trigger as fallbacks when fewer than 3 primary modes fire.

_MUTATION_SPECS: list[tuple[str, str, str, dict[str, Any]]] = [
    # (priority_mode, name, rationale, param_changes)
    (
        "cb_active",
        "RAISE_CB_THRESHOLD",
        "CB fires frequently — raise ATR spike multiplier 3.0→3.5 to reduce false flattenings",
        {"cb_atr_mult": 3.5},
    ),
    (
        "bear_regime",
        "LOOSEN_BEAR_FILTER",
        "Bear filter suppresses many cycles — raise size factor 0.08→0.20 to capture bear bounces",
        {"bear_size_factor": 0.20},
    ),
    (
        "cancel_timeout",
        "TIGHTEN_ENTRY_FILTER",
        "Many cancelled orders — raise ADX threshold 25→30 to enter only in strong trends",
        {"adx_threshold": 30.0},
    ),
    (
        "high_slippage",
        "TIGHTER_SQUEEZE_GATE",
        "High retry slippage — tighten BBW squeeze 10th→7th percentile for higher-conviction entries",
        {"squeeze_pctile": 0.07},
    ),
    (
        "weak_trend",
        "REDUCE_WEAK_SIZING",
        "Weak-trend fills drag performance — raise ADX 25→28 and lower weak factor 0.5→0.35",
        {"adx_threshold": 28.0, "adx_weak_factor": 0.35},
    ),
    # Pad A & B — always available as fallbacks
    (
        "pad_a",
        "WIDER_BANDS",
        "Best-practice — widen Bollinger Bands std 2.0→2.2 to reduce whipsaw entries",
        {"bb_std": 2.2},
    ),
    (
        "pad_b",
        "WIDER_STOPS",
        "Best-practice — widen ATR stop multiplier 3.5→4.0 for more breathing room",
        {"atr_stop_mult": 4.0},
    ),
]


# ===================================================================
# Phase 1: Audit ingestion
# ===================================================================

def ingest_audit(log_dir: str) -> AuditMetrics:
    """Import audit functions directly and compute slippage metrics."""
    files = _audit.discover_log_files(log_dir)
    if not files:
        return AuditMetrics(
            has_data=False, filled_count=0, cancelled_count=0,
            mean_alpha_leak_bps=0.0, median_alpha_leak_bps=0.0,
            mean_fill_time_s=0.0, p95_fill_time_s=0.0, stuck_count=0,
        )

    all_records: list[dict] = []
    for f in files:
        all_records.extend(_audit.load_jsonl(f))

    parsed = _audit.classify_events(all_records)
    rows = _audit.build_audit_rows(parsed)

    filled = [r for r in rows if r.status == "FILLED"]
    cancelled = [r for r in rows if r.status == "CANCELLED"]

    if not filled:
        return AuditMetrics(
            has_data=False, filled_count=0, cancelled_count=len(cancelled),
            mean_alpha_leak_bps=0.0, median_alpha_leak_bps=0.0,
            mean_fill_time_s=0.0, p95_fill_time_s=0.0, stuck_count=0,
        )

    leaks = [abs(r.alpha_leak_bps) for r in filled if r.alpha_leak_bps is not None]
    times = [r.fill_time_s for r in filled if r.fill_time_s is not None]

    mean_leak = mean(leaks) if leaks else 0.0
    median_leak = median(leaks) if leaks else 0.0
    mean_time = mean(times) if times else 0.0
    p95_time = quantiles(times, n=20)[18] if len(times) >= 2 else (times[0] if times else 0.0)
    stuck = sum(1 for r in filled if r.is_stuck)

    return AuditMetrics(
        has_data=True,
        filled_count=len(filled),
        cancelled_count=len(cancelled),
        mean_alpha_leak_bps=mean_leak,
        median_alpha_leak_bps=median_leak,
        mean_fill_time_s=mean_time,
        p95_fill_time_s=p95_time,
        stuck_count=stuck,
    )


# ===================================================================
# Phase 2: Bridge config update
# ===================================================================

def write_bridge_override(
    metrics: AuditMetrics,
    threshold_bps: float = ALPHA_LEAK_THRESHOLD_BPS,
    dry_run: bool = False,
) -> Optional[dict]:
    """Write bridge_override.json if alpha leakage exceeds threshold.

    Returns the overrides dict if triggered, else None.
    """
    if not metrics.has_data or metrics.mean_alpha_leak_bps <= threshold_bps:
        return None

    mean_leak = metrics.mean_alpha_leak_bps
    new_retry_slip = round(min(max(mean_leak * 1.5, 15.0), 50.0), 1)
    new_cancel_after = min(max(300, int(metrics.p95_fill_time_s * 2)), 1200)

    overrides = {
        "retry_slippage_bps": new_retry_slip,
        "cancel_after_seconds": new_cancel_after,
    }
    payload = {
        "_meta": {
            "generated_by": "meta_loop.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mean_alpha_leak_bps": round(mean_leak, 2),
            "trade_count_analyzed": metrics.filled_count,
            "trigger": f"mean_alpha_leak_bps={mean_leak:.2f} > threshold={threshold_bps}",
        },
        "overrides": overrides,
    }

    if not dry_run:
        BRIDGE_OVERRIDE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return overrides


# ===================================================================
# Phase 3: Failure narrative
# ===================================================================

def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def build_failure_stats(log_dir: str, lookback_hours: int = 24) -> FailureStats:
    """Parse JSONL logs for the last N hours and count failure modes."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    files = _audit.discover_log_files(log_dir)
    if not files:
        return FailureStats(
            total_cycles=0, cb_active_count=0, bear_regime_count=0,
            cancelled_count=0, placed_count=0, network_error_count=0,
            other_error_count=0, narrative="No log files found.",
            ranked_modes=[],
        )

    all_records: list[dict] = []
    for f in files:
        all_records.extend(_audit.load_jsonl(f))

    parsed = _audit.classify_events(all_records)

    # Filter THEORETICAL records to last 24h
    recent_theo = [
        r for r in parsed.theoretical.values()
        if _parse_ts(r.ts) >= cutoff
    ]
    total_cycles = len(recent_theo)
    cb_active_count = sum(1 for r in recent_theo if r.cb_active)
    bear_regime_count = sum(1 for r in recent_theo if r.bear_regime)

    # Count cancelled/placed orders (all time, since logs may be single-day)
    placed_count = len(parsed.orders_placed)
    cancelled_count = len(parsed.orders_cancelled)

    # Count errors
    network_errors = sum(
        1 for e in parsed.order_errors
        if e.error_code in ("NetworkError", "RequestTimeout")
    )
    other_errors = len(parsed.order_errors) - network_errors

    # Rank failure modes
    ranked = _rank_failure_modes(
        total_cycles, cb_active_count, bear_regime_count,
        cancelled_count, placed_count, network_errors, other_errors,
    )

    # Build narrative
    narrative = _build_narrative(
        total_cycles, cb_active_count, bear_regime_count,
        cancelled_count, placed_count, network_errors, other_errors,
    )

    return FailureStats(
        total_cycles=total_cycles,
        cb_active_count=cb_active_count,
        bear_regime_count=bear_regime_count,
        cancelled_count=cancelled_count,
        placed_count=placed_count,
        network_error_count=network_errors,
        other_error_count=other_errors,
        narrative=narrative,
        ranked_modes=ranked,
    )


def _rank_failure_modes(
    total_cycles: int,
    cb_active_count: int,
    bear_regime_count: int,
    cancelled_count: int,
    placed_count: int,
    network_errors: int,
    other_errors: int,
) -> list[str]:
    """Rank failure modes by weighted severity. Returns mode keys in order."""
    denom = max(total_cycles, 1)
    cancel_denom = max(placed_count, 1)

    scores: dict[str, float] = {}

    # CB is weighted 2x because it forces position flattenings (missed trades)
    if cb_active_count > 0:
        scores["cb_active"] = (cb_active_count / denom) * 2.0
    if bear_regime_count > 0:
        scores["bear_regime"] = bear_regime_count / denom
    if cancelled_count > 0:
        scores["cancel_timeout"] = cancelled_count / cancel_denom
    if other_errors > 0:
        scores["order_errors"] = other_errors / denom

    # Derive secondary modes from scores
    if scores.get("cancel_timeout", 0) > 0.2:
        # High cancel rate often means chop / weak trend entries
        scores.setdefault("weak_trend", 0.05)

    if not scores:
        # No failures detected — use best-practice defaults
        scores["pad_a"] = 0.01
        scores["pad_b"] = 0.009

    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)

    # Always include fallback pads so we can produce 3 mutations
    for pad in ("pad_a", "pad_b"):
        if pad not in ranked:
            ranked.append(pad)

    return ranked


def _build_narrative(
    total_cycles: int,
    cb_active: int,
    bear_regime: int,
    cancelled: int,
    placed: int,
    net_errors: int,
    other_errors: int,
) -> str:
    parts: list[str] = []

    if total_cycles == 0:
        return "No trading cycles found in the last 24 hours."

    if cb_active > 0:
        pct = cb_active / total_cycles * 100
        parts.append(
            f"ATR circuit breaker fired on {pct:.0f}% of cycles ({cb_active}/{total_cycles}), "
            f"forcibly flattening positions during elevated volatility regimes."
        )

    if bear_regime > 0:
        pct = bear_regime / total_cycles * 100
        parts.append(
            f"Bear regime filter suppressed entries on {pct:.0f}% of cycles "
            f"({bear_regime}/{total_cycles}) — position sizes reduced to 8% of full exposure."
        )

    cancel_denom = max(placed, 1)
    if cancelled > 0:
        pct = cancelled / cancel_denom * 100
        parts.append(
            f"{pct:.0f}% of limit orders ({cancelled}/{placed}) were cancelled before fill "
            f"— current timeout may be too tight for order book depth."
        )

    if net_errors > 0:
        parts.append(f"{net_errors} network error(s) prevented order placement.")

    if other_errors > 0:
        parts.append(f"{other_errors} non-network order error(s) detected.")

    if not parts:
        parts.append(
            "No significant failure modes detected in the last 24 hours. "
            "Applying best-practice parameter improvements."
        )

    return " ".join(parts)


# ===================================================================
# Phase 4: Mutation proposals
# ===================================================================

def propose_mutations(stats: FailureStats) -> list[MutationCandidate]:
    """Select exactly 3 MutationCandidates based on ranked failure modes.

    Deterministic: same FailureStats always produce the same mutations.
    Deduplicates by param_changes frozenset to avoid identical candidates.
    """
    spec_by_mode: dict[str, tuple] = {
        spec[0]: spec for spec in _MUTATION_SPECS
    }

    selected: list[MutationCandidate] = []
    seen_change_sets: set[frozenset] = set()

    for mode in stats.ranked_modes:
        if len(selected) >= 3:
            break
        if mode not in spec_by_mode:
            continue

        _, name, rationale, param_changes = spec_by_mode[mode]
        key = frozenset(param_changes.items())
        if key in seen_change_sets:
            continue

        full_params = BASELINE_PARAMS | param_changes
        selected.append(MutationCandidate(
            name=name,
            rationale=rationale,
            param_changes=param_changes,
            full_params=full_params,
        ))
        seen_change_sets.add(key)

    # Safety: should never be needed given pad_a/pad_b, but be defensive
    for spec in _MUTATION_SPECS:
        if len(selected) >= 3:
            break
        key = frozenset(spec[3].items())
        if key not in seen_change_sets:
            full_params = BASELINE_PARAMS | spec[3]
            selected.append(MutationCandidate(
                name=spec[1], rationale=spec[2],
                param_changes=spec[3], full_params=full_params,
            ))
            seen_change_sets.add(key)

    return selected[:3]


# ===================================================================
# Phase 5: Proving Ground
# ===================================================================

def run_proving_ground(
    mutations: list[MutationCandidate],
    friction_bps: float = PROVING_GROUND_SLIPPAGE_BPS,
) -> list[MutationResult]:
    """Run baseline + 3 mutations through the backtest engine.

    Returns list of 4 MutationResult objects; baseline is first (index 0).
    """
    df = generate_mock_ohlcv(symbol="BTC/USDT", hours=8760, seed=42)

    # Baseline
    baseline_strategy = VolatilitySqueezeBreakout(**BASELINE_PARAMS)
    baseline_report = run_backtest(
        df=df,
        signals=baseline_strategy.generate_signals(df),
        init_cash=10_000.0,
        fee_bps=10.0,
        slippage_bps=friction_bps,
        symbol="BTC/USDT",
        strategy_name="BASELINE",
    )

    baseline_result = MutationResult(
        candidate=None,
        label="Baseline (Production)",
        report=baseline_report,
        sharpe_improvement_pct=0.0,
        meets_threshold=False,
        is_valid=baseline_report.total_trades >= MIN_TRADE_COUNT,
    )

    results: list[MutationResult] = [baseline_result]

    for m in mutations:
        strategy = VolatilitySqueezeBreakout(**m.full_params)
        report = run_backtest(
            df=df,
            signals=strategy.generate_signals(df),
            init_cash=10_000.0,
            fee_bps=10.0,
            slippage_bps=friction_bps,
            symbol="BTC/USDT",
            strategy_name=m.name,
        )

        base_sharpe = baseline_report.sharpe_ratio
        if base_sharpe and base_sharpe != 0:
            improvement = (report.sharpe_ratio - base_sharpe) / abs(base_sharpe) * 100.0
        else:
            improvement = 0.0

        results.append(MutationResult(
            candidate=m,
            label=f"M: {m.name}",
            report=report,
            sharpe_improvement_pct=improvement,
            meets_threshold=improvement >= (SHARPE_IMPROVEMENT_THRESHOLD - 1) * 100,
            is_valid=report.total_trades >= MIN_TRADE_COUNT,
        ))

    return results


# ===================================================================
# Phase 6: Hot-swap
# ===================================================================

def _format_new_value(param: str, new_val: Any) -> str:
    """Format a new parameter value to match Python source conventions."""
    old_val = BASELINE_PARAMS[param]
    if isinstance(old_val, int):
        return str(int(new_val))
    # Float: preserve the decimal point style (e.g. 3.5, 0.07, 28.0)
    formatted = f"{float(new_val)}"
    if "." not in formatted:
        formatted += ".0"
    return formatted


def _rewrite_constructor_defaults(source: str, param_changes: dict[str, Any]) -> str:
    """Rewrite constructor default values using exact str.replace() matching.

    Raises ValueError if any parameter pattern is not found in source
    (indicates the constructor signature has drifted from BASELINE_PARAMS).
    """
    result = source
    for param, new_val in param_changes.items():
        old_val = BASELINE_PARAMS[param]
        type_hint = "int" if isinstance(old_val, int) else "float"
        old_src = PARAM_SOURCE_STRINGS.get(param, str(old_val))
        new_src = _format_new_value(param, new_val)

        old_pattern = f"{param}: {type_hint} = {old_src}"
        new_pattern = f"{param}: {type_hint} = {new_src}"

        if old_pattern not in result:
            raise ValueError(
                f"Pattern '{old_pattern}' not found in source. "
                f"Constructor signature may have changed since meta_loop was written."
            )
        result = result.replace(old_pattern, new_pattern, 1)

    return result


def hotswap_decision(
    results: list[MutationResult],
    dry_run: bool = False,
) -> Optional[MutationResult]:
    """Evaluate the hot-swap gate and execute swap if threshold is met.

    Returns the winning MutationResult if swapped, else None.
    """
    baseline = results[0]
    valid_mutations = [r for r in results[1:] if r.is_valid and r.candidate is not None]

    if not valid_mutations:
        print("  No valid mutations. Keeping production unchanged.")
        return None

    best = max(valid_mutations, key=lambda r: r.report.sharpe_ratio)

    if not best.meets_threshold:
        print(
            f"  No mutation exceeds +10% Sharpe threshold "
            f"(best: {best.label} at {best.sharpe_improvement_pct:+.1f}%)."
        )
        return None

    if not baseline.is_valid:
        print(
            f"  Baseline has only {baseline.report.total_trades} trades "
            f"(< {MIN_TRADE_COUNT}). Cannot validate improvement safely."
        )
        return None

    print(f"  Winner: {best.label}  (Sharpe {best.sharpe_improvement_pct:+.1f}% ≥ 10% threshold)")
    print(f"  Changes: {best.candidate.param_changes}")

    if dry_run:
        print("  [DRY-RUN] Would execute hot-swap (skipped).")
        return best  # Return to allow alert preview

    # Read source
    source = STRATEGY_FILE.read_text(encoding="utf-8")

    # Build new source and validate before writing anything
    try:
        new_source = _rewrite_constructor_defaults(source, best.candidate.param_changes)
    except ValueError as e:
        print(f"  HOT-SWAP ABORTED: {e}")
        return None

    try:
        compile(new_source, str(STRATEGY_FILE), "exec")
    except SyntaxError as e:
        print(f"  HOT-SWAP ABORTED: compile() failed: {e}")
        return None

    # Backup then write
    STRATEGY_BACKUP.write_text(source, encoding="utf-8")
    print(f"  Backup: {STRATEGY_BACKUP}  ✓")

    STRATEGY_FILE.write_text(new_source, encoding="utf-8")
    print(f"  Swap:   {STRATEGY_FILE}  ✓")

    return best


# ===================================================================
# Output formatting
# ===================================================================

def _print_audit_metrics(m: AuditMetrics, threshold_bps: float) -> None:
    print(f"  Filled orders:     {m.filled_count:>4}    Cancelled: {m.cancelled_count}")
    if m.has_data:
        flag = "  ⚠  EXCEEDS threshold" if m.mean_alpha_leak_bps > threshold_bps else "  ✓  within threshold"
        print(f"  Mean alpha leak:   {m.mean_alpha_leak_bps:>7.2f} bps{flag}")
        print(f"  Median alpha leak: {m.median_alpha_leak_bps:>7.2f} bps")
        print(f"  Mean fill time:    {m.mean_fill_time_s:>7.1f}s")
        print(f"  P95 fill time:     {m.p95_fill_time_s:>7.1f}s")
        if m.stuck_count:
            print(f"  Stuck orders:      {m.stuck_count}  (>60s)")
    else:
        print("  No filled trades in logs — audit metrics unavailable.")


def _print_failure_stats(s: FailureStats) -> None:
    denom = max(s.total_cycles, 1)
    cancel_denom = max(s.placed_count, 1)
    print(f"  Cycles analysed:  {s.total_cycles}")
    print(f"  CB triggered:     {s.cb_active_count:>3} / {s.total_cycles}  ({s.cb_active_count/denom*100:.0f}%)")
    print(f"  Bear regime:      {s.bear_regime_count:>3} / {s.total_cycles}  ({s.bear_regime_count/denom*100:.0f}%)")
    print(f"  Cancelled orders: {s.cancelled_count:>3} / {s.placed_count}  ({s.cancelled_count/cancel_denom*100:.0f}%)")
    print(f"  Network errors:   {s.network_error_count}")
    print(f"  Other errors:     {s.other_error_count}")
    print()
    print(f"  Narrative:")
    # Word-wrap at 72 chars
    words = s.narrative.split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 74:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)


def _print_comparison_table(results: list[MutationResult]) -> None:
    COL_W = [28, 9, 8, 9, 8, 13]
    headers = ["Variant", "CAGR", "Sharpe", "MaxDD", "Trades", "vs Baseline"]

    def row(cells: list[str], suffix: str = "") -> str:
        parts = []
        for i, (c, w) in enumerate(zip(cells, COL_W)):
            align = "<" if i == 0 else ">"
            parts.append(f"{c:{align}{w}}")
        return " | ".join(parts) + suffix

    sep = "-+-".join("-" * w for w in COL_W)

    print(row(headers))
    print(sep)

    best_sharpe = max(r.report.sharpe_ratio for r in results[1:]) if len(results) > 1 else -999

    for r in results:
        is_best = (r.candidate is not None and r.report.sharpe_ratio == best_sharpe
                   and r.meets_threshold)
        suffix = "  ✓ WINNER" if is_best else ""
        cells = [
            r.label[:28],
            f"{r.report.cagr*100:+.1f}%",
            f"{r.report.sharpe_ratio:+.3f}",
            f"{r.report.max_drawdown*100:.1f}%",
            str(r.report.total_trades),
            "—" if r.candidate is None else f"{r.sharpe_improvement_pct:+.1f}%",
        ]
        print(row(cells, suffix))
        if not r.is_valid:
            print(f"  ⚠ {r.label}: only {r.report.total_trades} trades — below validity threshold")


# ===================================================================
# Entry point
# ===================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Meta Loop: closed-loop strategy optimization for VolatilitySqueezeBreakout"
    )
    p.add_argument("--log-dir", default="./logs", help="Bridge JSONL log directory (default: ./logs)")
    p.add_argument("--dry-run", action="store_true", help="Analyse only — no file writes or alerts")
    p.add_argument("--friction-bps", type=float, default=PROVING_GROUND_SLIPPAGE_BPS,
                   help=f"Slippage floor for Proving Ground (default: {PROVING_GROUND_SLIPPAGE_BPS})")
    p.add_argument("--threshold-bps", type=float, default=ALPHA_LEAK_THRESHOLD_BPS,
                   help=f"Alpha leak threshold for bridge override (default: {ALPHA_LEAK_THRESHOLD_BPS})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
    mode_tag = "DRY-RUN" if args.dry_run else "LIVE"

    W = 72
    print("=" * W)
    print(f"  META LOOP EXECUTION REPORT — {now_str}")
    print(f"  Mode: {mode_tag} | Log dir: {args.log_dir} | Friction: {args.friction_bps}bps")
    print("=" * W)
    print()

    exit_code = 0

    try:
        # -----------------------------------------------------------------
        # Phase 1: Audit ingestion
        # -----------------------------------------------------------------
        print("[1/6] AUDIT INGESTION")
        metrics = ingest_audit(args.log_dir)
        _print_audit_metrics(metrics, args.threshold_bps)
        print()

        # -----------------------------------------------------------------
        # Phase 2: Bridge config update
        # -----------------------------------------------------------------
        print("[2/6] BRIDGE CONFIG UPDATE")
        if not metrics.has_data:
            print("  Skipped — no filled trades to analyse.")
        else:
            ov = write_bridge_override(metrics, args.threshold_bps, dry_run=args.dry_run)
            if ov:
                tag = "  [DRY-RUN]" if args.dry_run else ""
                print(f"  retry_slippage_bps:   15.0 → {ov['retry_slippage_bps']}{tag}")
                print(f"  cancel_after_seconds: 300  → {ov['cancel_after_seconds']}{tag}")
                if not args.dry_run:
                    print(f"  → {BRIDGE_OVERRIDE_PATH}  (restart bridge to apply)")
            else:
                print(f"  Not needed — mean slippage {metrics.mean_alpha_leak_bps:.2f}bps ≤ {args.threshold_bps}bps")
        print()

        # -----------------------------------------------------------------
        # Phase 3: Failure narrative
        # -----------------------------------------------------------------
        print("[3/6] FAILURE NARRATIVE  (last 24h)")
        stats = build_failure_stats(args.log_dir, lookback_hours=24)
        _print_failure_stats(stats)
        print()

        # -----------------------------------------------------------------
        # Phase 4: Mutation proposals
        # -----------------------------------------------------------------
        print("[4/6] MUTATION PROPOSALS")
        mutations = propose_mutations(stats)
        for i, m in enumerate(mutations, 1):
            changes = ", ".join(f"{k}: {BASELINE_PARAMS[k]} → {v}" for k, v in m.param_changes.items())
            print(f"  M{i}  {m.name:<30}  {changes}")
            print(f"       {m.rationale}")
        print()

        # -----------------------------------------------------------------
        # Phase 5: Proving Ground
        # -----------------------------------------------------------------
        print(f"[5/6] PROVING GROUND  (slippage_bps={args.friction_bps})")
        results = run_proving_ground(mutations, friction_bps=args.friction_bps)
        _print_comparison_table(results)
        print()

        # -----------------------------------------------------------------
        # Phase 6: Hot-swap decision
        # -----------------------------------------------------------------
        print("[6/6] HOT-SWAP DECISION")
        winner = hotswap_decision(results, dry_run=args.dry_run)

        if winner is not None and winner.candidate is not None:
            # Send Telegram alert
            alerter = TelegramAlerter()
            baseline = results[0]
            param_changes_with_old = {
                k: (BASELINE_PARAMS[k], v)
                for k, v in winner.candidate.param_changes.items()
            }
            alerter.mutation_accepted(
                mutation_name=winner.candidate.name,
                param_changes=param_changes_with_old,
                old_cagr=baseline.report.cagr,
                new_cagr=winner.report.cagr,
                old_sharpe=baseline.report.sharpe_ratio,
                new_sharpe=winner.report.sharpe_ratio,
            )
            if not args.dry_run:
                print("  Alert: Telegram dispatched  ✓")
            else:
                print("  [DRY-RUN] Alert: Telegram would be dispatched")
            exit_code = 2

    except Exception:
        print("\nFATAL ERROR:", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1

    print()
    print("=" * W)
    print(f"  Meta loop complete.  Exit code: {exit_code}")
    print("=" * W)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
