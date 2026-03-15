"""Meta Loop — Closed-Loop Strategy Optimization.

Connects the execution audit, live failure diagnosis, parameter mutation,
and Proving Ground backtester into a single automated pipeline.

Pipeline:
  1. Ingest audit.py results → measure real slippage vs 10bps assumption
  2. Update bridge_override.json if alpha leakage exceeds threshold
  3. Parse last 180d of JSONL logs → generate failure narrative
  4. Propose 3 parameter mutations targeted at the top failure modes
  4b. (Optional) LLM Consultant proposes one structural code mutation
  5. Run baseline + mutations through the Proving Ground at 15bps friction
  6. Hot-swap strategies/volatility_squeeze.py if best mutation wins by >10% Sharpe

Usage:
    python meta_loop.py                     # full run, reads ./logs/
    python meta_loop.py --log-dir logs/     # explicit log directory
    python meta_loop.py --dry-run           # analyse only, no file writes
    python meta_loop.py --friction-bps 20  # override slippage floor
    python meta_loop.py --enable-llm       # activate LLM structural mutation

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
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Internal project imports
# ---------------------------------------------------------------------------
import audit as _audit  # import as module to avoid name collisions
from alerts import TelegramAlerter
from backtester.data import generate_mock_ohlcv, generate_mock_ohlcv_mtf  # noqa: F401
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
PROVING_GROUND_CACHE = Path("strategies/proving_ground_cache.json")

# Exact literal strings from the volatility_squeeze.py __init__ signature.
# Used by str.replace() to avoid float formatting ambiguity.
# All lookback params scaled 4x for 15m primary timeframe.
PARAM_SOURCE_STRINGS: dict[str, str] = {
    "bb_period": "80",
    "bb_std": "2.0",
    "atr_period": "56",
    "squeeze_lookback": "960",
    "squeeze_pctile": "0.10",
    "atr_stop_mult": "3.5",
    "release_window": "12",
    "candle_body_threshold": "0.30",
    "adx_period": "56",
    "adx_threshold": "25.0",
    "adx_weak_factor": "0.5",
    "be_atr_threshold": "999.0",
    "be_stop_tighten": "0.5",
    "bear_ema_span": "3200",
    "bear_size_factor": "0.08",
    "bear_stop_factor": "1.0",
    "cb_atr_mult": "3.0",
    "cb_atr_lookback": "672",
    "sniper_ema_fast": "8",
    "sniper_ema_slow": "32",
}

# Canonical production defaults — used to build full_params for each mutation
# All lookback params scaled 4x for 15m primary timeframe.
BASELINE_PARAMS: dict[str, Any] = {
    "bb_period": 80,
    "bb_std": 2.0,
    "atr_period": 56,
    "squeeze_lookback": 960,
    "squeeze_pctile": 0.10,
    "atr_stop_mult": 3.5,
    "release_window": 12,
    "candle_body_threshold": 0.30,
    "adx_period": 56,
    "adx_threshold": 25.0,
    "adx_weak_factor": 0.5,
    "be_atr_threshold": 999.0,
    "be_stop_tighten": 0.5,
    "bear_ema_span": 3200,
    "bear_size_factor": 0.08,
    "bear_stop_factor": 1.0,
    "cb_atr_mult": 3.0,
    "cb_atr_lookback": 672,
    "sniper_ema_fast": 8,
    "sniper_ema_slow": 32,
}


def _read_current_params(strategy_path: Path = STRATEGY_FILE) -> dict[str, Any]:
    """Parse current constructor defaults from the strategy source file.

    Uses the same regex pattern style as _rewrite_constructor_defaults() to
    ensure consistency between reading and writing.

    Falls back to BASELINE_PARAMS if the file cannot be read or a param is
    not found (safe degradation).
    """
    try:
        source = strategy_path.read_text(encoding="utf-8")
    except OSError:
        return dict(BASELINE_PARAMS)

    current = {}
    for param, baseline_val in BASELINE_PARAMS.items():
        type_hint = "int" if isinstance(baseline_val, int) else "float"
        if type_hint == "int":
            value_pattern = r"(\d+)"
        else:
            value_pattern = r"([\d]+\.[\d]+)"

        match = re.search(
            rf"{re.escape(param)}:\s*{re.escape(type_hint)}\s*=\s*{value_pattern}",
            source,
        )
        if match:
            raw = match.group(1)
            current[param] = int(raw) if type_hint == "int" else float(raw)
        else:
            current[param] = baseline_val

    return current


def _make_cache_key(
    current_params: dict[str, Any],
    mutations: list["MutationCandidate"],
) -> str:
    """Deterministic fingerprint of baseline params + actual proposed mutations.

    Includes mutation names and targets so the cache invalidates when code
    changes produce different proposals for the same inputs.
    """
    param_str = json.dumps(current_params, sort_keys=True)
    mut_parts = []
    for m in mutations:
        changes_str = json.dumps(m.param_changes, sort_keys=True)
        mut_parts.append(f"{m.name}:{changes_str}")
    mut_str = "|".join(mut_parts)
    return f"{param_str}||{mut_str}"


def _load_proving_ground_cache() -> dict:
    """Load the proving ground result cache (if it exists)."""
    try:
        return json.loads(PROVING_GROUND_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_proving_ground_cache(
    cache_key: str,
    results: list["MutationResult"],
) -> None:
    """Persist the proving ground results so identical runs can be skipped."""
    entries = []
    for r in results:
        entries.append({
            "label": r.label,
            "sharpe": r.report.sharpe_ratio,
            "cagr": r.report.cagr,
            "max_dd": r.report.max_drawdown,
            "trades": r.report.total_trades,
            "improvement_pct": r.sharpe_improvement_pct,
            "meets_threshold": r.meets_threshold,
        })
    cache = {
        "cache_key": cache_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": entries,
    }
    PROVING_GROUND_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


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
    is_llm: bool = False                # True for LLM structural mutations
    llm_source: Optional[str] = None    # full file source for LLM hot-swap


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
    # Sniper entry filter mutations
    (
        "high_slippage",
        "TIGHTEN_SNIPER_EMA",
        "High alpha leak — tighten sniper slow EMA 32→20 for faster entry confirmation",
        {"sniper_ema_slow": 20},
    ),
    (
        "weak_trend",
        "WIDEN_SNIPER_EMA",
        "Weak-trend entries — widen sniper slow EMA 32→48 to filter noise",
        {"sniper_ema_slow": 48},
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

# ---------------------------------------------------------------------------
# Incremental mutation definitions
# ---------------------------------------------------------------------------
# When fixed-target mutations are exhausted (target == current), generate
# incremental steps from the current value.  Each entry maps a failure mode
# to a list of (param, step, min, max, rationale_template) tuples.
# step is added to current value; negative steps explore the other direction.
# rationale_template uses {param}, {old}, {new}.

_INCREMENTAL_SPECS: dict[str, list[tuple[str, float, float, float, str]]] = {
    "bear_regime": [
        ("bear_size_factor", 0.06, 0.02, 0.50,
         "Incremental: raise bear sizing {old}→{new} to capture more bear-market entries"),
        ("bear_ema_span", -400, 800, 4800,
         "Incremental: shorten bear EMA {old}→{new} for faster regime detection"),
    ],
    "cb_active": [
        ("cb_atr_mult", 0.5, 2.0, 6.0,
         "Incremental: raise CB threshold {old}→{new} to reduce false flattenings"),
        ("cb_atr_lookback", 96, 288, 1344,
         "Incremental: lengthen CB lookback {old}→{new} for more stable baseline ATR"),
    ],
    "cancel_timeout": [
        ("adx_threshold", 2.0, 15.0, 40.0,
         "Incremental: raise ADX threshold {old}→{new} for higher-conviction entries"),
    ],
    "pad_a": [
        ("bb_std", 0.2, 1.5, 3.5,
         "Incremental: widen BB std {old}→{new} to reduce whipsaw entries"),
        ("bb_period", 20, 40, 200,
         "Incremental: lengthen BB period {old}→{new} for smoother bands"),
    ],
    "pad_b": [
        ("atr_stop_mult", 0.5, 2.0, 6.0,
         "Incremental: widen ATR stop {old}→{new} for more breathing room"),
        ("squeeze_pctile", -0.02, 0.03, 0.20,
         "Incremental: adjust squeeze percentile {old}→{new}"),
    ],
}


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

    # Filter THEORETICAL records to lookback window
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
        return "No trading cycles found in the lookback window."

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
            "No significant failure modes detected in the lookback window. "
            "Applying best-practice parameter improvements."
        )

    return " ".join(parts)


# ===================================================================
# Phase 4: Mutation proposals
# ===================================================================

def _make_incremental(
    mode: str,
    base: dict[str, Any],
    seen_params: set[str],
) -> MutationCandidate | None:
    """Generate an incremental mutation for a failure mode.

    Steps the first available parameter from its current value.
    Skips params already mutated in this batch.
    """
    specs = _INCREMENTAL_SPECS.get(mode)
    if not specs:
        return None

    for param, step, lo, hi, rationale_tpl in specs:
        if param in seen_params:
            continue
        current = base.get(param)
        if current is None:
            continue

        new_val = current + step
        # Clamp to bounds
        new_val = max(lo, min(hi, new_val))
        # Round to avoid float drift
        if isinstance(BASELINE_PARAMS.get(param), int):
            new_val = int(round(new_val))
        else:
            new_val = round(new_val, 4)

        if new_val == current:
            continue  # at boundary, try next param

        rationale = rationale_tpl.format(param=param, old=current, new=new_val)
        suffix = f"_{param.upper()}"
        return MutationCandidate(
            name=f"INCR{suffix}",
            rationale=rationale,
            param_changes={param: new_val},
            full_params=base | {param: new_val},
        )

    return None


def propose_mutations(
    stats: FailureStats,
    current_params: dict[str, Any] | None = None,
) -> list[MutationCandidate]:
    """Select up to 3 MutationCandidates based on ranked failure modes.

    First tries fixed-target mutations from _MUTATION_SPECS.  When a fixed
    target matches the current production value (already applied or no-op),
    falls back to incremental mutations that step the parameter from its
    current value.

    Deterministic: same FailureStats + current_params always produce
    the same mutations.
    """
    base = current_params if current_params is not None else BASELINE_PARAMS

    spec_by_mode: dict[str, tuple] = {
        spec[0]: spec for spec in _MUTATION_SPECS
    }

    selected: list[MutationCandidate] = []
    seen_change_sets: set[frozenset] = set()
    seen_params: set[str] = set()  # params already mutated (for incremental dedup)

    for mode in stats.ranked_modes:
        if len(selected) >= 3:
            break
        if mode not in spec_by_mode:
            continue

        _, name, rationale, param_changes = spec_by_mode[mode]

        # Skip mutations whose target already matches production or
        # where current has already moved past the fixed target
        effective = {k: v for k, v in param_changes.items() if base.get(k) != v}
        use_incremental = False
        if not effective:
            use_incremental = True
        else:
            # Check if any param has moved past the fixed target
            # (e.g., bear_size_factor is 0.26 but target is 0.20 — going backwards)
            for k, target_v in param_changes.items():
                baseline_v = BASELINE_PARAMS.get(k)
                current_v = base.get(k)
                if baseline_v is not None and current_v is not None:
                    direction = target_v - baseline_v  # positive = increase
                    if direction > 0 and current_v > target_v:
                        use_incremental = True
                        break
                    if direction < 0 and current_v < target_v:
                        use_incremental = True
                        break
        if use_incremental:
            incr = _make_incremental(mode, base, seen_params)
            if incr is not None:
                key = frozenset(incr.param_changes.items())
                if key not in seen_change_sets:
                    selected.append(incr)
                    seen_change_sets.add(key)
                    seen_params.update(incr.param_changes.keys())
            continue

        key = frozenset(effective.items())
        if key in seen_change_sets:
            continue

        full_params = base | effective
        selected.append(MutationCandidate(
            name=name,
            rationale=rationale,
            param_changes=effective,
            full_params=full_params,
        ))
        seen_change_sets.add(key)
        seen_params.update(effective.keys())

    # Fallback: fill remaining slots from specs + incremental
    for spec in _MUTATION_SPECS:
        if len(selected) >= 3:
            break
        param_changes = spec[3]
        effective = {k: v for k, v in param_changes.items() if base.get(k) != v}
        use_incr = False
        if not effective:
            use_incr = True
        else:
            for k, target_v in param_changes.items():
                baseline_v = BASELINE_PARAMS.get(k)
                current_v = base.get(k)
                if baseline_v is not None and current_v is not None:
                    direction = target_v - baseline_v
                    if direction > 0 and current_v > target_v:
                        use_incr = True
                        break
                    if direction < 0 and current_v < target_v:
                        use_incr = True
                        break
        if use_incr:
            incr = _make_incremental(spec[0], base, seen_params)
            if incr is not None:
                key = frozenset(incr.param_changes.items())
                if key not in seen_change_sets:
                    selected.append(incr)
                    seen_change_sets.add(key)
                    seen_params.update(incr.param_changes.keys())
            continue
        key = frozenset(effective.items())
        if key not in seen_change_sets:
            full_params = base | effective
            selected.append(MutationCandidate(
                name=spec[1], rationale=spec[2],
                param_changes=effective, full_params=full_params,
            ))
            seen_change_sets.add(key)
            seen_params.update(effective.keys())

    return selected[:3]


# ===================================================================
# Phase 4b: LLM Consultant (parameter-only mutations)
# ===================================================================

_LLM_SYSTEM_PROMPT = """\
You are a quantitative strategy parameter tuner for a BTC/USDT \
Volatility Squeeze Breakout strategy running on 15-minute candles.

Your job: analyse execution audit metrics and failure narratives, \
then recommend PARAMETER changes only. You must NOT rewrite code. \
You must NOT propose structural changes or new indicators.

HARD RULES:
1. You may ONLY recommend changes to existing constructor parameters.
2. Do NOT propose new parameters, new indicators, or code rewrites.
3. Provide your recommendations as a JSON object mapping parameter \
   names to new numeric values.
4. Each recommendation must include a rationale tied to the failure data.
5. Recommend between 1 and 3 parameter changes per response.
6. All parameter values must be numeric (int or float).

Available parameters and their current defaults:
  bb_period: int = 80         (Bollinger Band lookback)
  bb_std: float = 2.0         (BB standard deviation multiplier)
  atr_period: int = 56        (ATR lookback)
  squeeze_lookback: int = 960  (BBW ranking window)
  squeeze_pctile: float = 0.10 (BBW percentile for squeeze detection)
  atr_stop_mult: float = 3.5  (ATR stop-loss multiplier)
  release_window: int = 12    (squeeze release lookback)
  candle_body_threshold: float = 0.30  (momentum candle threshold)
  adx_period: int = 56        (ADX lookback)
  adx_threshold: float = 25.0 (ADX trend confirmation level)
  adx_weak_factor: float = 0.5 (position scale when ADX declining)
  be_atr_threshold: float = 999.0 (profit ATR for stop tightening)
  be_stop_tighten: float = 0.5 (stop tightening factor)
  bear_ema_span: int = 3200   (bear regime EMA span)
  bear_size_factor: float = 0.08 (position scale in bear regime)
  bear_stop_factor: float = 1.0 (stop factor in bear regime)
  cb_atr_mult: float = 3.0    (circuit breaker ATR threshold)
  cb_atr_lookback: int = 672   (circuit breaker baseline window)
  sniper_ema_fast: int = 8    (sniper fast EMA)
  sniper_ema_slow: int = 32   (sniper slow EMA)
"""

_LLM_USER_TEMPLATE = """\
## Current Audit Metrics
- Mean slippage: {mean_slip:.2f} bps (alpha leak — target: reduce below 10 bps)
- Median slippage: {median_slip:.2f} bps
- P95 fill time: {p95_time:.1f}s
- Mean fill time: {mean_time:.1f}s
- Filled orders: {filled}    Cancelled: {cancelled}
- Stuck orders (>60s): {stuck}

## Failure Narrative (last 180d)
{narrative}

## Current Parameter Values
{current_params_str}

## Task
Analyse the failure modes in the metrics and narrative. Recommend \
parameter changes that address the top failure mode(s). Explain your \
reasoning for each change. Do NOT suggest code changes.

## Output Format (STRICT — follow exactly)
DIAGNOSIS: <2-3 sentences explaining the primary failure mode>

PARAMS:
```json
{{"param_name": new_value, "param_name2": new_value2}}
```

RATIONALE: <1-2 sentences per parameter explaining why this change helps>
"""

# Preserved for shadow_lab.py — unrestricted structural mutation prompts
_SHADOW_LLM_SYSTEM_PROMPT = """\
You are a quantitative strategy engineer specialising in Polars-based \
vectorized trading signal generation for BTC/USDT on 15-minute candles.

HARD RULES — violating ANY of these invalidates your output:
1. Use ONLY Polars (import polars as pl). NEVER pandas, numpy for-loops, \
   .apply(), or .map_elements().
2. The method signature MUST be exactly:
       def generate_signals(self, df: pl.DataFrame, df_fast: pl.DataFrame | None = None) -> pl.Series:
   df is 15m OHLCV data (primary timeframe). df_fast is optional higher-TF data (may be None).
3. df has: timestamp, open, high, low, close, volume (all Float64). \
   The method adds computed columns (bb_mid, bb_upper, bb_lower, bb_std_val, \
   true_range, atr, bbw, atr_baseline, atr_spike_ratio, cb_active, etc.). \
   If you reference a column, you MUST ensure it is computed in a prior phase.
4. Return a pl.Series of Float64 in [-1.0, 1.0], same length as df. \
   Use .clip(-1.0, 1.0) on the final signal to guarantee this.
5. Reference ONLY self.xxx attributes defined in the existing __init__. \
   Do NOT add new constructor parameters. All lookback params are already \
   scaled for 15m bars (e.g., bb_period=80, atr_period=56, bear_ema_span=3200).
6. Preserve the existing 15-phase architecture. Add or modify phases — \
   do NOT delete existing phases unless replacing their purpose.
7. Polars API gotchas: \
   - Natural log: use .log(base=math.e) or (col / col).log(), NOT .ln() (does not exist). \
   - Use .ewm_mean() NOT .ewm().mean(). \
   - Use pl.col("x").rolling_mean(window_size=N) NOT .rolling(N).mean(). \
   - Use .shift(n) NOT .shift(periods=n).
8. df_fast is optional higher-timeframe data. When provided, aggregate it \
   before joining to the primary indicator DataFrame. The output must be \
   the same length as df.
"""

_SHADOW_LLM_USER_TEMPLATE = """\
## Current Audit Metrics
- Mean slippage: {mean_slip:.2f} bps
- Filled orders: {filled}    Cancelled: {cancelled}

## Failure Narrative
{narrative}

## Current generate_signals Method
```python
{method_source}
```

## Task
Diagnose the primary failure mode visible in the metrics and narrative, \
then propose ONE structural change to the signal generation logic. \
This must be a logic shift, NOT a parameter tweak. Examples of valid changes:
  - Add a volume-weighted VWAP confirmation filter
  - Add an RSI divergence filter rejecting entries when momentum diverges
  - Add an ATR contraction gate that waits for volatility to compress
  - Use order-flow imbalance (buy vs sell volume) to confirm direction
  - Add a higher-high/higher-low structure confirmation for long entries
  - Add a Bollinger Band squeeze-within-squeeze filter

## Output Format (STRICT — follow exactly)
DIAGNOSIS: <2-3 sentences explaining the primary failure mode>

CHANGE: <1 sentence describing your structural change>

```python
    def generate_signals(self, df: pl.DataFrame, df_fast: pl.DataFrame | None = None) -> pl.Series:
        <complete method body here>
```
"""


def _call_anthropic_api(
    api_key: str,
    system: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-20250514",
) -> Optional[str]:
    """Call Anthropic Messages API via urllib. Returns response text or None."""
    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": model,
        "max_tokens": 12000,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
    }).encode("utf-8")

    req = Request(url, data=payload, method="POST", headers={
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })

    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"]
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        print(f"  LLM API error {e.code}: {body}")
        return None
    except (URLError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"  LLM API call failed: {e}")
        return None


def _extract_generate_signals(source: str) -> str:
    """Extract the generate_signals method from the strategy file."""
    marker = "    def generate_signals(self, df: pl.DataFrame, df_fast: pl.DataFrame | None = None) -> pl.Series:"
    idx = source.find(marker)
    if idx == -1:
        raise ValueError("generate_signals not found in source")
    return source[idx:]


def _parse_llm_response(response: str) -> tuple[str, str, Optional[dict[str, Any]]]:
    """Parse diagnosis, rationale, and parameter recommendations from LLM response.

    Returns (diagnosis, rationale, param_dict_or_None).
    """
    # Extract diagnosis
    diag_match = re.search(r"DIAGNOSIS:\s*(.+?)(?=\nPARAMS:|\nRATIONALE:|\n```)", response, re.DOTALL)
    diagnosis = diag_match.group(1).strip() if diag_match else "No diagnosis provided."

    # Extract rationale
    rat_match = re.search(r"RATIONALE:\s*(.+?)$", response, re.DOTALL)
    rationale = rat_match.group(1).strip() if rat_match else "No rationale provided."

    # Extract JSON code block
    json_match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
    if not json_match:
        return diagnosis, rationale, None

    try:
        params = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return diagnosis, rationale, None

    if not isinstance(params, dict):
        return diagnosis, rationale, None

    # Validate all keys are known parameters with numeric values
    valid_params: dict[str, Any] = {}
    for k, v in params.items():
        if k in BASELINE_PARAMS and isinstance(v, (int, float)):
            valid_params[k] = type(BASELINE_PARAMS[k])(v)

    if not valid_params:
        return diagnosis, rationale, None

    return diagnosis, rationale, valid_params


def _splice_method(original_source: str, new_method: str) -> str:
    """Replace the generate_signals method body in the strategy file.

    Keeps everything before the method (docstring, imports, class, __init__)
    and replaces from 'def generate_signals' to EOF. This works because
    generate_signals is the LAST method in the class (lines 117-EOF).
    """
    marker = "    def generate_signals(self, df: pl.DataFrame, df_fast: pl.DataFrame | None = None) -> pl.Series:"
    idx = original_source.find(marker)
    if idx == -1:
        raise ValueError("generate_signals not found in source — cannot splice")
    prefix = original_source[:idx]
    return prefix + new_method.rstrip() + "\n"


def _exec_and_generate(full_source: str, df, df_fast=None) -> "pl.Series":
    """Execute modified source via exec() and generate signals.

    The source includes its own imports (polars, Strategy ABC), so the
    exec namespace starts empty — imports resolve via sys.path.
    """
    import polars as pl  # noqa: F811 — needed for type check below

    namespace: dict[str, Any] = {}
    exec(compile(full_source, "volatility_squeeze_llm.py", "exec"), namespace)

    strategy_cls = namespace.get("VolatilitySqueezeBreakout")
    if strategy_cls is None:
        raise RuntimeError("VolatilitySqueezeBreakout class not found in exec'd source")

    strategy = strategy_cls()
    signals = strategy.generate_signals(df, df_fast=df_fast)

    # Validate signal contract
    if not isinstance(signals, pl.Series):
        raise TypeError(f"Expected pl.Series, got {type(signals).__name__}")
    if len(signals) != len(df):
        raise ValueError(f"Signal length {len(signals)} != DataFrame length {len(df)}")

    vals = signals.drop_nulls()
    if len(vals) > 0:
        mn, mx = vals.min(), vals.max()
        if mn < -1.0 or mx > 1.0:
            # Clamp slightly out-of-range signals rather than rejecting outright.
            # LLM code often has minor scaling overshoot (~1.1x) that doesn't
            # indicate a logic bug.  Hard reject only for egregious violations.
            if mn < -2.0 or mx > 2.0:
                raise ValueError(f"Signals far out of range: min={mn}, max={mx}")
            signals = signals.clip(-1.0, 1.0)

    return signals


def get_llm_mutation(
    metrics: AuditMetrics,
    stats: FailureStats,
    model: str = "claude-sonnet-4-20250514",
    current_params: dict[str, Any] | None = None,
) -> Optional[MutationCandidate]:
    """Phase 4b orchestrator: call LLM for parameter recommendations.

    Production lockdown: the LLM only recommends parameter changes, not
    code rewrites.  The candidate goes through the standard deterministic
    hot-swap path (regex-based constructor default replacement).

    Returns MutationCandidate or None on any failure.
    All failures are non-fatal — the deterministic path continues.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("  ANTHROPIC_API_KEY not set — skipping LLM mutation.")
        return None

    base = current_params if current_params is not None else dict(BASELINE_PARAMS)

    # Format current params for the prompt
    current_params_str = "\n".join(
        f"  {k}: {v}" for k, v in sorted(base.items())
    )

    # Build prompt
    user_prompt = _LLM_USER_TEMPLATE.format(
        mean_slip=metrics.mean_alpha_leak_bps,
        median_slip=metrics.median_alpha_leak_bps,
        p95_time=metrics.p95_fill_time_s,
        mean_time=metrics.mean_fill_time_s,
        filled=metrics.filled_count,
        cancelled=metrics.cancelled_count,
        stuck=metrics.stuck_count,
        narrative=stats.narrative,
        current_params_str=current_params_str,
    )

    # Persist full prompt for dashboard "Brain Audit" viewer
    prompt_log = Path("logs/last_claude_prompt.txt")
    prompt_log.parent.mkdir(parents=True, exist_ok=True)
    prompt_log.write_text(
        f"=== SYSTEM PROMPT ===\n{_LLM_SYSTEM_PROMPT}\n\n"
        f"=== USER PROMPT ===\n{user_prompt}\n",
        encoding="utf-8",
    )

    # Call API
    print(f"  Calling {model}...")
    response = _call_anthropic_api(api_key, _LLM_SYSTEM_PROMPT, user_prompt, model=model)
    if not response:
        return None

    # Parse response
    diagnosis, rationale, param_dict = _parse_llm_response(response)
    print(f"  Diagnosis: {diagnosis[:120]}...")

    if param_dict is None:
        print("  No valid parameter recommendations in LLM response — skipping.")
        return None

    # Filter out no-op params (same as current value)
    effective = {k: v for k, v in param_dict.items() if base.get(k) != v}
    if not effective:
        print("  LLM recommended current values (no-op) — skipping.")
        return None

    changes_str = ", ".join(f"{k}: {base.get(k)} -> {v}" for k, v in effective.items())
    print(f"  Params:    {changes_str}")
    print(f"  Rationale: {rationale[:200]}")

    full_params = {**base, **effective}
    return MutationCandidate(
        name="LLM_PARAMETER",
        rationale=f"{diagnosis} => {rationale}",
        param_changes=effective,
        full_params=full_params,
        is_llm=True,
        llm_source=None,
    )


def _run_llm_backtest(
    candidate: MutationCandidate,
    baseline_report: GradingReport,
    friction_bps: float,
) -> Optional[MutationResult]:
    """Run the LLM mutation through the full Proving Ground backtest."""
    df = generate_mock_ohlcv(
        symbol="BTC/USDT", hours=8760, seed=42, timeframe_minutes=15,
    )

    try:
        signals = _exec_and_generate(candidate.llm_source, df)
    except Exception as e:
        print(f"  LLM backtest runtime error: {e}")
        return None

    report = run_backtest(
        df=df,
        signals=signals,
        init_cash=10_000.0,
        fee_bps=10.0,
        slippage_bps=friction_bps,
        symbol="BTC/USDT",
        strategy_name="LLM_STRUCTURAL",
        candles_per_year=35_040,
    )

    base_sharpe = baseline_report.sharpe_ratio
    if base_sharpe and base_sharpe != 0:
        improvement = (report.sharpe_ratio - base_sharpe) / abs(base_sharpe) * 100.0
    else:
        improvement = 0.0

    return MutationResult(
        candidate=candidate,
        label="L: LLM_STRUCTURAL",
        report=report,
        sharpe_improvement_pct=improvement,
        meets_threshold=improvement >= (SHARPE_IMPROVEMENT_THRESHOLD - 1) * 100,
        is_valid=report.total_trades >= MIN_TRADE_COUNT,
    )


# ===================================================================
# Phase 5: Proving Ground
# ===================================================================

def run_proving_ground(
    mutations: list[MutationCandidate],
    friction_bps: float = PROVING_GROUND_SLIPPAGE_BPS,
    current_params: dict[str, Any] | None = None,
) -> list[MutationResult]:
    """Run baseline + mutations through the backtest engine.

    Returns list of MutationResult objects; baseline is first (index 0).
    ``current_params`` should be the live strategy file defaults so the
    baseline matches actual production.
    """
    base = current_params if current_params is not None else BASELINE_PARAMS
    # Generate 15m data (primary timeframe) for the proving ground
    df = generate_mock_ohlcv(
        symbol="BTC/USDT", hours=8760, seed=42, timeframe_minutes=15,
    )

    # Baseline — use current production params, not hardcoded originals
    baseline_strategy = VolatilitySqueezeBreakout(**base)
    baseline_report = run_backtest(
        df=df,
        signals=baseline_strategy.generate_signals(df),
        init_cash=10_000.0,
        fee_bps=10.0,
        slippage_bps=friction_bps,
        symbol="BTC/USDT",
        strategy_name="BASELINE",
        candles_per_year=35_040,
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
            candles_per_year=35_040,
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
    """Rewrite constructor default values using regex matching.

    Uses regex to match the current value in the source (whatever it is),
    so hot-swap works even after a previous swap changed the value away
    from BASELINE_PARAMS.

    Raises ValueError if the parameter declaration is not found in source.
    """
    result = source
    for param, new_val in param_changes.items():
        old_val = BASELINE_PARAMS[param]
        type_hint = "int" if isinstance(old_val, int) else "float"
        new_src = _format_new_value(param, new_val)

        # Match the parameter with any current numeric value
        if type_hint == "int":
            value_pattern = r"\d+"
        else:
            value_pattern = r"[\d]+\.[\d]+"

        regex = re.compile(
            rf"({re.escape(param)}:\s*{re.escape(type_hint)}\s*=\s*){value_pattern}"
        )

        if not regex.search(result):
            raise ValueError(
                f"Parameter '{param}: {type_hint} = ...' not found in source. "
                f"Constructor signature may have changed."
            )
        result = regex.sub(rf"\g<1>{new_src}", result, count=1)

    return result


def _write_mutation_meta(
    best: MutationResult,
    baseline: MutationResult,
    failure_narrative: str,
) -> None:
    """Persist hot-swap metadata to strategies/mutation_meta.json for the dashboard."""
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mutation_name": best.candidate.name,
        "rationale": best.candidate.rationale,
        "is_llm": best.candidate.is_llm,
        "param_changes": {k: str(v) for k, v in best.candidate.param_changes.items()},
        "old_sharpe": baseline.report.sharpe_ratio,
        "new_sharpe": best.report.sharpe_ratio,
        "old_cagr": baseline.report.cagr,
        "new_cagr": best.report.cagr,
        "sharpe_improvement_pct": best.sharpe_improvement_pct,
        "failure_narrative": failure_narrative,
    }
    dest = Path("strategies/mutation_meta.json")
    dest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Meta:   {dest}  ✓")


def hotswap_decision(
    results: list[MutationResult],
    failure_narrative: str = "",
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
    if best.candidate.is_llm:
        print(f"  Type:    LLM structural mutation")
        print(f"  Reason:  {best.candidate.rationale[:120]}")
    else:
        print(f"  Changes: {best.candidate.param_changes}")

    if dry_run:
        print("  [DRY-RUN] Would execute hot-swap (skipped).")
        return best  # Return to allow alert preview

    # Read current source for backup
    source = STRATEGY_FILE.read_text(encoding="utf-8")

    if best.candidate.is_llm and best.candidate.llm_source:
        # ── LLM hot-swap: write pre-validated full source ──
        new_source = best.candidate.llm_source
        try:
            compile(new_source, str(STRATEGY_FILE), "exec")
        except SyntaxError as e:
            print(f"  HOT-SWAP ABORTED (LLM): compile() failed: {e}")
            return None
    else:
        # ── Deterministic hot-swap: str.replace on constructor defaults ──
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

    _write_mutation_meta(best, baseline, failure_narrative)

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
    p.add_argument("--enable-llm", action="store_true",
                   help="Enable LLM structural mutation via Anthropic API (requires ANTHROPIC_API_KEY)")
    p.add_argument("--llm-model", default="claude-sonnet-4-20250514",
                   help="Anthropic model for LLM mutations (default: claude-sonnet-4-20250514)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
    mode_tag = "DRY-RUN" if args.dry_run else "LIVE"
    llm_tag = " | LLM: ON" if args.enable_llm else ""

    W = 72
    print("=" * W)
    print(f"  META LOOP EXECUTION REPORT — {now_str}")
    print(f"  Mode: {mode_tag} | Log dir: {args.log_dir} | Friction: {args.friction_bps}bps{llm_tag}")
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
        print("[3/6] FAILURE NARRATIVE  (last 180d)")
        stats = build_failure_stats(args.log_dir, lookback_hours=4320)
        _print_failure_stats(stats)
        print()

        # -----------------------------------------------------------------
        # Read current production params from strategy file
        # -----------------------------------------------------------------
        current_params = _read_current_params()
        diffs = {k: (BASELINE_PARAMS[k], current_params[k])
                 for k in BASELINE_PARAMS if BASELINE_PARAMS[k] != current_params[k]}
        if diffs:
            print("  Production params (vs original defaults):")
            for k, (old, new) in diffs.items():
                print(f"    {k}: {old} → {new}  (previously swapped)")
            print()

        # -----------------------------------------------------------------
        # Phase 4: Mutation proposals
        # -----------------------------------------------------------------
        print("[4/6] MUTATION PROPOSALS")
        mutations = propose_mutations(stats, current_params=current_params)
        if not mutations:
            print("  No effective mutations — all candidates match production params.")
        for i, m in enumerate(mutations, 1):
            changes = ", ".join(f"{k}: {current_params.get(k, '?')} → {v}" for k, v in m.param_changes.items())
            print(f"  M{i}  {m.name:<30}  {changes}")
            print(f"       {m.rationale}")

        # Check cache — skip deterministic proving ground if identical to last run
        cache_key = _make_cache_key(current_params, mutations)
        cached = _load_proving_ground_cache()
        deterministic_cached = cached.get("cache_key") == cache_key

        if deterministic_cached:
            print()
            print("  ↳ Deterministic mutations unchanged since last run — cached.")
            best_cached = max(
                (r for r in cached["results"] if r["label"] != "Baseline (Production)"),
                key=lambda r: r["improvement_pct"],
                default=None,
            )
            if best_cached:
                print(f"    Best was {best_cached['label']} at {best_cached['improvement_pct']:+.1f}% (threshold: +10%)")
        print()

        # -----------------------------------------------------------------
        # Phase 4b: LLM Consultant (optional)
        # -----------------------------------------------------------------
        llm_candidate = None
        if args.enable_llm:
            print(f"[4b/6] LLM CONSULTANT  ({args.llm_model})")
            llm_candidate = get_llm_mutation(
                metrics, stats, model=args.llm_model,
                current_params=current_params,
            )
            if llm_candidate is None:
                print("  LLM mutation: not available (see above).")
            else:
                mutations.append(llm_candidate)
            print()

        # -----------------------------------------------------------------
        # Phase 5: Proving Ground
        # -----------------------------------------------------------------
        has_llm = llm_candidate is not None
        if deterministic_cached and not has_llm:
            # Nothing new to test — skip proving ground entirely
            print(f"[5/6] PROVING GROUND  (skipped — deterministic cached, no LLM candidate)")
            print()
            print("[6/6] HOT-SWAP DECISION")
            print("  Skipped — no new mutations to evaluate.")
            winner = None
        else:
            print(f"[5/6] PROVING GROUND  (slippage_bps={args.friction_bps})")

            # LLM param mutations are already appended to `mutations` list above
            results = run_proving_ground(mutations, friction_bps=args.friction_bps, current_params=current_params)

            _print_comparison_table(results)
            print()

            # Cache deterministic results for next run (only if we ran them)
            if not deterministic_cached:
                det_results = [r for r in results if not (r.candidate and r.candidate.is_llm)]
                _save_proving_ground_cache(cache_key, det_results)

            # -----------------------------------------------------------
            # Phase 6: Hot-swap decision
            # -----------------------------------------------------------
            print("[6/6] HOT-SWAP DECISION")
            winner = hotswap_decision(results, failure_narrative=stats.narrative, dry_run=args.dry_run)

        if winner is not None and winner.candidate is not None:
            # Send Telegram alert
            alerter = TelegramAlerter()
            baseline = results[0]
            if winner.candidate.is_llm:
                # LLM mutation: show rationale and param changes
                param_changes_for_alert = {
                    "type": ("deterministic", "LLM parameter"),
                    "change": ("—", winner.candidate.rationale[:80]),
                }
            else:
                param_changes_for_alert = {
                    k: (current_params.get(k, BASELINE_PARAMS[k]), v)
                    for k, v in winner.candidate.param_changes.items()
                }
            alerter.mutation_accepted(
                mutation_name=winner.candidate.name,
                param_changes=param_changes_for_alert,
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

    # Allow Telegram sender thread to flush queued messages before exit.
    # TelegramAlerter uses a daemon thread — it dies with the process.
    if exit_code == 2:
        import time
        time.sleep(3)

    print()
    print("=" * W)
    print(f"  Meta loop complete.  Exit code: {exit_code}")
    print("=" * W)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
