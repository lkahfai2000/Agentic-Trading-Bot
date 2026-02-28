"""Execution quality audit for the paper-trading bridge.

Parses JSONL logs produced by bridge.py's DualTrackLogger, joins
THEORETICAL and ATTEMPTED tracks by cycle_id, and reports slippage,
fill latency, and PnL reconciliation.

Usage:
    python audit.py                 # reads logs/ (default)
    python audit.py logs/           # explicit log directory
    python audit.py logs/bridge_20260228.jsonl  # single file
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Optional


# ---------------------------------------------------------------------------
# Data models — mirror DualTrackLogger field names from bridge.py
# ---------------------------------------------------------------------------

@dataclass
class TheoreticalRecord:
    cycle_id: str
    ts: str
    signal: float
    current_qty_btc: float
    target_qty_btc: float
    delta_qty_btc: float
    current_price_usd: float
    bear_regime: bool
    cb_active: bool
    max_usdt_allowed: float
    sniper_active: bool = False
    sniper_bull_confirmed: bool = False
    sniper_bear_confirmed: bool = False


@dataclass
class OrderPlacedRecord:
    cycle_id: str
    ts: str
    order_id: str
    side: str
    qty: float
    price: float
    is_retry: bool


@dataclass
class OrderFilledRecord:
    cycle_id: str
    ts: str
    order_id: str
    filled_qty: float
    avg_price: float


@dataclass
class BalanceCheckRecord:
    cycle_id: str
    ts: str
    usdt_free: float
    btc_total: float
    max_usdt_allowed: float


@dataclass
class OrderErrorRecord:
    cycle_id: str
    ts: str
    error_code: str
    binance_code: Optional[str]
    error_msg: str


@dataclass
class SystemRecord:
    event: str
    ts: str
    detail: str


# ---------------------------------------------------------------------------
# Parsed event container
# ---------------------------------------------------------------------------

@dataclass
class ParsedEvents:
    theoretical: dict[str, TheoreticalRecord] = field(default_factory=dict)
    orders_placed: dict[str, OrderPlacedRecord] = field(default_factory=dict)
    orders_filled: dict[str, OrderFilledRecord] = field(default_factory=dict)
    orders_cancelled: dict[str, str] = field(default_factory=dict)  # order_id -> reason
    order_errors: list[OrderErrorRecord] = field(default_factory=list)
    balance_checks: list[BalanceCheckRecord] = field(default_factory=list)
    system_events: list[SystemRecord] = field(default_factory=list)
    skipped: int = 0


# ---------------------------------------------------------------------------
# Computed audit row (one per ORDER_PLACED)
# ---------------------------------------------------------------------------

@dataclass
class TradeAuditRow:
    trade_num: int
    order_id: str
    cycle_id: str
    side: str
    is_retry: bool
    signal_price: float
    placed_price: float
    fill_price: Optional[float]
    qty: Optional[float]
    expected_notional: float
    actual_notional: Optional[float]
    slippage_bps: Optional[float]
    alpha_leak_bps: Optional[float]
    fill_time_s: Optional[float]
    is_stuck: bool
    status: str  # FILLED, CANCELLED, PENDING


# ===================================================================
# Section 1: Log file discovery & JSONL loading
# ===================================================================

def discover_log_files(target: str) -> list[Path]:
    """Find bridge JSONL log files from a CLI target path."""
    p = Path(target)
    if p.is_file() and p.suffix == ".jsonl":
        return [p]
    if p.is_dir():
        return sorted(p.glob("bridge_*.jsonl"))
    return sorted(Path(".").glob(target))


def load_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file, skipping malformed lines."""
    records: list[dict] = []
    bad = 0
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                print(f"  WARN: malformed JSON on line {line_num} in {path.name}",
                      file=sys.stderr)
    if bad:
        print(f"  WARN: {bad} malformed line(s) skipped in {path.name}",
              file=sys.stderr)
    return records


# ===================================================================
# Section 2: Event classification
# ===================================================================

def _safe_get(d: dict, key: str, default=None):
    return d.get(key, default)


def classify_events(raw_records: list[dict]) -> ParsedEvents:
    """Route raw JSON dicts into typed dataclass instances."""
    parsed = ParsedEvents()

    for raw in raw_records:
        track = _safe_get(raw, "track")
        event = _safe_get(raw, "event")

        try:
            if track == "THEORETICAL":
                rec = TheoreticalRecord(
                    cycle_id=raw["cycle_id"],
                    ts=raw["ts"],
                    signal=raw["signal"],
                    current_qty_btc=raw["current_qty_btc"],
                    target_qty_btc=raw["target_qty_btc"],
                    delta_qty_btc=raw["delta_qty_btc"],
                    current_price_usd=raw["current_price_usd"],
                    bear_regime=raw["bear_regime"],
                    cb_active=raw["cb_active"],
                    max_usdt_allowed=raw["max_usdt_allowed"],
                    sniper_active=raw.get("sniper_active", False),
                    sniper_bull_confirmed=raw.get("sniper_bull_confirmed", False),
                    sniper_bear_confirmed=raw.get("sniper_bear_confirmed", False),
                )
                parsed.theoretical[rec.cycle_id] = rec

            elif track == "ATTEMPTED":
                if event == "ORDER_PLACED":
                    rec = OrderPlacedRecord(
                        cycle_id=raw["cycle_id"],
                        ts=raw["ts"],
                        order_id=raw["order_id"],
                        side=raw["side"],
                        qty=raw["qty"],
                        price=raw["price"],
                        is_retry=raw.get("is_retry", False),
                    )
                    parsed.orders_placed[rec.order_id] = rec

                elif event == "ORDER_FILLED":
                    rec = OrderFilledRecord(
                        cycle_id=raw["cycle_id"],
                        ts=raw["ts"],
                        order_id=raw["order_id"],
                        filled_qty=raw["filled_qty"],
                        avg_price=raw["avg_price"],
                    )
                    parsed.orders_filled[rec.order_id] = rec

                elif event == "ORDER_CANCELLED":
                    parsed.orders_cancelled[raw["order_id"]] = raw.get("reason", "UNKNOWN")

                elif event == "ORDER_ERROR":
                    parsed.order_errors.append(OrderErrorRecord(
                        cycle_id=raw.get("cycle_id", ""),
                        ts=raw["ts"],
                        error_code=raw.get("error_code", "UNKNOWN"),
                        binance_code=raw.get("binance_code"),
                        error_msg=raw.get("error_msg", ""),
                    ))

                elif event == "BALANCE_CHECK":
                    parsed.balance_checks.append(BalanceCheckRecord(
                        cycle_id=raw["cycle_id"],
                        ts=raw["ts"],
                        usdt_free=raw["usdt_free"],
                        btc_total=raw["btc_total"],
                        max_usdt_allowed=raw["max_usdt_allowed"],
                    ))

            elif track == "SYSTEM":
                parsed.system_events.append(SystemRecord(
                    event=raw.get("event", "UNKNOWN"),
                    ts=raw["ts"],
                    detail=raw.get("detail", ""),
                ))
            else:
                parsed.skipped += 1

        except (KeyError, TypeError):
            parsed.skipped += 1

    return parsed


# ===================================================================
# Section 3: Trade row construction (the core join)
# ===================================================================

def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO-8601 timestamp to aware datetime."""
    return datetime.fromisoformat(ts_str)


def build_audit_rows(parsed: ParsedEvents) -> list[TradeAuditRow]:
    """Join ORDER_FILLED -> ORDER_PLACED (order_id) -> THEORETICAL (cycle_id).

    Every ORDER_PLACED becomes one row.  If the order was filled, we compute
    slippage, latency, and PnL metrics.
    """
    placed_sorted = sorted(
        parsed.orders_placed.values(),
        key=lambda r: r.ts,
    )

    rows: list[TradeAuditRow] = []
    for idx, placed in enumerate(placed_sorted, 1):
        oid = placed.order_id
        cid = placed.cycle_id

        # Lookup THEORETICAL for signal price
        theo = parsed.theoretical.get(cid)
        signal_price = theo.current_price_usd if theo else placed.price
        expected_qty = abs(theo.delta_qty_btc) if theo else placed.qty
        expected_notional = expected_qty * signal_price

        filled = parsed.orders_filled.get(oid)
        if filled:
            fill_price = filled.avg_price
            qty = filled.filled_qty
            actual_notional = qty * fill_price

            # Slippage: direction-aware so positive always means cost
            raw_slip = ((fill_price - signal_price) / signal_price) * 10_000
            if placed.side == "BUY":
                alpha_leak = raw_slip       # paid more = cost
            else:
                alpha_leak = -raw_slip      # received less = cost

            # Fill latency
            dt = (_parse_ts(filled.ts) - _parse_ts(placed.ts)).total_seconds()
            is_stuck = dt > 60.0

            rows.append(TradeAuditRow(
                trade_num=idx,
                order_id=oid,
                cycle_id=cid,
                side=placed.side,
                is_retry=placed.is_retry,
                signal_price=signal_price,
                placed_price=placed.price,
                fill_price=fill_price,
                qty=qty,
                expected_notional=expected_notional,
                actual_notional=actual_notional,
                slippage_bps=raw_slip,
                alpha_leak_bps=alpha_leak,
                fill_time_s=dt,
                is_stuck=is_stuck,
                status="FILLED",
            ))
        else:
            # Cancelled or still pending
            reason = parsed.orders_cancelled.get(oid)
            status = "CANCELLED" if reason else "PENDING"
            rows.append(TradeAuditRow(
                trade_num=idx,
                order_id=oid,
                cycle_id=cid,
                side=placed.side,
                is_retry=placed.is_retry,
                signal_price=signal_price,
                placed_price=placed.price,
                fill_price=None,
                qty=None,
                expected_notional=expected_notional,
                actual_notional=None,
                slippage_bps=None,
                alpha_leak_bps=None,
                fill_time_s=None,
                is_stuck=False,
                status=status,
            ))

    return rows


# ===================================================================
# Section 4: Balance reconciliation
# ===================================================================

def compute_balance_deltas(
    checks: list[BalanceCheckRecord],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (start_usdt, end_usdt, start_btc, end_btc) from balance checks."""
    if len(checks) < 2:
        return None, None, None, None
    by_ts = sorted(checks, key=lambda c: c.ts)
    first, last = by_ts[0], by_ts[-1]
    return first.usdt_free, last.usdt_free, first.btc_total, last.btc_total


# ===================================================================
# Section 5: Terminal table formatting
# ===================================================================

_SEP_CHAR = "-"
_COL_SEP = " | "
_HEAD_SEP = "-+-"

# Column definitions: (header, width, align)
_COLUMNS = [
    ("Trade",        8, "<"),
    ("Side",         4, "<"),
    ("Signal ($)",  12, ">"),
    ("Fill ($)",    12, ">"),
    ("Slip (bps)", 10, ">"),
    ("Fill Time",   9, ">"),
    ("Flag",         6, "<"),
]


def _fmt_price(v: Optional[float]) -> str:
    return f"{v:,.2f}" if v is not None else "---"


def _fmt_bps(v: Optional[float]) -> str:
    return f"{v:+.2f}" if v is not None else "---"


def _fmt_time(v: Optional[float]) -> str:
    return f"{v:.1f}s" if v is not None else "---"


def format_trade_table(rows: list[TradeAuditRow]) -> str:
    """Render the main trade audit table."""
    if not rows:
        return "  (no trades to display)\n"

    header = _COL_SEP.join(f"{h:{a}{w}}" for h, w, a in _COLUMNS)
    sep = _HEAD_SEP.join(_SEP_CHAR * w for _, w, _ in _COLUMNS)

    lines = [header, sep]
    for r in rows:
        tag = f"T-{r.trade_num:03d}"
        if r.is_retry:
            tag += "[R]"

        flag = ""
        if r.is_stuck:
            flag = "STUCK"
        elif r.status == "CANCELLED":
            flag = "CANCEL"
        elif r.status == "PENDING":
            flag = "PEND"

        vals = [
            tag,
            r.side,
            _fmt_price(r.signal_price),
            _fmt_price(r.fill_price),
            _fmt_bps(r.slippage_bps),
            _fmt_time(r.fill_time_s),
            flag,
        ]
        line = _COL_SEP.join(
            f"{v:{a}{w}}" for v, (_, w, a) in zip(vals, _COLUMNS)
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def format_pnl_table(rows: list[TradeAuditRow]) -> str:
    """Render the PnL reconciliation table."""
    filled = [r for r in rows if r.status == "FILLED"]
    if not filled:
        return "  (no filled trades for PnL reconciliation)\n"

    cols = [
        ("Trade",        8, "<"),
        ("Expected ($)",14, ">"),
        ("Actual ($)",  14, ">"),
        ("Diff ($)",    12, ">"),
        ("Slip (bps)", 10, ">"),
    ]

    header = _COL_SEP.join(f"{h:{a}{w}}" for h, w, a in cols)
    sep = _HEAD_SEP.join(_SEP_CHAR * w for _, w, _ in cols)
    lines = [header, sep]

    for r in filled:
        tag = f"T-{r.trade_num:03d}"
        if r.is_retry:
            tag += "[R]"
        diff = (r.actual_notional or 0) - r.expected_notional
        vals = [
            tag,
            f"{r.expected_notional:,.2f}",
            _fmt_price(r.actual_notional),
            f"{diff:+,.2f}",
            _fmt_bps(r.slippage_bps),
        ]
        line = _COL_SEP.join(
            f"{v:{a}{w}}" for v, (_, w, a) in zip(vals, cols)
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


# ===================================================================
# Section 6: Summary statistics
# ===================================================================

def format_summary(
    rows: list[TradeAuditRow],
    parsed: ParsedEvents,
) -> str:
    """Build the full summary report as a string."""
    filled = [r for r in rows if r.status == "FILLED"]
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 72)
    lines.append("  SLIPPAGE SUMMARY")
    lines.append("=" * 72)

    if filled:
        slippages = [r.alpha_leak_bps for r in filled if r.alpha_leak_bps is not None]
        abs_slippages = [abs(s) for s in slippages]
        if slippages:
            lines.append(f"  Mean slippage:       {mean(abs_slippages):>8.2f} bps")
            lines.append(f"  Median slippage:     {median(abs_slippages):>8.2f} bps")
            lines.append(f"  Max slippage:        {max(abs_slippages):>8.2f} bps")

            # Weighted alpha leak (notional-weighted)
            weights = []
            weighted_slips = []
            for r in filled:
                if r.alpha_leak_bps is not None and r.actual_notional is not None:
                    w = abs(r.actual_notional)
                    weights.append(w)
                    weighted_slips.append(r.alpha_leak_bps * w)
            total_w = sum(weights)
            if total_w > 0:
                weighted_leak = sum(weighted_slips) / total_w
                leak_usd = sum(weighted_slips) / 10_000
                lines.append(f"  Weighted alpha leak: {weighted_leak:>+8.2f} bps")
                lines.append(f"  Total leak (USD):    ${leak_usd:>+10,.2f}")
            lines.append("")
    else:
        lines.append("  No filled orders — slippage analysis skipped.")
        lines.append("")

    # --- Latency ---
    lines.append("=" * 72)
    lines.append("  LATENCY SUMMARY")
    lines.append("=" * 72)

    if filled:
        times = [r.fill_time_s for r in filled if r.fill_time_s is not None]
        if times:
            lines.append(f"  Mean fill time:      {mean(times):>8.1f}s")
            lines.append(f"  Median fill time:    {median(times):>8.1f}s")
            lines.append(f"  Max fill time:       {max(times):>8.1f}s")
            stuck = sum(1 for r in filled if r.is_stuck)
            lines.append(
                f"  Stuck orders (>60s): {stuck:>5d} / {len(filled)} "
                f"({stuck / len(filled) * 100:.1f}%)"
            )
        lines.append("")
    else:
        lines.append("  No filled orders — latency analysis skipped.")
        lines.append("")

    # --- Balance reconciliation ---
    lines.append("=" * 72)
    lines.append("  BALANCE RECONCILIATION")
    lines.append("=" * 72)

    s_usdt, e_usdt, s_btc, e_btc = compute_balance_deltas(parsed.balance_checks)
    if s_usdt is not None:
        lines.append(f"  Starting USDT:  ${s_usdt:>12,.2f}    Ending USDT:  ${e_usdt:>12,.2f}")
        lines.append(f"  Starting BTC:    {s_btc:>12.8f}    Ending BTC:    {e_btc:>12.8f}")
        lines.append(f"  Net USDT change: ${(e_usdt - s_usdt):>+12,.2f}")
        lines.append(f"  Net BTC change:   {(e_btc - s_btc):>+12.8f}")
    else:
        lines.append("  Insufficient BALANCE_CHECK events for reconciliation.")
    lines.append("")

    # --- 10bps friction verdict ---
    lines.append("=" * 72)
    lines.append("  10bps FRICTION VERDICT")
    lines.append("=" * 72)
    lines.append(f"  Backtest assumption: 10 bps slippage + 10 bps fee = 20 bps round-trip")

    if filled:
        abs_slippages = [abs(r.alpha_leak_bps) for r in filled if r.alpha_leak_bps is not None]
        if abs_slippages:
            observed = mean(abs_slippages)
            lines.append(f"  Observed mean slip:  {observed:.2f} bps")
            if observed <= 10.0:
                lines.append(
                    "  Verdict:             REALISTIC — observed slippage within 10bps budget"
                )
            elif observed <= 15.0:
                lines.append(
                    "  Verdict:             OPTIMISTIC — consider raising slippage_bps to 15"
                )
            else:
                lines.append(
                    f"  Verdict:             UNREALISTIC — observed {observed:.1f}bps "
                    f"exceeds 10bps assumption. Raise Convexity friction parameters."
                )
        else:
            lines.append("  No slippage data to compare.")
    else:
        lines.append("  No filled trades — cannot evaluate friction assumption.")
        lines.append("")
        _append_diagnostic(lines, parsed)

    lines.append("=" * 72)
    return "\n".join(lines)


def _append_diagnostic(lines: list[str], parsed: ParsedEvents) -> None:
    """Add diagnostic info when no trades exist."""
    # Count system events by type
    sys_counts: dict[str, int] = {}
    for se in parsed.system_events:
        sys_counts[se.event] = sys_counts.get(se.event, 0) + 1

    err_counts: dict[str, int] = {}
    for oe in parsed.order_errors:
        err_counts[oe.error_code] = err_counts.get(oe.error_code, 0) + 1

    if sys_counts:
        lines.append("  System events observed:")
        for evt, cnt in sorted(sys_counts.items()):
            lines.append(f"    {evt}: {cnt}")

    if err_counts:
        lines.append("  Order errors observed:")
        for code, cnt in sorted(err_counts.items()):
            lines.append(f"    {code}: {cnt}")

    if not sys_counts and not err_counts:
        lines.append("  No system or error events found either.")


# ===================================================================
# Section 7: Entry point
# ===================================================================

def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "./logs"

    log_files = discover_log_files(target)
    if not log_files:
        print(f"audit.py: No bridge log files found in '{target}'")
        print(f"  Expected: bridge_YYYYMMDD.jsonl files")
        print(f"  Usage: python audit.py              (reads ./logs/)")
        print(f"         python audit.py logs/         (explicit directory)")
        print(f"         python audit.py path/to/file.jsonl")
        sys.exit(0)

    # ---- Load ----
    print(f"Audit: loading {len(log_files)} log file(s)...\n")
    all_records: list[dict] = []
    for lf in log_files:
        records = load_jsonl(lf)
        print(f"  {lf.name}: {len(records)} records")
        all_records.extend(records)
    print()

    if not all_records:
        print("No parseable records found. Exiting.")
        sys.exit(0)

    # ---- Classify ----
    parsed = classify_events(all_records)
    if parsed.skipped:
        print(f"  ({parsed.skipped} unrecognised record(s) skipped)\n")

    # ---- Build trade rows ----
    trades = build_audit_rows(parsed)

    # ---- Print report ----
    unique_cycles = set()
    for t in parsed.theoretical:
        unique_cycles.add(t)
    for op in parsed.orders_placed.values():
        unique_cycles.add(op.cycle_id)
    for oe in parsed.order_errors:
        if oe.cycle_id:
            unique_cycles.add(oe.cycle_id)

    filled_count = sum(1 for r in trades if r.status == "FILLED")

    print("=" * 72)
    print("               EXECUTION QUALITY AUDIT REPORT")
    print("=" * 72)
    print(f"  Log files:        {len(log_files)}")
    print(f"  Cycles scanned:   {len(unique_cycles)}")
    print(f"  Theoretical recs: {len(parsed.theoretical)}")
    print(f"  Orders placed:    {len(parsed.orders_placed)}")
    print(f"  Orders filled:    {len(parsed.orders_filled)}")
    print(f"  Orders cancelled: {len(parsed.orders_cancelled)}")
    print(f"  Order errors:     {len(parsed.order_errors)}")
    print(f"  Fills:            {filled_count}")
    print()

    # Trade detail table
    print("=" * 72)
    print("  TRADE DETAIL")
    print("=" * 72)
    print(format_trade_table(trades))

    # PnL reconciliation table
    print("=" * 72)
    print("  PnL RECONCILIATION")
    print("=" * 72)
    print(format_pnl_table(trades))

    # Legend
    has_stuck = any(r.is_stuck for r in trades)
    has_retry = any(r.is_retry for r in trades)
    if has_stuck or has_retry:
        if has_stuck:
            print("  STUCK = fill time > 60s")
        if has_retry:
            print("  [R]   = aggressive retry attempt")
        print()

    # Summary sections
    print(format_summary(trades, parsed))


if __name__ == "__main__":
    main()
