"""
dashboard.py — Command & Control Center for the Agentic Trading Bot.

Run:  python -m streamlit run dashboard.py
      (from the repo root so relative paths resolve correctly)

Sections:
  1. Live Equity vs. Theoretical backtest curve + Delta metric
  2. Mutation Monitor — side-by-side code diff + LLM diagnosis
  3. Alpha Leak Tracker — slippage bps + fill-time latency
  4. Regime Heatmap — 7-day Bull/Bear/Circuit-Break timeline

Auto-refresh via <meta http-equiv="refresh"> — no extra pip packages needed.
"""

from __future__ import annotations

import difflib
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGS_DIR = Path("logs")
STRATEGY_FILE = Path("strategies/volatility_squeeze.py")
STRATEGY_BACKUP = Path("strategies/volatility_squeeze.py.bak")
MUTATION_META_FILE = Path("strategies/mutation_meta.json")
BRIDGE_OVERRIDE_FILE = Path("bridge_override.json")

SLIPPAGE_THRESHOLD_BPS: float = 10.0  # friction floor; bars above this → orange
LATENCY_BASELINE_S: float = 30.1      # current fill-time baseline for reference line
LATENCY_STUCK_S: float = 60.0         # fill_time_s > this → considered stuck
BACKTEST_HOURS: int = 8_760           # 1 year of hourly bars (seed=42)
REGIME_LOOKBACK_DAYS: int = 7

UTC = timezone.utc

log = logging.getLogger("dashboard")

# ---------------------------------------------------------------------------
# Data loaders  (all cached; safe defaults on missing/empty files)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_bridge_records(days: int = REGIME_LOOKBACK_DAYS) -> list[dict]:
    """Read bridge_YYYYMMDD.jsonl files for the last *days* UTC days."""
    records: list[dict] = []
    today = datetime.now(UTC).date()
    for offset in range(days):
        day = today - timedelta(days=offset)
        path = LOGS_DIR / f"bridge_{day.strftime('%Y%m%d')}.jsonl"
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception as exc:
            log.warning("Failed to parse %s: %s", path, exc)
    return records


@st.cache_data(ttl=60)
def load_audit_rows() -> list[Any]:
    """Return TradeAuditRow list via audit.py; empty list on any failure."""
    try:
        from audit import build_audit_rows, classify_events, load_jsonl
    except ImportError:
        return []
    try:
        raw = load_jsonl(LOGS_DIR)
        parsed = classify_events(raw)
        return build_audit_rows(parsed)
    except Exception as exc:
        log.warning("audit rows failed: %s", exc)
        return []


@st.cache_data(ttl=3600)
def load_backtest_curve(init_cash: float = 10_000.0) -> tuple[list[str], list[float]]:
    """Compute the backtest equity curve on generate_mock_ohlcv(seed=42).

    Returns (iso_timestamps, equity_values).  Cached for 1 hour — the
    backtest is deterministic, so there is no need to rerun more often
    unless the strategy file changes (the TTL handles drift naturally).
    """
    try:
        import polars as pl
        from backtester.data import generate_mock_ohlcv
        from strategies.volatility_squeeze import VolatilitySqueezeBreakout
    except Exception as exc:
        log.warning("backtest import failed: %s", exc)
        return [], []

    try:
        df = generate_mock_ohlcv("BTC/USDT", hours=BACKTEST_HOURS, seed=42)
        strategy = VolatilitySqueezeBreakout()
        signals = strategy.generate_signals(df)

        # Simplified equity curve (same logic as engine.py, without fees)
        position = signals.cast(pl.Float64).shift(1).fill_null(0.0)
        ret = position * df["close"].pct_change().fill_null(0.0)
        equity = (1.0 + ret).cum_prod() * init_cash

        timestamps = df["timestamp"].cast(str).to_list()
        return timestamps, equity.to_list()
    except Exception as exc:
        log.warning("backtest curve failed: %s", exc)
        return [], []


@st.cache_data(ttl=30)
def load_strategy_diff() -> tuple[str, str, str]:
    """Return (current_source, bak_source, unified_diff_text)."""
    current = STRATEGY_FILE.read_text(encoding="utf-8") if STRATEGY_FILE.exists() else ""
    bak = STRATEGY_BACKUP.read_text(encoding="utf-8") if STRATEGY_BACKUP.exists() else ""
    if not current or not bak:
        return current, bak, ""
    diff_lines = list(
        difflib.unified_diff(
            bak.splitlines(),
            current.splitlines(),
            fromfile="volatility_squeeze.py.bak (previous)",
            tofile="volatility_squeeze.py (live)",
            lineterm="",
        )
    )
    return current, bak, "\n".join(diff_lines)


@st.cache_data(ttl=30)
def load_mutation_meta() -> dict:
    """Return parsed mutation_meta.json; empty dict if not present."""
    if not MUTATION_META_FILE.exists():
        return {}
    try:
        return json.loads(MUTATION_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_bridge_override() -> dict:
    """Return parsed bridge_override.json; empty dict if not present."""
    if not BRIDGE_OVERRIDE_FILE.exists():
        return {}
    try:
        return json.loads(BRIDGE_OVERRIDE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Derived views built from raw bridge records
# ---------------------------------------------------------------------------


def _balance_checks(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("track") == "ATTEMPTED" and r.get("event") == "BALANCE_CHECK"]


def _theoretical_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("track") == "THEORETICAL"]


def _live_equity_series(records: list[dict]) -> tuple[list[str], list[float]]:
    """Join BALANCE_CHECK with nearest THEORETICAL to get portfolio $ over time."""
    checks = _balance_checks(records)
    theoreticals = {r["cycle_id"]: r for r in _theoretical_records(records)}

    timestamps, values = [], []
    for chk in sorted(checks, key=lambda r: r["ts"]):
        cycle_id = chk.get("cycle_id")
        theo = theoreticals.get(cycle_id, {})
        price = theo.get("current_price_usd", 0.0)
        usdt = chk.get("usdt_free", 0.0)
        btc = chk.get("btc_total", 0.0)
        portfolio_val = usdt + btc * price
        timestamps.append(chk["ts"])
        values.append(portfolio_val)
    return timestamps, values


def _regime_series(records: list[dict], days: int = REGIME_LOOKBACK_DAYS) -> list[dict]:
    """Return list of {ts, bear_regime, cb_active} for last *days* days."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    out = []
    for r in _theoretical_records(records):
        try:
            ts = datetime.fromisoformat(r["ts"])
            if ts >= cutoff:
                out.append({"ts": ts, "bear_regime": r.get("bear_regime", False),
                            "cb_active": r.get("cb_active", False)})
        except Exception:
            pass
    return sorted(out, key=lambda x: x["ts"])


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_sidebar(override: dict) -> int:
    """Render sidebar; return chosen auto-refresh interval in seconds."""
    st.sidebar.markdown("## Settings")
    refresh_s = st.sidebar.slider("Auto-refresh (s)", 30, 300, 60, step=30)

    if st.sidebar.button("Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last update:** {datetime.now(UTC).strftime('%H:%M:%S UTC')}")

    if override:
        meta = override.get("_meta", {})
        ov = override.get("overrides", {})
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Bridge Override Active")
        if meta.get("trigger"):
            st.sidebar.caption(meta["trigger"])
        if ov:
            st.sidebar.json(ov)

    return refresh_s


def render_kpi_row(
    live_ts: list[str],
    live_vals: list[float],
    bt_ts: list[str],
    bt_vals: list[float],
    audit_rows: list[Any],
    records: list[dict],
) -> None:
    col1, col2, col3, col4 = st.columns(4)

    # KPI 1 — Live portfolio value
    live_value = live_vals[-1] if live_vals else None
    with col1:
        if live_value is not None:
            st.metric("Live Portfolio", f"${live_value:,.2f}")
        else:
            st.metric("Live Portfolio", "—")

    # KPI 2 — Delta vs theoretical
    with col2:
        if live_ts and bt_ts and live_vals and bt_vals:
            try:
                # Align: find bt equity at the same start as live
                first_live_ts = datetime.fromisoformat(live_ts[0])
                # Scale bt_vals so its t=0 matches live start value
                init_live = live_vals[0]
                init_bt = bt_vals[0]
                scale = init_live / init_bt if init_bt != 0 else 1.0
                bt_last_scaled = bt_vals[-1] * scale
                delta = live_value - bt_last_scaled if live_value is not None else 0.0
                pct = (delta / bt_last_scaled * 100) if bt_last_scaled else 0.0
                color = "normal" if delta >= 0 else "inverse"
                st.metric("Delta vs Theoretical", f"${delta:+,.2f}", f"{pct:+.1f}%", delta_color=color)
            except Exception:
                st.metric("Delta vs Theoretical", "—")
        else:
            st.metric("Delta vs Theoretical", "—")

    # KPI 3 — Mean slippage
    with col3:
        filled = [r for r in audit_rows if getattr(r, "slippage_bps", None) is not None
                  and r.status == "FILLED"]
        if filled:
            mean_slip = sum(r.slippage_bps for r in filled) / len(filled)
            flag = "inverse" if mean_slip > SLIPPAGE_THRESHOLD_BPS else "normal"
            st.metric("Mean Slippage", f"{mean_slip:.1f} bps",
                      f"threshold: {SLIPPAGE_THRESHOLD_BPS:.0f} bps", delta_color=flag)
        else:
            st.metric("Mean Slippage", "—")

    # KPI 4 — Active regime
    with col4:
        theo_list = _theoretical_records(records)
        if theo_list:
            last = sorted(theo_list, key=lambda r: r["ts"])[-1]
            if last.get("cb_active"):
                regime_label, regime_color = "Circuit Break", "#e74c3c"
            elif last.get("bear_regime"):
                regime_label, regime_color = "Bear", "#e67e22"
            else:
                regime_label, regime_color = "Bull", "#27ae60"
            st.markdown(
                f"**Active Regime**  \n"
                f'<span style="font-size:1.6rem;font-weight:700;color:{regime_color}">'
                f"{regime_label}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.metric("Active Regime", "—")


def render_equity_tab(
    live_ts: list[str],
    live_vals: list[float],
    bt_ts: list[str],
    bt_vals: list[float],
) -> None:
    if not live_vals and not bt_vals:
        st.info("No equity data yet — bridge logs are empty or missing.")
        return

    fig = go.Figure()

    # Live equity trace
    if live_ts and live_vals:
        fig.add_trace(go.Scatter(
            x=live_ts, y=live_vals,
            mode="lines", name="Live Portfolio ($)",
            line=dict(color="#3498db", width=2),
        ))

    # Theoretical backtest trace (scaled to live starting value)
    if bt_ts and bt_vals and live_vals:
        init_live = live_vals[0]
        init_bt = bt_vals[0] if bt_vals[0] != 0 else 1.0
        scale = init_live / init_bt
        bt_scaled = [v * scale for v in bt_vals]
        # Trim theoretical to same time span as live for visual alignment
        n = min(len(bt_ts), max(len(live_ts), 1))
        fig.add_trace(go.Scatter(
            x=bt_ts[:n], y=bt_scaled[:n],
            mode="lines", name="Theoretical Backtest ($)",
            line=dict(color="#95a5a6", width=1.5, dash="dot"),
        ))
    elif bt_ts and bt_vals:
        fig.add_trace(go.Scatter(
            x=bt_ts, y=bt_vals,
            mode="lines", name="Theoretical Backtest ($)",
            line=dict(color="#95a5a6", width=1.5, dash="dot"),
        ))

    fig.update_layout(
        title="Live Portfolio vs. Theoretical Backtest Equity",
        xaxis_title="Time (UTC)",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta annotation
    if live_vals and bt_vals:
        try:
            scale = live_vals[0] / (bt_vals[0] or 1.0)
            bt_final = bt_vals[min(len(live_vals), len(bt_vals)) - 1] * scale
            delta = live_vals[-1] - bt_final
            pct = delta / bt_final * 100 if bt_final else 0.0
            color = "green" if delta >= 0 else "red"
            direction = "beating" if delta >= 0 else "trailing"
            st.markdown(
                f'<p style="font-size:1.05rem">Live is <strong style="color:{color}">'
                f"${abs(delta):,.2f} ({abs(pct):.1f}%)</strong> "
                f"{direction} theoretical</p>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    elif not bt_ts:
        st.warning("Strategy could not be loaded for backtest — check logs.")


def render_mutation_tab(meta: dict, diff_text: str, bak_source: str) -> None:
    left, right = st.columns([4, 6])

    with left:
        if not meta:
            st.info("No mutations recorded yet.  Run meta_loop.py to generate the first mutation.")
        else:
            ts_raw = meta.get("timestamp", "")
            try:
                ts_fmt = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                ts_fmt = ts_raw

            st.markdown(f"**Last Mutation:** `{meta.get('mutation_name', '—')}`")
            st.markdown(f"**Accepted:** {ts_fmt}")

            old_sharpe = meta.get("old_sharpe", 0.0)
            new_sharpe = meta.get("new_sharpe", 0.0)
            old_cagr = meta.get("old_cagr", 0.0)
            new_cagr = meta.get("new_cagr", 0.0)
            imp = meta.get("sharpe_improvement_pct", 0.0)

            st.markdown(
                f"**Sharpe:** {old_sharpe:.3f} → {new_sharpe:.3f} "
                f"(*+{imp:.1f}%*)"
            )
            st.markdown(
                f"**CAGR:** {old_cagr * 100:.1f}% → {new_cagr * 100:.1f}%"
            )

            if meta.get("is_llm"):
                st.markdown("---")
                st.markdown("**LLM Structural Change**")
                st.markdown(f"> {meta.get('rationale', '—')}")
                if meta.get("failure_narrative"):
                    st.markdown("**Diagnosis Context:**")
                    st.caption(meta["failure_narrative"])
            else:
                changes = meta.get("param_changes", {})
                if changes:
                    st.markdown("---")
                    st.markdown("**Parameter Changes:**")
                    for k, v in changes.items():
                        st.markdown(f"- `{k}` → `{v}`")
                if meta.get("failure_narrative"):
                    st.markdown("**Failure Narrative:**")
                    st.caption(meta["failure_narrative"])

    with right:
        st.markdown("**Strategy Code Diff** (previous → live)")
        if not bak_source:
            st.info("No backup file found — strategy has never been mutated by meta_loop.")
        elif not diff_text:
            st.success("Live strategy is identical to backup — no changes detected.")
        else:
            st.code(diff_text, language="diff")


def render_alpha_tab(audit_rows: list[Any]) -> None:
    filled = [r for r in audit_rows
              if getattr(r, "status", None) == "FILLED"
              and getattr(r, "slippage_bps", None) is not None]

    if not filled:
        st.info("No filled trades in the log window. Slippage and latency charts will appear once trades are filled.")
        return

    trade_nums = [r.trade_num for r in filled]
    slippages = [r.slippage_bps for r in filled]
    latencies = [r.fill_time_s for r in filled]

    # ── Slippage bar chart ──────────────────────────────────────────────────
    bar_colors = [
        "#e67e22" if s > SLIPPAGE_THRESHOLD_BPS else "#27ae60"
        for s in slippages
    ]
    # Highlight > 2× threshold in red
    bar_colors = [
        "#e74c3c" if s > SLIPPAGE_THRESHOLD_BPS * 2 else c
        for s, c in zip(slippages, bar_colors)
    ]

    fig_slip = go.Figure()
    fig_slip.add_trace(go.Bar(
        x=trade_nums, y=slippages,
        marker_color=bar_colors,
        name="Slippage (bps)",
        hovertemplate="Trade #%{x}<br>Slippage: %{y:.2f} bps<extra></extra>",
    ))
    fig_slip.add_hline(
        y=SLIPPAGE_THRESHOLD_BPS,
        line_dash="dash", line_color="red", line_width=1.5,
        annotation_text=f"{SLIPPAGE_THRESHOLD_BPS:.0f} bps friction floor",
        annotation_position="top right",
    )
    fig_slip.update_layout(
        title="Slippage per Trade",
        xaxis_title="Trade #",
        yaxis_title="Slippage (bps)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_slip, use_container_width=True)

    # ── Latency line chart ──────────────────────────────────────────────────
    valid_lat = [(tn, lt) for tn, lt in zip(trade_nums, latencies) if lt is not None]
    if valid_lat:
        lat_x, lat_y = zip(*valid_lat)
        lat_colors = ["#e74c3c" if lt > LATENCY_STUCK_S else "#3498db" for lt in lat_y]

        fig_lat = go.Figure()
        fig_lat.add_trace(go.Scatter(
            x=list(lat_x), y=list(lat_y),
            mode="lines+markers",
            marker=dict(color=lat_colors, size=7),
            line=dict(color="#3498db", width=1.5),
            name="Fill Time (s)",
            hovertemplate="Trade #%{x}<br>Fill Time: %{y:.1f}s<extra></extra>",
        ))
        fig_lat.add_hline(
            y=LATENCY_BASELINE_S,
            line_dash="dash", line_color="#95a5a6", line_width=1.5,
            annotation_text=f"Baseline {LATENCY_BASELINE_S}s",
            annotation_position="top right",
        )
        fig_lat.add_hline(
            y=LATENCY_STUCK_S,
            line_dash="dot", line_color="#e74c3c", line_width=1,
            annotation_text="Stuck threshold (60s)",
            annotation_position="bottom right",
        )
        fig_lat.update_layout(
            title="Fill-Time Latency per Trade",
            xaxis_title="Trade #",
            yaxis_title="Fill Time (s)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Summary stats ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Trades Analysed", len(filled))
    c2.metric("Above Threshold", sum(1 for s in slippages if s > SLIPPAGE_THRESHOLD_BPS))
    if valid_lat:
        c3.metric("Avg Latency", f"{sum(lat_y) / len(lat_y):.1f}s")


def render_regime_tab(records: list[dict]) -> None:
    regime_data = _regime_series(records, days=REGIME_LOOKBACK_DAYS)

    if not regime_data:
        st.info(
            f"No THEORETICAL records in the last {REGIME_LOOKBACK_DAYS} days. "
            "The regime heatmap will populate as the bridge runs."
        )
        return

    # Build per-hour regime classification
    COLOR_MAP = {
        "Circuit Break": "#e74c3c",
        "Bear":          "#e67e22",
        "Bull":          "#27ae60",
    }

    xs, ys, colors, labels = [], [], [], []
    for row in regime_data:
        ts: datetime = row["ts"]
        day_label = ts.strftime("%Y-%m-%d")
        hour = ts.strftime("%H:00")
        if row["cb_active"]:
            regime = "Circuit Break"
        elif row["bear_regime"]:
            regime = "Bear"
        else:
            regime = "Bull"
        xs.append(hour)
        ys.append(day_label)
        colors.append(COLOR_MAP[regime])
        labels.append(regime)

    fig = go.Figure()
    for regime, color in COLOR_MAP.items():
        mask = [i for i, l in enumerate(labels) if l == regime]
        if not mask:
            continue
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in mask],
            y=[ys[i] for i in mask],
            mode="markers",
            marker=dict(symbol="square", size=10, color=color),
            name=regime,
            hovertemplate=f"<b>{regime}</b><br>%{{y}} %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Regime Heatmap — Last {REGIME_LOOKBACK_DAYS} Days",
        xaxis=dict(title="Hour (UTC)", tickangle=-45, categoryorder="category ascending"),
        yaxis=dict(title="Date", autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Regime breakdown summary ────────────────────────────────────────────
    total = len(labels)
    from collections import Counter
    counts = Counter(labels)
    c1, c2, c3 = st.columns(3)
    c1.metric("Bull", f"{counts.get('Bull', 0) / total * 100:.0f}%")
    c2.metric("Bear", f"{counts.get('Bear', 0) / total * 100:.0f}%")
    c3.metric("Circuit Break", f"{counts.get('Circuit Break', 0) / total * 100:.0f}%")

    # Longest CB streak
    cb_streak, max_cb_streak, current = 0, 0, False
    for label in labels:
        if label == "Circuit Break":
            cb_streak += 1
            max_cb_streak = max(max_cb_streak, cb_streak)
        else:
            cb_streak = 0
    if max_cb_streak:
        st.caption(f"Longest consecutive Circuit Break streak: **{max_cb_streak}h**")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="C&C Dashboard — Agentic Trading Bot",
        layout="wide",
        page_icon="📡",
        menu_items={"Get help": None, "Report a bug": None, "About": None},
    )

    # ── Load data ────────────────────────────────────────────────────────────
    records = load_bridge_records(days=REGIME_LOOKBACK_DAYS)
    audit_rows = load_audit_rows()
    meta = load_mutation_meta()
    override = load_bridge_override()
    _current_source, bak_source, diff_text = load_strategy_diff()

    live_ts, live_vals = _live_equity_series(records)
    init_cash = live_vals[0] if live_vals else 10_000.0
    bt_ts, bt_vals = load_backtest_curve(init_cash=init_cash)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    refresh_s = render_sidebar(override)

    # Browser-native auto-refresh — no extra packages required
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_s}">',
        unsafe_allow_html=True,
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("# 📡 Command & Control Center")
    st.caption("Agentic Trading Bot · Binance Testnet · BTC/USDT 1h")

    # ── KPI row ──────────────────────────────────────────────────────────────
    render_kpi_row(live_ts, live_vals, bt_ts, bt_vals, audit_rows, records)
    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Equity",
        "Mutation Monitor",
        "Alpha Leak",
        "Regime Heatmap",
    ])

    with tab1:
        render_equity_tab(live_ts, live_vals, bt_ts, bt_vals)

    with tab2:
        render_mutation_tab(meta, diff_text, bak_source)

    with tab3:
        render_alpha_tab(audit_rows)

    with tab4:
        render_regime_tab(records)


if __name__ == "__main__":
    main()
