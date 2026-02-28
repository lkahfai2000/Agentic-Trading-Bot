# CLAUDE.md — Agentic Trading Bot

## Project Overview

A self-optimizing algorithmic trading system for BTC/USDT on Binance Testnet.
The bot trades autonomously, monitors its own execution quality, proposes
strategy mutations, backtests them, and hot-swaps winning code — all in a
closed feedback loop with no human intervention required.

**Stack**: Python 3.11+ | Polars (vectorized) | CCXT (exchange) | Telegram (alerts)
**Exchange**: Binance Testnet (paper trading)
**Timeframe**: 1-hour candles
**Strategy**: Volatility Squeeze Breakout (convex alpha generator)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_lab.sh (Orchestrator)                    │
│  while true:                                                        │
│    1. Start bridge.py &                                             │
│    2. Run meta_loop.py --enable-llm                                 │
│    3. Exit code 2 → kill bridge, restart loop                       │
│       Exit code 0 → sleep 1h, loop                                  │
│       Exit code 1 → Telegram alert, halt                            │
└────────────┬──────────────────────────────┬─────────────────────────┘
             │                              │
             ▼                              ▼
┌────────────────────────┐   ┌──────────────────────────────────────┐
│   bridge.py (Daemon)   │   │   meta_loop.py (Batch Optimizer)     │
│                        │   │                                      │
│  Hourly trading cycle: │   │  6-phase pipeline:                   │
│  1. Fetch 1100 OHLCV   │   │  1. Ingest audit metrics             │
│  2. generate_signals()  │   │  2. Update bridge_override.json      │
│  3. Query wallet        │   │  3. Build failure narrative           │
│  4. Compute trade delta │   │  4. Propose 3 deterministic mutations │
│  5. Place limit order   │   │  4b. LLM structural mutation (Claude)│
│  6. Log to JSONL        │   │  5. Proving Ground backtest           │
│                        │   │  6. Hot-swap if winner > +10% Sharpe  │
│  Health monitor:       │   │                                      │
│  60s checks, 3 fails   │   │  Exit: 0=no swap, 1=error, 2=swapped │
│  → System Pause        │   │                                      │
└───────┬────────────────┘   └──────┬───────────────────────────────┘
        │                           │
        │  writes                   │  reads logs, writes strategy
        ▼                           ▼
┌────────────────┐   ┌──────────────────────────────────────────────┐
│  logs/          │   │  strategies/volatility_squeeze.py            │
│  bridge_YYMMDD │   │  strategies/volatility_squeeze.py.bak        │
│  .jsonl        │   │  strategies/mutation_meta.json                │
│                │   │  bridge_override.json                         │
└────────────────┘   └──────────────────────────────────────────────┘
        │                           │
        └──────────┬────────────────┘
                   ▼
         ┌──────────────────┐       ┌───────────────────────┐
         │   audit.py       │       │   dashboard.py        │
         │   Slippage &     │       │   Streamlit C&C       │
         │   fill latency   │       │   - Equity curve      │
         │   analysis       │       │   - Mutation monitor   │
         └──────────────────┘       │   - Alpha leak tracker │
                                    │   - Regime heatmap     │
         ┌──────────────────┐       └───────────────────────┘
         │   alerts.py      │
         │   Telegram fire- │
         │   and-forget     │
         │   (daemon thread)│
         └──────────────────┘
```

---

## Data Flow

```
bridge.py                    meta_loop.py                   dashboard.py
    │                             │                              │
    │ writes JSONL ──────────────▶│ reads logs (Phase 1)         │
    │                             │                              │
    │ reads on startup ◀──────────│ writes bridge_override.json  │
    │                             │ (Phase 2)                    │
    │                             │                              │
    │                             │ reads strategy file           │
    │                             │ writes new strategy ─────────▶│ reads .py + .bak
    │                             │ writes .bak backup            │ (code diff)
    │                             │ writes mutation_meta.json ───▶│ (mutation info)
    │                             │                              │
    │ writes JSONL ─────────────────────────────────────────────▶│ (equity, regime)
    │                             │                              │
    │ calls alerts.py            │ calls alerts.py               │
    │   trade_signal()            │   mutation_accepted()         │
    │   system_pause()            │                              │
    └─────────────────────────────┴──────────────────────────────┘
```

---

## File Map

### Core System

| File | Purpose |
|------|---------|
| `bridge.py` | Paper-trading daemon. Binance Testnet via CCXT. Hourly cycle: fetch OHLCV → signals → orders. Dual-track JSONL logging. Health monitor → System Pause after 3 failures. |
| `meta_loop.py` | Closed-loop optimizer. 6-phase pipeline: audit → config → failure narrative → mutations → backtest → hot-swap. Includes LLM Consultant (Phase 4b) via Anthropic API. |
| `run_lab.sh` | Shell orchestrator. Infinite loop: bridge + meta_loop. Handles exit codes (0/1/2), PID management, signal trapping, .env sourcing. macOS launchd-ready. |
| `alerts.py` | Telegram alerter. Background daemon thread, fire-and-forget queue. Methods: `trade_signal()`, `order_event()`, `system_pause()`, `mutation_accepted()`, `startup()`, `hourly_pnl()`. |
| `audit.py` | Execution quality audit. Parses JSONL logs, joins THEORETICAL/ATTEMPTED by cycle_id. Reports slippage, fill latency, stuck orders. |
| `dashboard.py` | Streamlit C&C Center. 4 tabs: Equity curve, Mutation monitor, Alpha leak tracker, Regime heatmap. |
| `main.py` | CLI backtest runner. `python main.py [all|sma|squeeze]` for standalone strategy comparison. |

### Strategy Layer

| File | Purpose |
|------|---------|
| `strategies/__init__.py` | Exports `SmaCrossover`, `VolatilitySqueezeBreakout` |
| `strategies/volatility_squeeze.py` | Production strategy. 15-phase vectorized signal pipeline. **Auto-mutated by meta_loop.** |
| `strategies/sma_crossover.py` | Baseline strategy (deprecated, -82% CAGR). Kept for comparison. |

### Backtester (Proving Ground)

| File | Purpose |
|------|---------|
| `backtester/__init__.py` | Exports `GradingReport`, `Strategy`, `load_ohlcv`, `generate_mock_ohlcv`, `run_backtest` |
| `backtester/strategy.py` | Abstract base class. Contract: `generate_signals(df) → pl.Series` in [-1.0, 1.0] |
| `backtester/engine.py` | Vectorized backtest engine. T+1 execution, configurable fees/slippage. Returns `GradingReport`. |
| `backtester/data.py` | `generate_mock_ohlcv()`: GBM synthetic data (seed=42). `load_ohlcv()`: Parquet loader. |
| `backtester/schema.py` | `GradingReport` Pydantic model: CAGR, Sharpe, Sortino, MaxDD, win rate, failure narrative. |
| `backtester/feedback.py` | `generate_failure_narrative()`: Vectorized regime classification → 2-sentence analysis. |

### Diagnostics & Config

| File | Purpose |
|------|---------|
| `test_alerts.py` | Telegram diagnostic. 3 test messages with full HTTP error reporting. |
| `test_telegram.py` | Legacy Telegram test. |
| `.env.example` | Template for environment variables. |
| `requirements.txt` | `polars`, `pydantic`, `pyarrow`, `numpy`, `ccxt` |

### Runtime Artifacts (gitignored)

| File | Written By | Read By |
|------|-----------|---------|
| `logs/bridge_YYYYMMDD.jsonl` | bridge.py | audit.py, meta_loop.py, dashboard.py |
| `logs/wrapper.log` | run_lab.sh | Human operator |
| `bridge_override.json` | meta_loop.py | bridge.py (at startup only) |
| `strategies/volatility_squeeze.py.bak` | meta_loop.py | dashboard.py (diff view) |
| `strategies/mutation_meta.json` | meta_loop.py | dashboard.py (mutation info) |

---

## Strategy: Volatility Squeeze Breakout

### Signal Pipeline (15 Phases)

```
OHLCV Input (1h candles)
    │
    ▼ Core Indicators
Phase 1:  BB mid, BB std, True Range components
Phase 2:  BB upper/lower bands, True Range
Phase 3:  ATR + Bollinger Band Width (BBW)
Phase 3A: Flash Crash Circuit Breaker (ATR spike > 3x baseline → flatten)
Phase 3B: Bear Filter (800-bar EMA, close < EMA → bear regime)
Phase 4:  Squeeze detection (BBW in bottom 10th percentile)
Phase 5:  Squeeze RELEASE (was squeezed → now expanding)
Phase 6:  Momentum candle confirmation (body position threshold)
Phase 7:  ATR-based stop levels (regime-aware)
Phase 8:  Entry conditions (FIRST bar of release only)
    │
    ▼ Convex Alpha Generator (V2)
Phase 9:  ADX computation (Wilder EMA, +DI/-DI → DX → ADX)
Phase 10: Kelly-Lite position sizing (squeeze intensity)
Phase 11: Combined size = kelly × adx_factor × regime_scale
Phase 12: Two-pass signal (forward-fill entries as direction)
Phase 13: Break-even stop logic (profit-based ATR tightening)
Phase 14: Final signal assembly (entries + stops + sizing)
    │
    ▼ Risk Override (V3)
Phase 15: Circuit Breaker override (ATR spike → force 0.0)
    │
    ▼
Output: pl.Series in [-1.0, 1.0]
```

### Key Parameters (mutated by meta_loop)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bb_period` | 20 | Bollinger Band lookback |
| `bb_std` | 2.0 | BB standard deviation multiplier |
| `atr_stop_mult` | 3.5 | ATR stop-loss multiplier |
| `squeeze_pctile` | 0.10 | BBW percentile for squeeze detection |
| `bear_size_factor` | 0.08 | Position scale in bear regime (8%) |
| `bear_stop_factor` | 1.0 | Stop tightening in bear (1.0 = no change) |
| `cb_atr_mult` | 3.0 | Circuit breaker ATR spike threshold |
| `adx_threshold` | 25.0 | ADX level for trend confirmation |
| `adx_weak_factor` | 0.5 | Position scale when ADX declining |
| `be_atr_threshold` | 999.0 | Profit ATR for stop tightening (disabled) |

### Evolution History

```
V1 (Linear):    +11.3% CAGR, -33.2% DD, Sharpe 0.46, 49 trades
V2 (Convex):    +16.5% CAGR, -26.1% DD, Sharpe 0.64
V3 (Regime):    +23.5% CAGR, -15.0% DD, Sharpe 1.14, 42 trades
DD/Return:       2.8 → 1.6 → 0.64  (77% total improvement)
```

---

## Meta Loop Pipeline

```
Phase 1: AUDIT INGESTION
    audit.py → AuditMetrics (slippage bps, fill latency, stuck orders)

Phase 2: BRIDGE CONFIG UPDATE
    If mean_alpha_leak > 10bps → write bridge_override.json
    Adjusts: retry_slippage_bps, cancel_after_seconds

Phase 3: FAILURE NARRATIVE
    Parse 24h of JSONL logs → FailureStats
    Rank: CB triggers, bear regime %, cancelled orders, network errors

Phase 4: MUTATION PROPOSALS
    3 deterministic mutations targeted at top failure mode
    Examples: LOOSEN_BEAR_FILTER, WIDER_BANDS, WIDER_STOPS
    Each mutation: {name, rationale, param_changes}

Phase 4b: LLM CONSULTANT (--enable-llm)
    Model: claude-sonnet-4-20250514 via Anthropic API
    Input: audit metrics + failure narrative + generate_signals() source
    Output: rewritten generate_signals() method (structural change)
    5-gate validation: compile → exec → instantiate → shape → range
    Any gate failure → excluded (non-fatal, deterministic path continues)

Phase 5: PROVING GROUND
    generate_mock_ohlcv(hours=8760, seed=42)
    Backtest each variant at 15bps friction
    Compare: CAGR, Sharpe, MaxDD, trade count

Phase 6: HOT-SWAP DECISION
    Winner must beat baseline Sharpe by ≥10% AND have ≥10 trades
    Deterministic: regex-based constructor default replacement
    LLM: full source file replacement (pre-validated)
    Always: backup .bak, write mutation_meta.json, Telegram alert
    Exit code 2
```

### Hot-Swap Mechanism

Uses **regex matching** to find the current parameter value (not baseline):

```python
regex = re.compile(rf"({re.escape(param)}:\s*{re.escape(type_hint)}\s*=\s*){value_pattern}")
result = regex.sub(rf"\g<1>{new_src}", result, count=1)
```

This makes consecutive swaps work — the pattern matches whatever value is
currently in the file, not just the original default.

---

## Bridge Architecture

### Cycle Rhythm

```
:00  Full trading cycle (fetch OHLCV → signals → orders → log)
:01  Health check (60s interval)
:02  Health check
...  3 consecutive failures → System Pause → exit(1)
:59  Health check
:00  Full trading cycle
```

### JSONL Log Tracks

```
SYSTEM:      {track, event, ts, detail}
             Events: STARTUP, HEALTH_FAIL, SYSTEM_PAUSE

THEORETICAL: {track, cycle_id, ts, signal, current_qty_btc, target_qty_btc,
              delta_qty_btc, current_price_usd, bear_regime, cb_active}

ATTEMPTED:   {track, event, cycle_id, ts, order_id, side, qty, price, ...}
             Events: ORDER_PLACED, ORDER_FILLED, ORDER_CANCELLED,
                     ORDER_ERROR, BALANCE_CHECK
```

### Key Behavior

- **Stateless**: Wallet is single source of truth. Safe to restart mid-cycle.
- **Config**: Reads `bridge_override.json` once at startup (not hot-reloaded).
- **Shutdown**: `finally` block auto-runs `audit.py` on any exit.
- **System Pause**: 3 consecutive health failures → Telegram alert → `exit(1)`.

---

## Orchestration (run_lab.sh)

```
meta_loop.py exit code
    │
    ├── 0 (No swap) → Sleep until next hour, bridge keeps running
    │
    ├── 2 (Hot-swap) → SIGTERM bridge (30s grace for audit.py)
    │                   Sleep 5s, loop restarts, new bridge loads new code
    │
    └── 1 (Error) → Telegram critical alert
                     Kill bridge, script exits with code 1
                     launchd manages restart policy
```

**Features**: PID tracking, SIGTERM/SIGINT trapping, .env auto-sourcing,
venv Python discovery, 60s bridge health checks during sleep,
hour-boundary-aligned sleep.

---

## Environment Variables

| Variable | Required | Used By |
|----------|----------|---------|
| `BINANCE_TESTNET_API_KEY` | Yes (bridge) | bridge.py |
| `BINANCE_TESTNET_API_SECRET` | Yes (bridge) | bridge.py |
| `TELEGRAM_BOT_TOKEN` | Optional | alerts.py (no-op if missing) |
| `TELEGRAM_CHAT_ID` | Optional | alerts.py (no-op if missing) |
| `ANTHROPIC_API_KEY` | Optional | meta_loop.py Phase 4b (skipped if missing) |

---

## Commands

```bash
# Full autonomous loop
./run_lab.sh

# Bridge only (dry-run)
python bridge.py --dry-run

# Meta loop (analysis + optimization)
python meta_loop.py --enable-llm
python meta_loop.py --enable-llm --dry-run    # no file writes

# Standalone backtest
python main.py all

# Dashboard
python -m streamlit run dashboard.py

# Telegram diagnostic
python test_alerts.py

# Monitor
tail -f logs/wrapper.log
tail -5 logs/bridge_$(date -u +%Y%m%d).jsonl
```

---

## Development Notes

- **Vectorized only** — no for-loops over time series. Polars expressions exclusively.
- **T+1 execution** — signal at close of candle T, fill at open of T+1.
- **Deterministic backtests** — `seed=42` for reproducibility across mutations.
- **Non-destructive hot-swap** — always backs up to `.bak` before rewriting.
- **Regex param replacement** — matches current value, not baseline. Consecutive swaps work.
- **Fire-and-forget alerts** — Telegram failures never crash the trading loop.
- **Graceful degradation** — missing Telegram creds = silent no-ops. Missing API key = LLM skipped.
