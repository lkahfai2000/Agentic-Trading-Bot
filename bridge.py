"""Paper-trading bridge: 1h Signal / 15m Execution Hybrid Model on Binance Testnet.

Architecture
============
The bridge runs a 1h/15m hybrid execution model:

  1h Signal Layer (every epoch-hour boundary):
     Fetches 1100 x 1h OHLCV candles and runs VolatilitySqueezeBreakout.generate_signals()
     to compute the directional signal in [-1.0, 1.0]. The signal is stored in a
     PersistenceWindow with 4 execution opportunities (one per 15m slot).

  15m Sniper Layer (every epoch-15min boundary):
     Fetches 100 x 15m OHLCV candles and checks whether the current 15m close is on
     the correct side of the 15m EMA(21) — the "sniper confirmation" gate.
     Only confirmed signals result in a limit order. The persistence window
     counts down 4→0; at 0 the signal expires until the next 1h bar.

  Execution (OrderManager):
     Passive limit → Cancel-After-X timeout → Aggressive retry at mid ± slippage.

  Dual-Track JSONL logging:
     THEORETICAL records now include sniper_confirmed, ema_15m, persistence_remaining.
     ATTEMPTED records are unchanged.

No position state is stored in memory. The wallet IS the state.

Usage
=====
    python bridge.py                                    # live hybrid mode
    python bridge.py --symbol BTC/USDT --max-exposure 0.5 --log-dir ./logs
    python bridge.py --dry-run                          # log THEORETICAL only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import polars as pl

from alerts import TelegramAlerter
from strategies.volatility_squeeze import VolatilitySqueezeBreakout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BridgeConfig:
    """All tuneable parameters for the paper-trading bridge."""

    # Exchange
    symbol: str = "BTC/USDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"

    # 1h signal timeframe
    timeframe: str = "1h"
    # Candle lookback — needs warmup for EMA (bear_ema_span=800)
    # + squeeze_lookback(240) + ATR warmup buffer
    candle_lookback: int = 1100

    # 15m sniper timeframe
    timeframe_15m: str = "15m"
    candle_lookback_15m: int = 100  # 25h of 15m data; sufficient for EMA(21) warmup
    ema_15m_period: int = 21        # 15m EMA period for sniper confirmation (~5.25h)
    persistence_cycles: int = 4     # max 15m execution windows per 1h signal

    # Risk
    max_account_exposure: float = 0.95  # max fraction of USDT balance to risk
    min_trade_notional: float = 10.0  # minimum USD value to bother trading

    # Order persistence (Cancel-After-X)
    cancel_after_seconds: int = 300  # cancel unfilled orders after 5 minutes
    poll_interval_seconds: int = 30  # how often to poll order status
    retry_slippage_bps: float = 15.0  # extra slippage on aggressive retry

    # Health check
    max_consecutive_failures: int = 3  # failures before System Pause
    cycle_interval_seconds: int = 900  # 15m execution cycle (was 3600)
    health_poll_interval: int = 60  # sub-cycle health poll (seconds)

    # Regime detection params (must match strategy exactly)
    bear_ema_com: float = 399.5  # (bear_ema_span - 1) / 2.0 where span=800
    cb_atr_mult: float = 3.0
    cb_atr_lookback: int = 168

    # Logging
    log_dir: str = "./logs"
    log_level: str = "INFO"

    # Mode
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Dual-Track Logger
# ---------------------------------------------------------------------------


class DualTrackLogger:
    """Logs Theoretical Signal vs Attempted Order as structured JSONL entries.

    Each cycle produces two kinds of records:
      THEORETICAL - what the strategy computed (signal, target_qty, price)
      ATTEMPTED   - what the exchange received (order_id, status, fill, error)

    Both records share a cycle_id for easy join/correlation.
    """

    def __init__(self, log_dir: str, log_level: str = "INFO") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = Path(log_dir) / f"bridge_{date_str}.jsonl"

        # Human-readable stderr logger
        self._console = logging.getLogger("bridge.console")
        self._console.setLevel(getattr(logging, log_level))
        if not self._console.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%SZ",
                )
            )
            self._console.addHandler(h)

        # Structured JSONL file logger
        self._file = logging.getLogger("bridge.jsonl")
        self._file.setLevel(logging.DEBUG)
        if not self._file.handlers:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._file.addHandler(fh)

        self._cycle_id: Optional[str] = None

    def new_cycle(self, ts: datetime) -> str:
        """Start a new logging cycle aligned to 15m boundaries. Returns cycle_id."""
        aligned_min = (ts.minute // 15) * 15
        self._cycle_id = ts.replace(
            minute=aligned_min, second=0, microsecond=0
        ).isoformat()
        self._console.info(f"{'─' * 20} Cycle {self._cycle_id} {'─' * 20}")
        return self._cycle_id

    def _write(self, record: dict) -> None:
        self._file.info(json.dumps(record, default=str))

    # -- THEORETICAL track --------------------------------------------------

    def log_theoretical(
        self,
        signal: float,
        current_qty_btc: float,
        target_qty_btc: float,
        delta_qty_btc: float,
        current_price_usd: float,
        bear_regime: bool,
        cb_active: bool,
        max_usdt_allowed: float,
        sniper_confirmed: bool = False,
        ema_15m: float = 0.0,
        persistence_remaining: int = 0,
    ) -> None:
        record = {
            "track": "THEORETICAL",
            "cycle_id": self._cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "signal": round(signal, 6),
            "current_qty_btc": round(current_qty_btc, 8),
            "target_qty_btc": round(target_qty_btc, 8),
            "delta_qty_btc": round(delta_qty_btc, 8),
            "current_price_usd": round(current_price_usd, 2),
            "bear_regime": bear_regime,
            "cb_active": cb_active,
            "max_usdt_allowed": round(max_usdt_allowed, 2),
            # 1h/15m hybrid fields
            "sniper_confirmed": sniper_confirmed,
            "ema_15m": round(ema_15m, 2),
            "persistence_remaining": persistence_remaining,
        }
        self._write(record)
        side = (
            "BUY" if delta_qty_btc > 0 else "SELL" if delta_qty_btc < 0 else "FLAT"
        )
        sniper_tag = "SNIPER:OK" if sniper_confirmed else "SNIPER:WAIT"
        self._console.info(
            f"[THEORETICAL] signal={signal:+.4f} | delta={delta_qty_btc:+.6f} BTC "
            f"({side}) | bear={bear_regime} cb={cb_active} | "
            f"{sniper_tag} ema15m={ema_15m:.2f} persist={persistence_remaining}/4"
        )

    # -- ATTEMPTED track ----------------------------------------------------

    def log_order_placed(
        self,
        order_id: str,
        side: str,
        qty: float,
        price: float,
        is_retry: bool = False,
    ) -> None:
        record = {
            "track": "ATTEMPTED",
            "event": "ORDER_PLACED",
            "cycle_id": self._cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "side": side,
            "qty": round(qty, 8),
            "price": round(price, 2),
            "is_retry": is_retry,
        }
        self._write(record)
        retry_tag = " [RETRY-AGGRESSIVE]" if is_retry else ""
        self._console.info(
            f"[ORDER PLACED{retry_tag}] {side} {qty:.6f} BTC "
            f"@ ${price:.2f} | id={order_id}"
        )

    def log_order_filled(
        self, order_id: str, filled_qty: float, avg_price: float
    ) -> None:
        record = {
            "track": "ATTEMPTED",
            "event": "ORDER_FILLED",
            "cycle_id": self._cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "filled_qty": round(filled_qty, 8),
            "avg_price": round(avg_price, 2),
        }
        self._write(record)
        self._console.info(
            f"[ORDER FILLED] {filled_qty:.6f} BTC "
            f"@ ${avg_price:.2f} | id={order_id}"
        )

    def log_order_cancelled(self, order_id: str, reason: str) -> None:
        record = {
            "track": "ATTEMPTED",
            "event": "ORDER_CANCELLED",
            "cycle_id": self._cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "reason": reason,
        }
        self._write(record)
        self._console.warning(
            f"[ORDER CANCELLED] id={order_id} | reason={reason}"
        )

    def log_order_error(
        self, error_code: str, error_msg: str, context: dict
    ) -> None:
        binance_code = _extract_binance_code(error_msg)
        record = {
            "track": "ATTEMPTED",
            "event": "ORDER_ERROR",
            "cycle_id": self._cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "error_code": error_code,
            "binance_code": binance_code,
            "error_msg": error_msg,
            "context": context,
        }
        self._write(record)
        self._console.error(
            f"[ORDER ERROR] code={error_code} binance={binance_code} "
            f"| {error_msg}"
        )

    def log_balance_check(
        self, usdt_free: float, btc_total: float, max_usdt: float
    ) -> None:
        record = {
            "track": "ATTEMPTED",
            "event": "BALANCE_CHECK",
            "cycle_id": self._cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "usdt_free": round(usdt_free, 2),
            "btc_total": round(btc_total, 8),
            "max_usdt_allowed": round(max_usdt, 2),
        }
        self._write(record)
        self._console.info(
            f"[BALANCE] USDT free=${usdt_free:.2f} | BTC={btc_total:.6f} "
            f"| max_deploy=${max_usdt:.2f}"
        )

    # -- SYSTEM track -------------------------------------------------------

    def log_system_event(self, event: str, detail: str) -> None:
        record = {
            "track": "SYSTEM",
            "event": event,
            "cycle_id": self._cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "detail": detail,
        }
        self._write(record)
        self._console.warning(f"[SYSTEM:{event}] {detail}")

    # -- Convenience --------------------------------------------------------

    def info(self, msg: str) -> None:
        self._console.info(msg)

    def warning(self, msg: str) -> None:
        self._console.warning(msg)

    def error(self, msg: str) -> None:
        self._console.error(msg)


def _extract_binance_code(msg: str) -> str | None:
    """Extract 4-5 digit Binance error code (e.g. -1013, -2010) from message."""
    m = re.search(r"(-\d{4,5})", msg)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Tracks consecutive data-fetch failures and triggers System Pause.

    After max_failures consecutive failed health checks (each 60s apart),
    sets paused=True.  Caller must check is_paused() and handle accordingly.
    """

    def __init__(self, max_failures: int = 3) -> None:
        self.max_failures = max_failures
        self._consecutive_failures = 0
        self._paused = False

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._paused = False

    def record_failure(self) -> int:
        """Record a failure. Returns current consecutive failure count."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_failures:
            self._paused = True
        return self._consecutive_failures

    def is_paused(self) -> bool:
        return self._paused

    def reset(self) -> None:
        self._consecutive_failures = 0
        self._paused = False


# ---------------------------------------------------------------------------
# Persistence Window — 1h/15m Hybrid Signal State
# ---------------------------------------------------------------------------


@dataclass
class PersistenceWindow:
    """Carries the 1h signal across up to 4 x 15m sniper execution windows.

    Lifecycle:
      - On each epoch-hour boundary: signal is refreshed, cycles_remaining reset to 4.
      - On each epoch-15min boundary: cycles_remaining decremented after sniper check.
      - When cycles_remaining reaches 0: no execution until the next 1h refresh.
    """

    signal: float = 0.0
    epoch_hour: int = -1       # epoch-hour that last generated this signal
    cycles_remaining: int = 0  # counts down 4→0 across 15m windows


# ---------------------------------------------------------------------------
# Binance Testnet Client (CCXT wrapper)
# ---------------------------------------------------------------------------


class BinanceTestnetClient:
    """Thin CCXT wrapper for Binance Testnet with Hummingbot-compatible calls.

    All methods return raw CCXT dicts or raise ccxt exceptions with full
    error codes — callers must handle exceptions.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": {
                    "defaultType": "spot",
                    "warnOnFetchOpenOrdersWithoutSymbol": False,
                },
                "enableRateLimit": True,
            }
        )
        # Activate Binance Testnet (redirects all API calls to testnet URLs)
        self._exchange.set_sandbox_mode(True)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[list]:
        """Fetch recent OHLCV candles. Returns [[ts_ms, O, H, L, C, V], ...]"""
        return self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch best bid/ask and last price."""
        return self._exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> dict:
        """Fetch full account balance dict."""
        return self._exchange.fetch_balance()

    def create_limit_buy(
        self, symbol: str, qty: float, price: float
    ) -> dict:
        return self._exchange.create_limit_buy_order(symbol, qty, price)

    def create_limit_sell(
        self, symbol: str, qty: float, price: float
    ) -> dict:
        return self._exchange.create_limit_sell_order(symbol, qty, price)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        return self._exchange.fetch_order(order_id, symbol)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return self._exchange.cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: str) -> list[dict]:
        return self._exchange.fetch_open_orders(symbol)


# ---------------------------------------------------------------------------
# Order Manager — Cancel-After-X + Aggressive Retry
# ---------------------------------------------------------------------------


class OrderManager:
    """Places a limit order and monitors it until filled or timeout.

    Cancel-After-X logic:
      1. Place passive limit order (at bid for buys, ask for sells).
      2. Poll every poll_interval_seconds for up to cancel_after_seconds.
      3. If still open at timeout:
           a. Cancel the passive order.
           b. Retry once with an aggressive price (mid +/- slippage) to
              cross the spread and fill immediately.
      4. If retry also times out or fails, log and wait for next cycle.
    """

    def __init__(
        self,
        client: BinanceTestnetClient,
        logger: DualTrackLogger,
        alerter: TelegramAlerter,
        cancel_after_seconds: int = 300,
        poll_interval_seconds: int = 30,
        retry_slippage_bps: float = 15.0,
        dry_run: bool = False,
    ) -> None:
        self._client = client
        self._log = logger
        self._alerts = alerter
        self._cancel_after = cancel_after_seconds
        self._poll_interval = poll_interval_seconds
        self._retry_slip = retry_slippage_bps / 10_000.0
        self._dry_run = dry_run

    def place_and_watch(
        self,
        symbol: str,
        side: str,
        qty: float,
        bid: float,
        ask: float,
    ) -> bool:
        """Place a passive limit order, watch it, retry aggressively if needed.

        Returns True if a fill was confirmed, False otherwise.
        """
        # Passive price: bid for buys (join the queue), ask for sells
        passive_price = bid if side == "buy" else ask
        mid = (bid + ask) / 2.0

        order_id = self._place_order(
            symbol, side, qty, passive_price, is_retry=False
        )
        if order_id is None:
            return False

        # In dry-run mode, fake order is always "filled"
        if self._dry_run:
            return True

        filled = self._watch_order(order_id, symbol)
        if filled:
            return True

        # Timeout — cancel and retry aggressively
        self._cancel_existing(order_id, symbol, "CANCEL_AFTER_X_TIMEOUT")

        # Aggressive retry: cross the spread by retry_slippage_bps
        if side == "buy":
            aggressive_price = mid * (1 + self._retry_slip)
        else:
            aggressive_price = mid * (1 - self._retry_slip)

        retry_id = self._place_order(
            symbol, side, qty, aggressive_price, is_retry=True
        )
        if retry_id is None:
            return False

        filled = self._watch_order(retry_id, symbol)
        if not filled:
            # Retry also timed out — give up until next cycle
            self._cancel_existing(
                retry_id, symbol, "CANCEL_AFTER_X_RETRY_TIMEOUT"
            )
        return filled

    def _place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        is_retry: bool,
    ) -> Optional[str]:
        """Place a single limit order. Returns order_id or None on failure."""
        if self._dry_run:
            fake_id = f"DRY_RUN_{int(time.time())}"
            self._log.log_order_placed(
                fake_id, side.upper(), qty, price, is_retry
            )
            self._alerts.order_event(
                "ORDER_PLACED", fake_id,
                side=side.upper(), qty=qty, price=price, is_retry=is_retry,
            )
            return fake_id

        try:
            if side == "buy":
                order = self._client.create_limit_buy(symbol, qty, price)
            else:
                order = self._client.create_limit_sell(symbol, qty, price)

            order_id = order["id"]
            self._log.log_order_placed(
                order_id, side.upper(), qty, price, is_retry
            )
            self._alerts.order_event(
                "ORDER_PLACED", order_id,
                side=side.upper(), qty=qty, price=price, is_retry=is_retry,
            )
            return order_id

        except ccxt.InsufficientFunds as e:
            self._log.log_order_error(
                "InsufficientFunds",
                str(e),
                {"side": side, "qty": qty, "price": price},
            )
        except ccxt.InvalidOrder as e:
            self._log.log_order_error(
                "InvalidOrder",
                str(e),
                {"side": side, "qty": qty, "price": price},
            )
        except ccxt.AuthenticationError as e:
            self._log.log_order_error(
                "AuthenticationError",
                str(e),
                {"side": side, "qty": qty, "price": price},
            )
        except ccxt.RateLimitExceeded as e:
            self._log.log_order_error(
                "RateLimitExceeded",
                str(e),
                {"side": side, "qty": qty, "price": price},
            )
        except ccxt.NetworkError as e:
            self._log.log_order_error(
                "NetworkError",
                str(e),
                {"side": side, "qty": qty, "price": price},
            )
        except ccxt.ExchangeError as e:
            self._log.log_order_error(
                type(e).__name__,
                str(e),
                {"side": side, "qty": qty, "price": price},
            )
        return None

    def _watch_order(self, order_id: str, symbol: str) -> bool:
        """Poll order until filled or cancel_after timeout."""
        deadline = time.monotonic() + self._cancel_after

        while time.monotonic() < deadline:
            time.sleep(self._poll_interval)
            try:
                order = self._client.fetch_order(order_id, symbol)
                status = order.get("status", "unknown")

                if status == "closed":
                    fq = float(order.get("filled", 0.0))
                    ap = float(order.get("average", 0.0))
                    self._log.log_order_filled(
                        order_id, filled_qty=fq, avg_price=ap,
                    )
                    self._alerts.order_event(
                        "ORDER_FILLED", order_id,
                        filled_qty=fq, avg_price=ap,
                    )
                    return True

                if status == "canceled":
                    self._log.log_order_cancelled(
                        order_id, reason="EXCHANGE_CANCELLED"
                    )
                    self._alerts.order_event(
                        "ORDER_CANCELLED", order_id,
                        reason="EXCHANGE_CANCELLED",
                    )
                    return False

                # status == "open" — keep watching
                remaining = deadline - time.monotonic()
                self._log.info(
                    f"[WATCHING] id={order_id} status={status} "
                    f"filled={order.get('filled', 0)} "
                    f"remaining={remaining:.0f}s"
                )

            except ccxt.OrderNotFound:
                self._log.log_order_error(
                    "OrderNotFound",
                    f"order {order_id} not found on exchange",
                    {"order_id": order_id},
                )
                return False
            except ccxt.NetworkError as e:
                # Non-fatal during polling — log and continue watching
                self._log.log_order_error(
                    "NetworkError",
                    str(e),
                    {"action": "fetch_order", "order_id": order_id},
                )

        return False  # Timed out

    def _cancel_existing(
        self, order_id: str, symbol: str, reason: str
    ) -> None:
        """Best-effort cancel of an existing order."""
        self._log.log_order_cancelled(order_id, reason=reason)
        try:
            self._client.cancel_order(order_id, symbol)
        except ccxt.OrderNotFound:
            pass  # Already gone — treat as success
        except ccxt.BaseError as e:
            self._log.log_order_error(
                type(e).__name__,
                str(e),
                {"action": "cancel", "order_id": order_id},
            )


# ---------------------------------------------------------------------------
# Trading Bridge — Main Orchestration
# ---------------------------------------------------------------------------


class TradingBridge:
    """1h-Signal / 15m-Execution hybrid paper-trading bridge.

    On each epoch-hour boundary:
      1. Fetch 1h OHLCV -> run VolatilitySqueezeBreakout -> store signal in PersistenceWindow
      2. Reset persistence window to 4 execution opportunities.

    On each epoch-15min boundary (while cycles_remaining > 0):
      1. Fetch 15m OHLCV -> compute 15m EMA(21) -> check sniper confirmation
      2. If NOT confirmed: log THEORETICAL with sniper_confirmed=False, skip order
      3. If confirmed: query balance, compute delta, execute via OrderManager
      4. Decrement cycles_remaining
    """

    def __init__(
        self, config: BridgeConfig, client: BinanceTestnetClient
    ) -> None:
        self._cfg = config
        self._client = client
        self._strategy = VolatilitySqueezeBreakout()
        self._log = DualTrackLogger(config.log_dir, config.log_level)
        self._health = HealthMonitor(config.max_consecutive_failures)
        self._alerts = TelegramAlerter()
        self._order_mgr = OrderManager(
            client=client,
            logger=self._log,
            alerter=self._alerts,
            cancel_after_seconds=config.cancel_after_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
            retry_slippage_bps=config.retry_slippage_bps,
            dry_run=config.dry_run,
        )
        # 1h/15m hybrid state — signal persists across 4 x 15m sniper windows
        self._persistence = PersistenceWindow()

    # -- Data fetching ------------------------------------------------------

    def _fetch_ohlcv_df(self) -> pl.DataFrame:
        """Fetch 1h OHLCV from exchange and return as Polars DataFrame."""
        raw = self._client.fetch_ohlcv(
            self._cfg.symbol,
            self._cfg.timeframe,
            limit=self._cfg.candle_lookback,
        )
        # CCXT format: [[timestamp_ms, open, high, low, close, volume], ...]
        df = pl.DataFrame(
            raw,
            schema=[
                "timestamp_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
            orient="row",
        )
        return (
            df.with_columns(
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

    def _fetch_15m_ohlcv_df(self) -> pl.DataFrame:
        """Fetch 15m OHLCV from exchange for sniper EMA computation."""
        raw = self._client.fetch_ohlcv(
            self._cfg.symbol,
            self._cfg.timeframe_15m,
            limit=self._cfg.candle_lookback_15m,
        )
        df = pl.DataFrame(
            raw,
            schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        return (
            df.with_columns(
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

    # -- Regime context for logging -----------------------------------------

    def _compute_regime_flags(
        self, df: pl.DataFrame
    ) -> tuple[bool, bool]:
        """Compute bear_regime and cb_active from OHLCV for dual-track logging.

        Uses the same parameters as the strategy so the flags are identical
        to what generate_signals() applies internally.
        """
        # Bear regime: close < EMA(com=399.5)
        ema = df["close"].ewm_mean(
            com=self._cfg.bear_ema_com, adjust=False
        )
        bear_regime = bool(df["close"][-1] < ema[-1])

        # Circuit breaker: ATR spike > 3x rolling baseline
        # Use with_columns to keep Polars expressions in DataFrame context
        tmp = df.with_columns([
            (pl.col("high") - pl.col("low")).alias("_tr_hl"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("_tr_hc"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("_tr_lc"),
        ]).with_columns(
            pl.max_horizontal("_tr_hl", "_tr_hc", "_tr_lc").alias("_tr")
        ).with_columns(
            pl.col("_tr").rolling_mean(14).alias("_atr")
        ).with_columns(
            pl.col("_atr")
            .rolling_mean(self._cfg.cb_atr_lookback)
            .alias("_atr_baseline")
        )

        atr_val = float(tmp["_atr"][-1])
        baseline_val = float(tmp["_atr_baseline"][-1])
        ratio = atr_val / (baseline_val + 1e-10)
        cb_active = ratio > self._cfg.cb_atr_mult

        return bear_regime, cb_active

    # -- Balance / position -------------------------------------------------

    def _query_position(self) -> tuple[float, float, float]:
        """Query wallet. Returns (usdt_free, btc_total, max_usdt_to_deploy)."""
        balance = self._client.fetch_balance()
        usdt_free = float(
            balance.get(self._cfg.quote_asset, {}).get("free", 0.0)
        )
        btc_total = float(
            balance.get(self._cfg.base_asset, {}).get("total", 0.0)
        )
        max_usdt = usdt_free * self._cfg.max_account_exposure
        self._log.log_balance_check(usdt_free, btc_total, max_usdt)
        return usdt_free, btc_total, max_usdt

    # -- Signal -> size computation -----------------------------------------

    @staticmethod
    def _compute_trade(
        signal: float,
        btc_held: float,
        max_usdt: float,
        current_price: float,
    ) -> tuple[float, float]:
        """Compute (target_btc, delta_btc) from the current signal.

        Spot-only: signal clamped to [0, 1] (no short selling on spot).
        target_btc = (clamped_signal * max_usdt) / price
        delta_btc  = target_btc - btc_held
        """
        effective_signal = max(min(signal, 1.0), 0.0)

        if current_price <= 0:
            return 0.0, -btc_held

        target_usdt = effective_signal * max_usdt
        target_btc = target_usdt / current_price
        delta_btc = target_btc - btc_held

        return target_btc, delta_btc

    # -- 1h signal layer -------------------------------------------------------

    def _run_1h_signal_cycle(self, epoch_hour: int) -> None:
        """Fetch 1h OHLCV, run strategy, refresh the PersistenceWindow.

        Called once per epoch-hour boundary. Updates self._persistence with
        the new signal and resets cycles_remaining to persistence_cycles (4).
        """
        self._log.info(
            f"[1H SIGNAL] Refreshing signal for epoch_hour={epoch_hour}"
        )

        # Fetch 1h OHLCV
        try:
            df_1h = self._fetch_ohlcv_df()
            self._health.record_success()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            count = self._health.record_failure()
            self._log.log_order_error(
                type(e).__name__,
                str(e),
                {"action": "fetch_ohlcv_1h", "consecutive_failures": count},
            )
            return

        # Run strategy
        try:
            signals = self._strategy.generate_signals(df_1h)
        except Exception as e:
            self._log.log_system_event("STRATEGY_ERROR", f"{type(e).__name__}: {e}")
            return

        new_signal = float(signals[-1]) if len(signals) > 0 else 0.0

        # Update persistence window
        self._persistence.signal = new_signal
        self._persistence.epoch_hour = epoch_hour
        self._persistence.cycles_remaining = self._cfg.persistence_cycles

        self._log.info(
            f"[1H SIGNAL] signal={new_signal:+.4f} | "
            f"persistence_cycles={self._cfg.persistence_cycles} (15m windows loaded)"
        )

    # -- 15m sniper layer ------------------------------------------------------

    def _run_15m_sniper_cycle(self) -> None:
        """15m sniper: confirm 1h signal via 15m EMA gate, then execute if confirmed.

        Called every epoch-15min boundary while persistence.cycles_remaining > 0.
        Decrements cycles_remaining after the sniper check regardless of outcome.
        """
        now = datetime.now(timezone.utc)
        self._log.new_cycle(now)

        signal = self._persistence.signal
        remaining_before = self._persistence.cycles_remaining

        # Step 1: Decrement persistence window (consume this 15m slot)
        self._persistence.cycles_remaining -= 1
        persistence_remaining = self._persistence.cycles_remaining

        self._log.info(
            f"[15M SNIPER] signal={signal:+.4f} | "
            f"window={remaining_before}/4 → {persistence_remaining}/4"
        )

        # Step 2: Fetch 15m OHLCV for sniper EMA
        try:
            df_15m = self._fetch_15m_ohlcv_df()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            self._log.log_order_error(
                type(e).__name__, str(e), {"action": "fetch_ohlcv_15m"}
            )
            return

        # Step 3: Sniper confirmation gate
        sniper_confirmed, ema_15m = VolatilitySqueezeBreakout.get_15m_confirmation(
            df_15m, signal, self._cfg.ema_15m_period
        )

        # Step 4: Compute regime context for dual-track logging
        try:
            df_1h_cache = self._fetch_ohlcv_df()
            bear_regime, cb_active = self._compute_regime_flags(df_1h_cache)
        except (ccxt.NetworkError, ccxt.ExchangeError):
            # Non-fatal: use safe defaults if 1h fetch fails during 15m check
            bear_regime, cb_active = False, False

        # Step 5: Fetch ticker for bid/ask
        try:
            ticker = self._client.fetch_ticker(self._cfg.symbol)
            bid = float(ticker["bid"])
            ask = float(ticker["ask"])
            current_price = float(ticker["last"])
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            self._log.log_order_error(
                type(e).__name__, str(e), {"action": "fetch_ticker"}
            )
            return

        # Step 6: Balance check
        try:
            usdt_free, btc_held, max_usdt = self._query_position()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            self._log.log_order_error(
                type(e).__name__, str(e), {"action": "fetch_balance"}
            )
            return

        # Step 7: Compute trade
        target_btc, delta_btc = self._compute_trade(
            signal, btc_held, max_usdt, current_price
        )

        # Step 8: Log THEORETICAL (always, sniper confirmed or not)
        self._log.log_theoretical(
            signal=signal,
            current_qty_btc=btc_held,
            target_qty_btc=target_btc,
            delta_qty_btc=delta_btc,
            current_price_usd=current_price,
            bear_regime=bear_regime,
            cb_active=cb_active,
            max_usdt_allowed=max_usdt,
            sniper_confirmed=sniper_confirmed,
            ema_15m=ema_15m,
            persistence_remaining=persistence_remaining,
        )

        # Telegram: trade signal snapshot
        self._alerts.trade_signal(
            signal=signal,
            delta_qty_btc=delta_btc,
            current_price_usd=current_price,
            bear_regime=bear_regime,
            cb_active=cb_active,
            cycle_id=self._log._cycle_id,
            sniper_confirmed=sniper_confirmed,
            persistence_remaining=persistence_remaining,
            ema_15m=ema_15m,
        )

        # Step 9: Sniper gate — skip execution if not confirmed
        if not sniper_confirmed:
            self._log.info(
                f"[SNIPER SKIP] 15m close {float(df_15m['close'][-1]):.2f} "
                f"vs EMA(21) {ema_15m:.2f} — signal not confirmed, no order"
            )
            self._log.log_system_event(
                "SNIPER_SKIP",
                f"signal={signal:+.4f} close={float(df_15m['close'][-1]):.2f} "
                f"ema15m={ema_15m:.2f} persist={persistence_remaining}",
            )
            return

        # PnL snapshot only when executing (not on every 15m check)
        self._alerts.hourly_pnl(
            usdt_free=usdt_free,
            btc_total=btc_held,
            btc_price=current_price,
            cycle_id=self._log._cycle_id,
        )

        # Step 10: Is the trade worth executing?
        delta_notional = abs(delta_btc) * current_price
        if delta_notional < self._cfg.min_trade_notional:
            self._log.info(
                f"[SKIP] delta_notional=${delta_notional:.2f} < "
                f"min=${self._cfg.min_trade_notional:.2f} — no trade"
            )
            return

        # Step 11: Execute via OrderManager
        side = "buy" if delta_btc > 0 else "sell"
        qty_to_trade = abs(delta_btc)

        if side == "sell":
            qty_to_trade = min(qty_to_trade, btc_held)
            if qty_to_trade < 1e-8:
                self._log.info("[SKIP] Nothing to sell — no BTC held")
                return

        self._order_mgr.place_and_watch(
            symbol=self._cfg.symbol,
            side=side,
            qty=qty_to_trade,
            bid=bid,
            ask=ask,
        )

    # -- Health check -------------------------------------------------------

    def _run_health_check(self) -> None:
        """Lightweight health check — fetch ticker every 60s."""
        try:
            self._client.fetch_ticker(self._cfg.symbol)
            self._health.record_success()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            count = self._health.record_failure()
            detail = (
                f"consecutive={count}/{self._cfg.max_consecutive_failures} "
                f"| {type(e).__name__}: {e}"
            )
            self._log.log_system_event("HEALTH_FAIL", detail)
            self._alerts.health_failure(
                consecutive=count,
                max_failures=self._cfg.max_consecutive_failures,
                error_detail=str(e),
            )

    # -- System pause -------------------------------------------------------

    def _trigger_system_pause(self, reason: str) -> None:
        """Halt trading and alert operator until manual restart."""
        self._log.log_system_event("SYSTEM_PAUSE", reason)
        self._alerts.system_pause(reason)
        print(
            f"\n{'=' * 60}\n"
            f"  *** SYSTEM PAUSE TRIGGERED ***\n"
            f"  Reason: {reason}\n"
            f"  Action: Review logs in {self._cfg.log_dir}\n"
            f"          Fix the issue, then restart the bridge.\n"
            f"{'=' * 60}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Startup cleanup ----------------------------------------------------

    def _cancel_stale_orders(self) -> None:
        """Cancel any open orders left over from a previous run."""
        try:
            open_orders = self._client.fetch_open_orders(self._cfg.symbol)
            for order in open_orders:
                try:
                    self._client.cancel_order(
                        order["id"], self._cfg.symbol
                    )
                    self._log.log_order_cancelled(
                        order["id"], reason="STARTUP_CLEANUP"
                    )
                except ccxt.BaseError as e:
                    self._log.log_order_error(
                        type(e).__name__,
                        str(e),
                        {
                            "action": "startup_cleanup",
                            "order_id": order["id"],
                        },
                    )
        except ccxt.BaseError as e:
            self._log.warning(
                f"Could not fetch open orders on startup: {e}"
            )

    # -- Main loop ----------------------------------------------------------

    def run_forever(self) -> None:
        """Main event loop — 1h Signal / 15m Execution hybrid.

        Epoch-hour boundary  → regenerate 1h signal via VolatilitySqueezeBreakout,
                               reset PersistenceWindow to 4 execution slots.
        Epoch-15min boundary → run 15m sniper: check EMA confirmation, execute if
                               confirmed and cycles_remaining > 0.
        Every 60s            → lightweight health check (ticker ping).
        3 consecutive fails  → System Pause → sys.exit(1).
        """
        mode = "DRY-RUN" if self._cfg.dry_run else "LIVE"
        self._log.log_system_event(
            "STARTUP",
            f"Bridge starting [{mode}] | symbol={self._cfg.symbol} "
            f"exposure={self._cfg.max_account_exposure:.0%} "
            f"model=1h-Signal/15m-Exec "
            f"ema15m_period={self._cfg.ema_15m_period} "
            f"persistence_cycles={self._cfg.persistence_cycles}",
        )
        self._alerts.startup(
            mode=mode,
            symbol=self._cfg.symbol,
            exposure=self._cfg.max_account_exposure,
        )

        # Clean up any orphaned orders from a previous run
        if not self._cfg.dry_run:
            self._cancel_stale_orders()

        last_epoch_hour = -1
        last_epoch_15min = -1

        try:
            while True:
                # Health check every 60s (unchanged from V3)
                self._run_health_check()

                if self._health.is_paused():
                    self._trigger_system_pause(
                        f"Data fetch failed "
                        f"{self._cfg.max_consecutive_failures} consecutive "
                        f"health checks ({self._cfg.max_consecutive_failures} "
                        f"minutes). Exchange may be unreachable."
                    )

                now_ts = int(datetime.now(timezone.utc).timestamp())
                epoch_hour = now_ts // 3600
                epoch_15min = now_ts // 900  # 15m = 900s

                # 1h boundary: refresh signal and load 4 persistence slots
                if epoch_hour != last_epoch_hour:
                    try:
                        self._run_1h_signal_cycle(epoch_hour)
                    except Exception as e:
                        self._log.log_order_error(
                            "UNHANDLED_EXCEPTION",
                            str(e),
                            {"type": type(e).__name__, "phase": "1h_signal"},
                        )
                    last_epoch_hour = epoch_hour

                # 15m boundary: run sniper if persistence window is live
                if epoch_15min != last_epoch_15min:
                    if self._persistence.cycles_remaining > 0:
                        try:
                            self._run_15m_sniper_cycle()
                        except Exception as e:
                            self._log.log_order_error(
                                "UNHANDLED_EXCEPTION",
                                str(e),
                                {"type": type(e).__name__, "phase": "15m_sniper"},
                            )
                    else:
                        self._log.info(
                            "[15M SNIPER] Persistence window exhausted — "
                            "awaiting next 1h signal refresh"
                        )
                    last_epoch_15min = epoch_15min

                time.sleep(self._cfg.health_poll_interval)
        finally:
            # Auto-audit on any exit: Ctrl+C, sys.exit, or crash
            self._log.log_system_event("SHUTDOWN", "Running post-session audit...")
            subprocess.run(
                [sys.executable, "audit.py", self._cfg.log_dir],
                cwd=Path(__file__).parent,
            )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def _apply_config_overrides(cfg: BridgeConfig) -> None:
    """Load friction parameter overrides written by meta_loop.py, if present."""
    override_path = Path("bridge_override.json")
    if not override_path.exists():
        return
    try:
        data = json.loads(override_path.read_text())
        for key, val in data.get("overrides", {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, type(getattr(cfg, key))(val))
                print(f"[CONFIG OVERRIDE] {key} = {getattr(cfg, key)}", file=sys.stderr)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[CONFIG OVERRIDE] bridge_override.json error (ignored): {e}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "V3 Convex Strategy -> Binance Testnet paper-trading bridge"
        )
    )
    p.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair (default: BTC/USDT)",
    )
    p.add_argument(
        "--max-exposure",
        type=float,
        default=0.95,
        help="Max fraction of USDT balance to deploy (default: 0.95)",
    )
    p.add_argument(
        "--cancel-after",
        type=int,
        default=300,
        help="Cancel unfilled orders after N seconds (default: 300)",
    )
    p.add_argument(
        "--log-dir",
        default="./logs",
        help="Directory for JSONL log files (default: ./logs)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all order placement — log THEORETICAL only",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")

    if not api_key or not api_secret:
        print(
            "ERROR: Set BINANCE_TESTNET_API_KEY and "
            "BINANCE_TESTNET_API_SECRET environment variables "
            "before starting the bridge.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = BridgeConfig(
        symbol=args.symbol,
        max_account_exposure=args.max_exposure,
        cancel_after_seconds=args.cancel_after,
        log_dir=args.log_dir,
        log_level=args.log_level,
        dry_run=args.dry_run,
    )

    _apply_config_overrides(config)

    client = BinanceTestnetClient(api_key, api_secret)
    bridge = TradingBridge(config, client)
    bridge.run_forever()


if __name__ == "__main__":
    main()
