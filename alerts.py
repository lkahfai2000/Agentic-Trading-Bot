"""Async Telegram alert system for the paper-trading bridge.

Sends non-blocking notifications via the Telegram Bot API using a background
thread so the main trading loop is never delayed by network I/O.

Three alert types:
  1. Trade Alert   — every THEORETICAL / ATTEMPTED event
  2. Heartbeat     — health failures and SYSTEM_PAUSE emergencies
  3. Hourly PnL    — wallet snapshot at the end of each cycle

Setup
=====
    1. Message @BotFather on Telegram -> /newbot -> copy the token.
    2. Message your new bot, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       to find your chat_id.
    3. Add to .env:
         TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
         TELEGRAM_CHAT_ID=987654321

All sends are fire-and-forget — a failed Telegram call never crashes the bridge.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger("bridge.telegram")

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """Fire-and-forget Telegram message sender backed by a daemon thread.

    Messages are queued and sent from a single background thread so the caller
    never blocks on network I/O.  If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID
    are missing, all methods become silent no-ops.
    """

    def __init__(self) -> None:
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)

        if not self._enabled:
            logger.info(
                "Telegram alerts disabled — "
                "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
            )
            return

        self._url = _TELEGRAM_API.format(token=self._token)
        self._queue: queue.Queue[str] = queue.Queue(maxsize=200)
        self._worker = threading.Thread(
            target=self._sender_loop, daemon=True
        )
        self._worker.start()
        logger.info("Telegram alerts enabled — sender thread started")

    # -- internal sender thread -----------------------------------------------

    def _sender_loop(self) -> None:
        while True:
            text = self._queue.get()
            try:
                data = urlencode(
                    {
                        "chat_id": self._chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": "true",
                    }
                ).encode()
                req = Request(self._url, data=data, method="POST")
                with urlopen(req, timeout=10) as resp:
                    resp.read()
            except (URLError, OSError) as e:
                logger.warning(f"Telegram send failed: {e}")
            finally:
                self._queue.task_done()

    def _send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.warning("Telegram queue full — dropping message")

    # -- public alert methods -------------------------------------------------

    def trade_signal(
        self,
        signal: float,
        delta_qty_btc: float,
        current_price_usd: float,
        bear_regime: bool,
        cb_active: bool,
        cycle_id: Optional[str] = None,
    ) -> None:
        """Send a trade signal alert (THEORETICAL track)."""
        if delta_qty_btc > 0:
            icon, direction = "\U0001f7e2", "BUY"
        elif delta_qty_btc < 0:
            icon, direction = "\U0001f534", "SELL"
        else:
            icon, direction = "\u26aa", "FLAT"

        regime = "\U0001f43b Bear" if bear_regime else "\U0001f402 Bull"
        cb = "\u26a0\ufe0f ON" if cb_active else "\u2705 OFF"

        text = (
            f"{icon} <b>{direction} Signal</b>\n"
            f"Signal: <code>{signal:+.4f}</code>\n"
            f"Delta: <code>{delta_qty_btc:+.6f} BTC</code>\n"
            f"Price: <code>${current_price_usd:,.2f}</code>\n"
            f"Regime: {regime} | CB: {cb}\n"
            f"Cycle: <code>{cycle_id or 'N/A'}</code>"
        )
        self._send(text)

    def order_event(
        self,
        event: str,
        order_id: str,
        side: str = "",
        qty: float = 0.0,
        price: float = 0.0,
        filled_qty: float = 0.0,
        avg_price: float = 0.0,
        reason: str = "",
        is_retry: bool = False,
    ) -> None:
        """Send an order lifecycle alert (ATTEMPTED track)."""
        if event == "ORDER_PLACED":
            icon = "\U0001f4e4"
            retry_tag = " [RETRY]" if is_retry else ""
            detail = f"{side} {qty:.6f} BTC @ ${price:,.2f}{retry_tag}"
        elif event == "ORDER_FILLED":
            icon = "\u2705"
            detail = f"Filled {filled_qty:.6f} BTC @ ${avg_price:,.2f}"
        elif event == "ORDER_CANCELLED":
            icon = "\u274c"
            detail = f"Cancelled — {reason}"
        else:
            icon = "\u2753"
            detail = event

        text = (
            f"{icon} <b>{event}</b>\n"
            f"ID: <code>{order_id}</code>\n"
            f"{detail}"
        )
        self._send(text)

    def order_error(
        self,
        error_code: str,
        error_msg: str,
    ) -> None:
        """Send an order error alert."""
        # Truncate long error messages
        short_msg = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg
        text = (
            f"\u26a0\ufe0f <b>Order Error</b>\n"
            f"Code: <code>{error_code}</code>\n"
            f"<code>{short_msg}</code>"
        )
        self._send(text)

    def health_failure(
        self,
        consecutive: int,
        max_failures: int,
        error_detail: str,
    ) -> None:
        """Send a health check failure alert."""
        if consecutive >= max_failures:
            text = (
                f"\U0001f6a8 <b>CRITICAL — SYSTEM PAUSE</b>\n"
                f"Health check failed {consecutive}/{max_failures} times.\n"
                f"Bridge is HALTING. Manual restart required.\n"
                f"<code>{error_detail[:300]}</code>"
            )
        else:
            text = (
                f"\u26a0\ufe0f <b>Health Check Failed</b>\n"
                f"Consecutive: {consecutive}/{max_failures}\n"
                f"<code>{error_detail[:300]}</code>"
            )
        self._send(text)

    def system_pause(self, reason: str) -> None:
        """Send an emergency SYSTEM_PAUSE alert."""
        text = (
            f"\U0001f6a8\U0001f6a8\U0001f6a8 <b>EMERGENCY — SYSTEM PAUSE</b>\n\n"
            f"{reason}\n\n"
            f"Bridge has STOPPED. Check logs and restart manually."
        )
        self._send(text)

    def hourly_pnl(
        self,
        usdt_free: float,
        btc_total: float,
        btc_price: float,
        cycle_id: Optional[str] = None,
    ) -> None:
        """Send an hourly PnL / wallet snapshot."""
        btc_value = btc_total * btc_price
        total_value = usdt_free + btc_value

        text = (
            f"\U0001f4ca <b>Hourly Wallet Snapshot</b>\n"
            f"USDT: <code>${usdt_free:,.2f}</code>\n"
            f"BTC: <code>{btc_total:.6f}</code> "
            f"(${btc_value:,.2f})\n"
            f"Total: <code>${total_value:,.2f}</code>\n"
            f"BTC Price: <code>${btc_price:,.2f}</code>\n"
            f"Cycle: <code>{cycle_id or 'N/A'}</code>"
        )
        self._send(text)

    def startup(self, mode: str, symbol: str, exposure: float) -> None:
        """Send a startup notification."""
        text = (
            f"\U0001f680 <b>Bridge Started [{mode}]</b>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Exposure: <code>{exposure:.0%}</code>\n"
            f"Time: <code>"
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            f"</code>"
        )
        self._send(text)
