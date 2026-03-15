#!/usr/bin/env python3
"""Telegram integration diagnostic for the Agentic Trading Bot.

Verifies that TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are valid and that
messages render correctly on macOS/iOS Telegram clients.

Sends three test messages:
  1. Connection test   — simple HTML text
  2. Mutation accepted — rocket emoji + HTML formatting via TelegramAlerter
  3. Bridge heartbeat  — current atr_stop_mult from the live strategy file

Usage:
    python test_alerts.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR / ".env"
STRATEGY_FILE = SCRIPT_DIR / "strategies" / "volatility_squeeze.py"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ── .env loader (no external dependencies) ───────────────────────────

def load_env() -> None:
    """Parse .env file and export variables into os.environ."""
    if not ENV_FILE.exists():
        print(f"WARNING: No .env file at {ENV_FILE}")
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


# ── Credential check ─────────────────────────────────────────────────

def check_credentials() -> tuple[str, str]:
    """Verify both Telegram env vars are set. Exit 1 if missing."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    missing = []
    if not token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not chat_id:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        print(f"CRITICAL: ENV MISSING — {', '.join(missing)}")
        print("Set these in your .env file or shell environment.")
        sys.exit(1)

    print(f"  Token:   {token[:8]}...{token[-4:]}")
    print(f"  Chat ID: {chat_id}")
    return token, chat_id


# ── Raw Telegram send (with full error reporting) ────────────────────

def send_raw(
    token: str,
    chat_id: str,
    text: str,
    parse_mode: str = "HTML",
) -> bool:
    """Send a message via Telegram API and report the result.

    Returns True on success, False on failure.
    """
    url = TELEGRAM_API.format(token=token)
    data = urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": "true",
    }).encode()

    try:
        req = Request(url, data=data, method="POST")
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            msg_id = result["result"]["message_id"]
            print(f"  PASS — message_id: {msg_id}")
            return True
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  FAIL — HTTP {e.code}: {body}")
        return False
    except URLError as e:
        print(f"  FAIL — Network error: {e.reason}")
        return False


# ── Strategy file reader ─────────────────────────────────────────────

def read_atr_stop_mult() -> str:
    """Read current atr_stop_mult from the strategy constructor."""
    if not STRATEGY_FILE.exists():
        return "unknown (file not found)"
    source = STRATEGY_FILE.read_text(encoding="utf-8")
    match = re.search(r"atr_stop_mult:\s*float\s*=\s*([\d.]+)", source)
    if match:
        return match.group(1)
    return "unknown (pattern not found)"


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Telegram Integration Diagnostic")
    print("=" * 60)
    print()

    # Load environment
    load_env()

    # Check credentials
    print("[0/3] Credential Check")
    token, chat_id = check_credentials()
    print()

    results: list[bool] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Test 1: Connection test (raw HTTP) ───────────────────────
    print("[1/3] Connection Test")
    ok = send_raw(
        token, chat_id,
        f"\U0001f517 <b>Connection Test</b>\n\n"
        f"Timestamp: <code>{ts}</code>\n"
        f"Source: <code>test_alerts.py</code>",
    )
    results.append(ok)
    print()
    time.sleep(2)  # rate limit

    # ── Test 2: Mutation accepted (via TelegramAlerter) ──────────
    print("[2/3] Strategy Mutation (TelegramAlerter.mutation_accepted)")
    try:
        from alerts import TelegramAlerter
        alerter = TelegramAlerter()
        alerter.mutation_accepted(
            mutation_name="WIDER_STOPS",
            param_changes={"atr_stop_mult": (3.5, 4.0)},
            old_cagr=0.235,
            new_cagr=0.380,
            old_sharpe=1.138,
            new_sharpe=1.448,
        )
        # Wait for background thread to send
        time.sleep(3)
        print("  PASS — mutation_accepted() queued (check Telegram for formatting)")
        results.append(True)
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        results.append(False)
    print()

    # ── Test 3: Bridge heartbeat with live atr_stop_mult ─────────
    print("[3/3] Bridge Heartbeat")
    atr_val = read_atr_stop_mult()
    print(f"  atr_stop_mult = {atr_val}")
    ok = send_raw(
        token, chat_id,
        f"\U0001f493 <b>Bridge Heartbeat</b>\n\n"
        f"<b>atr_stop_mult:</b> <code>{atr_val}</code>\n"
        f"<b>Strategy:</b> <code>volatility_squeeze.py</code>\n"
        f"<b>Time:</b> <code>{ts}</code>",
    )
    results.append(ok)
    print()

    # ── Summary ──────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    if passed == total:
        print(f"  {passed}/{total} tests passed. Check your Telegram.")
    else:
        print(f"  {passed}/{total} tests passed. Review failures above.")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
