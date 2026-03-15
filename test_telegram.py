"""Quick test to verify Telegram alerts are working."""

import os
import time

from dotenv import load_dotenv
load_dotenv()

from alerts import TelegramAlerter

alerter = TelegramAlerter()

if not alerter._enabled:
    print("❌ Telegram is NOT configured.")
    print("   Make sure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in your .env file.")
else:
    print("📤 Sending test message to Telegram...")
    alerter.startup(mode="TEST", symbol="BTCUSDT", exposure=0.5)
    time.sleep(3)
    print("✅ Done! Check your Telegram for the message.")
