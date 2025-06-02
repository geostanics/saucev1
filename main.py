#!/usr/bin/env python3
"""
Binance Futures – RSI + StochRSI Overbought Alert Bot (async version)
========================================================================
This version uses `python-telegram-bot >=20` (asyncio-based) and requires:

    pip install python-telegram-bot requests pandas numpy

Ensure these environment variables are set:
    TELEGRAM_BOT_TOKEN – your bot token from @BotFather
    TELEGRAM_CHAT_ID   – chat/channel/user ID to receive alerts

Trade responsibly – this bot is informational only!
"""

import logging
from dotenv import load_dotenv
import os
import asyncio
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from telegram import Bot
from telegram.constants import ParseMode

# ────────────────────────────── CONFIG ───────────────────────────────────
RSI_PERIOD = 30
STOCHRSI_PERIOD = 30
RSI_THRESHOLD = 60.0
STOCHRSI_THRESHOLD = 0.6

INTERVAL = "15m"
CANDLE_LIMIT = 100

SLEEP_BETWEEN_SYMBOLS = 0.25
SLEEP_BETWEEN_CYCLES = 60

BINANCE_FAPI = "https://fapi.binance.com"


load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CHAT_ID_2 = os.getenv("TELEGRAM_CHAT_ID_2")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("Environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")

bot = Bot(token=BOT_TOKEN)
session = requests.Session()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# ────────────────────────── Helper Functions ────────────────────────────

# def fetch_futures_symbols() -> List[str]:
#     url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
#     data = session.get(url, timeout=10).json()
#     print(data)
#     return [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]

def fetch_futures_symbols() -> List[str]:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return []

    if "symbols" not in data:
        print("Unexpected response structure:", data)
        return []

    return [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]

def fetch_klines(symbol: str, interval: str, limit: int = 150) -> pd.Series:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    raw = session.get(url, params=params, timeout=10).json()
    closes = pd.Series([float(k[4]) for k in raw])
    closes.index = pd.to_datetime([int(k[0]) for k in raw], unit="ms", utc=True)
    return closes

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def stoch_rsi(rsi_series: pd.Series, period: int) -> pd.Series:
    min_rsi = rsi_series.rolling(period).min()
    max_rsi = rsi_series.rolling(period).max()
    return (rsi_series - min_rsi) / (max_rsi - min_rsi)

def analyze_symbol(symbol: str) -> Tuple[float, float]:
    closes = fetch_klines(symbol, INTERVAL, CANDLE_LIMIT)
    rsi_series = rsi(closes, RSI_PERIOD)
    stoch_series = stoch_rsi(rsi_series, STOCHRSI_PERIOD)
    return float(rsi_series.iloc[-1]), float(stoch_series.iloc[-1])

async def send_alert(symbol: str, rsi_val: float, stoch_val: float):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = (
        f"⚠️ *Overbought alert* – `{symbol}`\n"
        f"RSI ({RSI_PERIOD}) = *{rsi_val:.2f}* (> {RSI_THRESHOLD})\n"
        f"StochRSI ({STOCHRSI_PERIOD}) = *{stoch_val:.2f}* (> {STOCHRSI_THRESHOLD})\n"
        f"Interval = {INTERVAL}  •  {timestamp}"
    )
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode=ParseMode.MARKDOWN)
        await bot.send_message(chat_id=CHAT_ID_2, text=text, parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        logging.warning("Telegram error for %s: %s", symbol, exc)

# ─────────────────────────────── Main Loop ──────────────────────────────

async def main():
    symbols = fetch_futures_symbols()
    logging.info("Scanning %d futures pairs every %d s (interval: %s)",
                 len(symbols), SLEEP_BETWEEN_CYCLES, INTERVAL)

    while True:
        cycle_start = asyncio.get_event_loop().time()
        for symbol in symbols:
            try:
                rsi_val, stoch_val = analyze_symbol(symbol)
                if (not np.isnan(rsi_val) and not np.isnan(stoch_val) and
                        rsi_val > RSI_THRESHOLD and stoch_val > STOCHRSI_THRESHOLD):
                    logging.info("Alert – %s | RSI %.2f | StochRSI %.2f", symbol, rsi_val, stoch_val)
                    await send_alert(symbol, rsi_val, stoch_val)
            except Exception as exc:
                logging.warning("%s – error: %s", symbol, exc)
            finally:
                await asyncio.sleep(SLEEP_BETWEEN_SYMBOLS)

        elapsed = asyncio.get_event_loop().time() - cycle_start
        logging.info("Cycle completed in %.1f s – sleeping for %d s", elapsed, SLEEP_BETWEEN_CYCLES)
        await asyncio.sleep(max(1, SLEEP_BETWEEN_CYCLES))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user – exiting…")
