# app/scanner.py

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict

import requests
import pandas as pd
from telegram import Bot

from app.strategy import mtf_strategy
from app.signals import create_base_signal, telegram_signal_blocked
from app.plans import PLAN_FREE, PLAN_PLUS, PLAN_PREMIUM
from app.notifier import notify_new_signal_alert
from app.database import signals_collection

logger = logging.getLogger(__name__)

BINANCE_FUTURES_API = "https://fapi.binance.com"

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
MIN_QUOTE_VOLUME = int(os.getenv("MIN_QUOTE_VOLUME", "20000000"))
DEDUP_MINUTES = int(os.getenv("DEDUP_MINUTES", "10"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

PREMIUM_SCORE_MIN = float(os.getenv("PREMIUM_SCORE_MIN", "90"))
PLUS_SCORE_MIN = float(os.getenv("PLUS_SCORE_MIN", "82"))
FREE_SCORE_MIN = float(os.getenv("FREE_SCORE_MIN", "76"))

class RateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self.last_request = 0.0

    def wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()

rate_limiter = RateLimiter(REQUEST_DELAY)

def get_klines(symbol: str, interval: str, limit: int = 220) -> pd.DataFrame:
    rate_limiter.wait()
    url = f"{BINANCE_FUTURES_API}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    df = pd.DataFrame(
        response.json(),
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ],
    )
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def get_active_futures_symbols() -> List[str]:
    rate_limiter.wait()
    url = f"{BINANCE_FUTURES_API}/fapi/v1/ticker/24hr"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    symbols = [
        item["symbol"]
        for item in response.json()
        if item["symbol"].endswith("USDT")
        and float(item["quoteVolume"]) >= MIN_QUOTE_VOLUME
    ]
    logger.info(f"📊 {len(symbols)} símbolos activos con volumen suficiente")
    return symbols

def recent_duplicate_exists(symbol: str, direction: str, visibility: str) -> bool:
    since = datetime.utcnow() - timedelta(minutes=DEDUP_MINUTES)
    exists = signals_collection().find_one(
        {
            "symbol": symbol,
            "direction": direction,
            "visibility": visibility,
            "created_at": {"$gte": since},
        }
    ) is not None

    if exists:
        logger.info(f"♻️ Duplicado reciente detectado: {symbol} {direction} ({visibility})")
    return exists

def _classify_plan_by_score(score: float):
    if score >= PREMIUM_SCORE_MIN:
        return PLAN_PREMIUM, "🥇 ORO"
    elif score >= PLUS_SCORE_MIN:
        return PLAN_PLUS, "🥈 PLATA"
    elif score >= FREE_SCORE_MIN:
        return PLAN_FREE, "🥉 BRONCE"
    return None, None

async def scan_market_async(bot: Bot):
    logger.info("📡 Scanner iniciado — señales SOLO por calidad")

    while True:
        try:
            if telegram_signal_blocked():
                logger.info("⏳ Señales aún vigentes en Telegram. Escaneo pausado.")
                await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                continue

            symbols = get_active_futures_symbols()
            candidates: List[Dict] = []

            for symbol in symbols:
                try:
                    df_1h = get_klines(symbol, "1h")
                    df_15m = get_klines(symbol, "15m")
                    df_5m = get_klines(symbol, "5m")

                    result = mtf_strategy(df_1h, df_15m, df_5m)
                    if result:
                        result["symbol"] = symbol
                        candidates.append(result)

                    await asyncio.sleep(0.05)

                except Exception as e:
                    logger.debug(f"⚠️ Error procesando {symbol}: {e}")

            if not candidates:
                logger.info("📭 No hay oportunidades fuertes en este ciclo")
                await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                continue

            candidates.sort(
                key=lambda x: (x.get("score", 0), x["symbol"]),
                reverse=True,
            )

            best_by_plan = {
                PLAN_PREMIUM: None,
                PLAN_PLUS: None,
                PLAN_FREE: None,
            }

            for signal in candidates:
                base_score = float(signal.get("score", 0))
                visibility, medal = _classify_plan_by_score(base_score)

                if not visibility:
                    continue

                if best_by_plan[visibility] is None:
                    signal["_visibility"] = visibility
                    signal["_medal"] = medal
                    best_by_plan[visibility] = signal

                if (
                    best_by_plan[PLAN_PREMIUM] is not None
                    and best_by_plan[PLAN_PLUS] is not None
                    and best_by_plan[PLAN_FREE] is not None
                ):
                    break

            selected = [
                best_by_plan[PLAN_PREMIUM],
                best_by_plan[PLAN_PLUS],
                best_by_plan[PLAN_FREE],
            ]

            for signal in selected:
                if not signal:
                    continue

                symbol = signal["symbol"]
                direction = signal["direction"]
                entry_price = float(signal["entry_price"])
                base_score = float(signal.get("score", 0))
                visibility = signal["_visibility"]
                medal = signal["_medal"]

                if recent_duplicate_exists(symbol, direction, visibility):
                    continue

                base_signal = create_base_signal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=float(signal["stop_loss"]),
                    take_profits=list(signal["take_profits"]),
                    timeframes=list(signal.get("timeframes", ["5M"])),
                    visibility=visibility,
                    score=base_score,
                    components=signal.get("components", []),
                    profiles=signal.get("profiles"),
                )

                if not base_signal:
                    continue

                try:
                    await notify_new_signal_alert(
                        bot,
                        visibility,
                        base_signal=base_signal
                    )
                except Exception as e:
                    logger.error(f"⚠️ Error notificando señal: {e}")

                logger.info(
                    "✅ %s | %s %s | base_score=%s | plan=%s",
                    medal,
                    symbol,
                    direction,
                    base_score,
                    visibility,
                )

            await asyncio.sleep(SCAN_INTERVAL_SECONDS)

        except Exception:
            logger.error("❌ Error crítico en scanner", exc_info=True)
            await asyncio.sleep(60)

def scan_market(bot: Bot):
    logger.info("🚀 Iniciando scanner en thread separado")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(scan_market_async(bot))
