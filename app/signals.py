# app/signals.py

import os
import time
import logging
import secrets
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Optional

import requests
import pytz

from app.models import new_signal
from app.plans import PLAN_FREE, PLAN_PLUS, PLAN_PREMIUM, allowed_visibilities_for_plan
from app.config import is_admin
from app.database import (
    signals_collection,
    user_signals_collection,
    users_collection,
    signal_results_collection,
)

logger = logging.getLogger(__name__)

# ======================================================
# CONFIGURACIÓN GLOBAL
# ======================================================
BINANCE_FUTURES_API = os.getenv("BINANCE_FUTURES_API", "https://fapi.binance.com")
MAX_SIGNALS_PER_QUERY = int(os.getenv("MAX_SIGNALS_PER_QUERY", "10"))
BINANCE_MAX_RETRIES = int(os.getenv("BINANCE_MAX_RETRIES", "3"))
BINANCE_RETRY_DELAY = float(os.getenv("BINANCE_RETRY_DELAY", "1.0"))
USER_TIMEZONE = os.getenv("USER_TIMEZONE", "America/Havana")

LEVERAGE_PROFILES = {
    "conservador": "20x-30x",
    "moderado": "30x-40x",
    "agresivo": "40x-50x",
}

TIMEFRAME_TO_MINUTES = {
    "5M": 5,
    "15M": 15,
    "1H": 60,
}

DEDUP_MINUTES = int(os.getenv("DEDUP_MINUTES", "10"))
TELEGRAM_SIGNAL_COOLDOWN_MINUTES = 15
MIN_SIGNAL_VALIDITY_MINUTES = int(os.getenv("MIN_SIGNAL_VALIDITY_MINUTES", "15"))
MAX_SIGNAL_VALIDITY_MINUTES = int(os.getenv("MAX_SIGNAL_VALIDITY_MINUTES", "45"))


def _normalize_price(value: float, decimals: int = 8) -> float:
    quant = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(float(value))).quantize(quant, rounding=ROUND_HALF_UP))


def _infer_price_decimals(price_hint: Optional[float]) -> int:
    try:
        price = abs(float(price_hint or 0))
    except Exception:
        price = 0.0

    if price >= 1000:
        return 2
    if price >= 100:
        return 3
    if price >= 1:
        return 4
    if price >= 0.1:
        return 5
    if price >= 0.01:
        return 6
    if price >= 0.001:
        return 6
    if price >= 0.0001:
        return 7
    return 8


def _tick_size_to_decimals(tick_size: str) -> Optional[int]:
    try:
        normalized = tick_size.rstrip("0")
        if "." not in normalized:
            return 0
        return len(normalized.split(".", 1)[1])
    except Exception:
        return None


@lru_cache(maxsize=1)
def _futures_price_precisions() -> Dict[str, int]:
    url = f"{BINANCE_FUTURES_API}/fapi/v1/exchangeInfo"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    precisions: Dict[str, int] = {}
    payload = response.json()
    for symbol_info in payload.get("symbols", []):
        symbol = symbol_info.get("symbol")
        if not symbol:
            continue

        precision = None
        for flt in symbol_info.get("filters", []):
            if flt.get("filterType") == "PRICE_FILTER":
                precision = _tick_size_to_decimals(str(flt.get("tickSize", "")))
                break

        if precision is None:
            try:
                precision = int(symbol_info.get("pricePrecision"))
            except Exception:
                precision = None

        if precision is not None:
            precisions[symbol] = max(0, min(8, int(precision)))

    return precisions


def get_symbol_price_precision(symbol: str, price_hint: Optional[float] = None) -> int:
    try:
        cached = _futures_price_precisions()
        if symbol in cached:
            return cached[symbol]
    except Exception as e:
        logger.debug(f"No se pudo obtener precision de Binance para {symbol}: {e}")

    return _infer_price_decimals(price_hint)


def format_price(symbol: str, value: float, *, price_hint: Optional[float] = None, precision: Optional[int] = None) -> str:
    try:
        if precision is None:
            precision = get_symbol_price_precision(symbol, price_hint if price_hint is not None else value)
        normalized = _normalize_price(float(value), decimals=max(precision, 2))
        return f"{normalized:.{precision}f}"
    except (InvalidOperation, ValueError, TypeError):
        return str(value)


# ======================================================
# UTILIDADES
# ======================================================

def _base_validity_by_plan(visibility: str) -> int:
    plan = str(visibility or "").lower()
    if plan == "premium":
        return 25
    if plan == "plus":
        return 18
    return 12


def calculate_signal_validity(
    timeframes: List[str],
    *,
    visibility: str = "",
    score: Optional[float] = None,
    entry_price: Optional[float] = None,
    current_price: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> int:
    """
    Validez dinámica:
    - plan: base distinta por nivel
    - distancia del precio actual a la entrada
    - score: mejor setup tolera un poco más de tiempo
    - ATR%: mercado muy rápido = menos tiempo; lento = más tiempo
    """
    validity = float(_base_validity_by_plan(visibility))

    minutes = [TIMEFRAME_TO_MINUTES.get(tf.upper(), 0) for tf in timeframes]
    tf_hint = max(minutes) if minutes else 5
    if tf_hint > 5:
        validity += min(10.0, (tf_hint - 5) * 0.15)

    if (
        entry_price is not None
        and current_price is not None
        and float(entry_price) > 0
        and float(current_price) > 0
    ):
        distance_pct = abs(float(current_price) - float(entry_price)) / float(current_price)

        if distance_pct >= 0.006:
            validity += 8
        elif distance_pct >= 0.004:
            validity += 6
        elif distance_pct >= 0.0025:
            validity += 4
        elif distance_pct >= 0.0015:
            validity += 2
        else:
            validity += 0

    if score is not None:
        try:
            score = float(score)
            if score >= 95:
                validity += 6
            elif score >= 90:
                validity += 5
            elif score >= 82:
                validity += 3
            elif score >= 76:
                validity += 2
        except Exception:
            pass

    if atr_pct is not None:
        try:
            atr_pct = float(atr_pct)
            if atr_pct >= 0.010:
                validity -= 4
            elif atr_pct >= 0.008:
                validity -= 3
            elif atr_pct <= 0.003:
                validity += 3
            elif atr_pct <= 0.004:
                validity += 2
        except Exception:
            pass

    validity = max(MIN_SIGNAL_VALIDITY_MINUTES, min(MAX_SIGNAL_VALIDITY_MINUTES, int(round(validity))))
    return validity


def calculate_entry_zone(entry: float, pct: float = 0.0015):
    low = _normalize_price(entry * (1 - pct))
    high = _normalize_price(entry * (1 + pct))
    return low, high


def get_current_price(symbol: str) -> float:
    url = f"{BINANCE_FUTURES_API}/fapi/v1/ticker/price"
    for attempt in range(BINANCE_MAX_RETRIES):
        try:
            r = requests.get(url, params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:
            if attempt == BINANCE_MAX_RETRIES - 1:
                raise
            time.sleep(BINANCE_RETRY_DELAY)


def estimate_minutes_to_entry(symbol: str, entry_zone: Dict[str, float], timeframes: List[str]) -> Dict[str, int]:
    try:
        current_price = get_current_price(symbol)
        zone_mid = (entry_zone["low"] + entry_zone["high"]) / 2

        if entry_zone["low"] <= current_price <= entry_zone["high"]:
            return {"min": 1, "max": 5}

        distance_pct = abs(current_price - zone_mid) / current_price

        if "5M" in timeframes:
            speed = 0.004
            base_tf = 5
        elif "15M" in timeframes:
            speed = 0.0025
            base_tf = 15
        else:
            speed = 0.0015
            base_tf = calculate_signal_validity(timeframes)

        candles_needed = max(1, distance_pct / speed)
        minutes_estimated = candles_needed * base_tf

        return {
            "min": max(1, int(minutes_estimated * 0.6)),
            "max": int(minutes_estimated * 1.4),
        }
    except Exception as e:
        logger.warning(f"Fallback estimate_minutes_to_entry: {e}")
        base = calculate_signal_validity(timeframes)
        return {"min": max(1, int(base * 0.5)), "max": int(base * 1.5)}


def recent_duplicate_exists(symbol: str, direction: str, visibility: str) -> bool:
    since = datetime.utcnow() - timedelta(minutes=DEDUP_MINUTES)
    return signals_collection().find_one({
        "symbol": symbol,
        "direction": direction,
        "visibility": visibility,
        "created_at": {"$gte": since},
    }) is not None


def telegram_signal_blocked(symbol: Optional[str] = None) -> bool:
    since = datetime.utcnow() - timedelta(minutes=TELEGRAM_SIGNAL_COOLDOWN_MINUTES)
    query = {"created_at": {"$gte": since}}
    if symbol:
        query["symbol"] = symbol
    return signals_collection().find_one(query, sort=[("created_at", -1)]) is not None

# ======================================================
# GENERAR SEÑALES POR PLAN
# ======================================================

def generate_user_signal_for_plan(base_signal: Dict):
    visibility = str(base_signal.get("visibility", PLAN_FREE)).lower()
    now = datetime.utcnow()

    for user in users_collection().find({}):
        user_id = user.get("user_id")
        user_plan = user.get("plan", PLAN_FREE)
        plan_end = user.get("plan_end")
        admin = is_admin(user_id)

        if plan_end and plan_end < now and not admin:
            continue

        allowed_visibilities = allowed_visibilities_for_plan(user_plan, admin=admin)
        if visibility not in allowed_visibilities:
            continue

        existing = user_signals_collection().find_one({
            "user_id": user_id,
            "symbol": base_signal["symbol"],
            "valid_until": {"$gt": now}
        })
        if existing:
            continue

        generate_user_signal(base_signal, user_id)

# ======================================================
# CREAR SEÑAL BASE
# ======================================================

def create_base_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profits: List[float],
    timeframes: List[str],
    visibility: str,
    score: Optional[float] = None,
    components: Optional[List[str]] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    atr_pct: Optional[float] = None,
) -> Dict:

    if telegram_signal_blocked(symbol):
        logger.info(f"⏳ Bloqueo activo para {symbol}, no se crea nueva señal")
        return {}

    zone_low, zone_high = calculate_entry_zone(entry_price)
    estimated_minutes = estimate_minutes_to_entry(symbol, {"low": zone_low, "high": zone_high}, timeframes)

    try:
        current_price = get_current_price(symbol)
    except Exception as e:
        logger.warning(f"Fallback current_price en create_base_signal: {e}")
        current_price = entry_price

    price_precision = get_symbol_price_precision(symbol, entry_price)

    signal = new_signal(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profits=take_profits,
        timeframes=timeframes,
        visibility=visibility,
        leverage=LEVERAGE_PROFILES,
        components=components,
        score=score
    )

    if profiles:
        signal["profiles"] = profiles
    signal["leverage_profiles"] = LEVERAGE_PROFILES
    signal["price_precision"] = price_precision

    now = datetime.utcnow()
    validity_minutes = calculate_signal_validity(
        timeframes,
        visibility=visibility,
        score=score,
        entry_price=entry_price,
        current_price=current_price,
        atr_pct=atr_pct,
    )
    valid_until = now + timedelta(minutes=validity_minutes)
    telegram_valid_until = now + timedelta(minutes=TELEGRAM_SIGNAL_COOLDOWN_MINUTES)

    inserted_id = signals_collection().insert_one(signal).inserted_id

    signals_collection().update_one(
        {"_id": inserted_id},
        {"$set": {
            "created_at": now,
            "valid_until": valid_until,
            "telegram_valid_until": telegram_valid_until,
            "entry_zone": {"low": zone_low, "high": zone_high},
            "estimated_entry_minutes": estimated_minutes,
            "profiles": profiles if profiles else signal.get("profiles"),
            "leverage_profiles": LEVERAGE_PROFILES,
            "validity_minutes": validity_minutes,
            "signal_market_price": current_price,
            "signal_atr_pct": atr_pct,
            "price_precision": price_precision,
            "evaluated": False,
        }}
    )

    signal["evaluated"] = False
    signal["_id"] = inserted_id
    signal["created_at"] = now
    signal["valid_until"] = valid_until
    signal["telegram_valid_until"] = telegram_valid_until
    signal["entry_zone"] = {"low": zone_low, "high": zone_high}
    signal["estimated_entry_minutes"] = estimated_minutes
    signal["validity_minutes"] = validity_minutes
    signal["signal_market_price"] = current_price
    signal["signal_atr_pct"] = atr_pct
    signal["price_precision"] = price_precision

    generate_user_signal_for_plan(signal)
    return signal

# ======================================================
# GENERAR SEÑAL USUARIO
# ======================================================

def _fallback_profiles(direction: str, entry: float) -> Dict[str, Dict]:
    if direction == "LONG":
        return {
            "conservador": {
                "stop_loss": _normalize_price(entry * 0.992),
                "take_profits": [_normalize_price(entry * 1.009), _normalize_price(entry * 1.016)],
            },
            "moderado": {
                "stop_loss": _normalize_price(entry * 0.9932),
                "take_profits": [_normalize_price(entry * 1.008), _normalize_price(entry * 1.014)],
            },
            "agresivo": {
                "stop_loss": _normalize_price(entry * 0.9942),
                "take_profits": [_normalize_price(entry * 1.007), _normalize_price(entry * 1.012)],
            },
        }

    return {
        "conservador": {
            "stop_loss": _normalize_price(entry * 1.008),
            "take_profits": [_normalize_price(entry * 0.991), _normalize_price(entry * 0.984)],
        },
        "moderado": {
            "stop_loss": _normalize_price(entry * 1.0068),
            "take_profits": [_normalize_price(entry * 0.992), _normalize_price(entry * 0.986)],
        },
        "agresivo": {
            "stop_loss": _normalize_price(entry * 1.0058),
            "take_profits": [_normalize_price(entry * 0.993), _normalize_price(entry * 0.988)],
        },
    }


def generate_user_signal(base_signal: Dict, user_id: int) -> Dict:
    existing = user_signals_collection().find_one({
        "user_id": user_id,
        "symbol": base_signal["symbol"],
        "valid_until": {"$gt": datetime.utcnow()}
    })
    if existing:
        return existing

    direction = str(base_signal["direction"]).upper()
    symbol = base_signal["symbol"]
    entry = float(base_signal["entry_price"])
    price_precision = int(base_signal.get("price_precision") or get_symbol_price_precision(symbol, entry))
    profiles = base_signal.get("profiles") or _fallback_profiles(direction, entry)

    normalized_profiles = {}
    for profile_name in ["conservador", "moderado", "agresivo"]:
        src = profiles.get(profile_name, {})
        raw_tps = src.get("take_profits", [entry, entry])
        normalized_profiles[profile_name] = {
            "stop_loss": _normalize_price(float(src.get("stop_loss", entry))),
            "take_profits": [
                _normalize_price(float(raw_tps[0])),
                _normalize_price(float(raw_tps[1])),
            ],
            "leverage": LEVERAGE_PROFILES[profile_name],
        }

    user_signal = {
        "user_id": user_id,
        "signal_id": str(base_signal["_id"]),
        "symbol": symbol,
        "direction": direction,
        "entry_price": _normalize_price(entry),
        "entry_zone": dict(zip(["low", "high"], calculate_entry_zone(entry))),
        "profiles": normalized_profiles,
        "leverage_profiles": LEVERAGE_PROFILES,
        "timeframes": base_signal["timeframes"],
        "created_at": datetime.utcnow(),
        "valid_until": base_signal["valid_until"],
        "telegram_valid_until": base_signal["telegram_valid_until"],
        "fingerprint": secrets.token_hex(4),
        "visibility": base_signal["visibility"],
        "score": base_signal.get("score"),
        "price_precision": price_precision,
        "evaluated": False,
    }

    user_signals_collection().insert_one(user_signal)
    return user_signal

# ======================================================
# FORMATO TELEGRAM
# ======================================================

def format_user_signal(user_signal: Dict) -> str:
    tz = pytz.timezone(USER_TIMEZONE)
    start = user_signal["created_at"].astimezone(tz).strftime("%H:%M")
    end = user_signal["telegram_valid_until"].astimezone(tz).strftime("%H:%M")

    symbol = user_signal["symbol"]
    precision = int(user_signal.get("price_precision") or get_symbol_price_precision(symbol, user_signal.get("entry_price")))
    entry_text = format_price(symbol, user_signal["entry_price"], precision=precision)

    text = (
        f"📊 NUEVA SEÑAL – FUTUROS USDT\n\n"
        f"🏷️ PLAN: {user_signal['visibility'].upper()}\n\n"
        f"Par: {symbol}\n"
        f"Dirección: {user_signal['direction']}\n"
        f"Entrada base: {entry_text}\n\n"
        f"Margen: ISOLATED\n"
        f"Timeframes: {' / '.join(user_signal['timeframes'])}\n\n"
    )

    for profile in ["conservador", "moderado", "agresivo"]:
        p = user_signal["profiles"][profile]
        sl_text = format_price(symbol, p['stop_loss'], precision=precision)
        tp1_text = format_price(symbol, p['take_profits'][0], precision=precision)
        tp2_text = format_price(symbol, p['take_profits'][1], precision=precision)
        text += (
            "━━━━━━━━━━━━━━━━━━\n"
            f"{profile.upper()}\n"
            f"SL: {sl_text}\n"
            f"TP1: {tp1_text}\n"
            f"TP2: {tp2_text}\n"
            f"Apalancamiento: {LEVERAGE_PROFILES[profile]}\n\n"
        )

    text += f"⏳ Activa: {start} → {end}\n"
    text += f"🔐 ID: {user_signal['fingerprint']}\n"
    return text

# ======================================================
# OBTENER SEÑALES USUARIO
# ======================================================

def get_latest_base_signal_for_plan(user_id: int, visibility: Optional[str] = None):
    now = datetime.utcnow()
    query = {
        "user_id": user_id,
        "valid_until": {"$gt": now},
    }

    if visibility:
        query["visibility"] = str(visibility).lower()

    return list(
        user_signals_collection()
        .find(query)
        .sort("created_at", -1)
        .limit(MAX_SIGNALS_PER_QUERY)
    )

# ======================================================
# EVALUACIÓN AUTOMÁTICA DE SEÑALES (PERFIL CONSERVADOR)
# ======================================================

def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _fetch_klines_between(symbol: str, start_dt: datetime, end_dt: datetime, interval: str = "1m") -> List[List]:
    url = f"{BINANCE_FUTURES_API}/fapi/v1/klines"
    start_ms = _dt_to_ms(start_dt)
    end_ms = _dt_to_ms(end_dt)
    all_rows: List[List] = []

    while start_ms < end_ms:
        r = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break

        all_rows.extend(rows)
        last_open_ms = int(rows[-1][0])
        next_start = last_open_ms + 60_000
        if next_start <= start_ms:
            break
        start_ms = next_start

        if len(rows) < 1000:
            break

    return all_rows


def _evaluate_signal_result(signal_doc: Dict) -> str:
    direction = str(signal_doc.get("direction", "")).upper()
    symbol = signal_doc.get("symbol")

    stop_loss = signal_doc.get("stop_loss")
    take_profits = signal_doc.get("take_profits", [])

    # Fallback por compatibilidad si existe estructura por perfiles
    if (stop_loss is None or not take_profits) and signal_doc.get("profiles"):
        conservador = signal_doc.get("profiles", {}).get("conservador", {})
        stop_loss = conservador.get("stop_loss")
        take_profits = conservador.get("take_profits", [])

    tp1 = take_profits[0] if take_profits else None
    created_at = signal_doc.get("created_at")
    valid_until = signal_doc.get("valid_until")

    if not symbol or not direction or stop_loss is None or tp1 is None or not created_at or not valid_until:
        return "expired"

    try:
        stop_loss = float(stop_loss)
        tp1 = float(tp1)
    except Exception:
        return "expired"

    try:
        klines = _fetch_klines_between(symbol, created_at, valid_until, interval="1m")
    except Exception as e:
        logger.error(f"❌ Error descargando velas para evaluar {symbol}: {e}")
        return "expired"

    for row in klines:
        try:
            high = float(row[2])
            low = float(row[3])
        except Exception:
            continue

        if direction == "LONG":
            if low <= stop_loss and high >= tp1:
                return "lost"
            if high >= tp1:
                return "won"
            if low <= stop_loss:
                return "lost"

        elif direction == "SHORT":
            if high >= stop_loss and low <= tp1:
                return "lost"
            if low <= tp1:
                return "won"
            if high >= stop_loss:
                return "lost"

    return "expired"


def evaluate_expired_signals(limit: int = 100) -> int:
    """
    Evalúa SOLO señales base para que:
    - estadísticas coincidan con scanner
    - no se dupliquen resultados por usuario
    """
    now = datetime.utcnow()
    pending = list(
        signals_collection()
        .find({
            "valid_until": {"$lte": now},
            "evaluated": {"$ne": True},
        })
        .sort("valid_until", 1)
        .limit(limit)
    )

    processed = 0

    for s in pending:
        try:
            result = _evaluate_signal_result(s)
            evaluated_at = datetime.utcnow()

            result_doc = {
                "base_signal_id": str(s.get("_id")),
                "signal_id": str(s.get("_id")),
                "user_id": None,
                "symbol": s.get("symbol"),
                "direction": s.get("direction"),
                "visibility": s.get("visibility"),
                "plan": s.get("visibility"),
                "score": s.get("score"),
                "result": result,
                "evaluated_at": evaluated_at,
                "evaluated_profile": "conservador",
                "evaluation_scope": "base",
                "tp_used": (s.get("take_profits") or [None])[0],
                "sl_used": s.get("stop_loss"),
                "signal_created_at": s.get("created_at"),
                "signal_valid_until": s.get("valid_until"),
            }

            signal_results_collection().insert_one(result_doc)

            signals_collection().update_one(
                {"_id": s["_id"]},
                {
                    "$set": {
                        "evaluated": True,
                        "result": result,
                        "evaluated_at": evaluated_at,
                        "evaluated_profile": "conservador",
                    }
                }
            )

            processed += 1

        except Exception as e:
            logger.error(f"❌ Error evaluando señal base {s.get('symbol')}: {e}", exc_info=True)

    if processed:
        logger.info(f"✅ Señales base evaluadas automáticamente: {processed}")

    return processed

