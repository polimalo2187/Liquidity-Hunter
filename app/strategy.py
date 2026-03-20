import pandas as pd
from typing import Optional, Dict, Tuple, List
import ta

# =======================================
# CONFIGURACIÓN BASE
# =======================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

ADX_PERIOD = 14
ATR_PERIOD = 14
BREAKOUT_LOOKBACK = 24

MAX_SCORE = 100

# =======================================
# PERFILES POR PLAN
# Premium NO se toca
# =======================================

PREMIUM_PROFILE = {
    "name": "premium",
    "adx_min": 19.0,
    "atr_pct_min": 0.0028,
    "atr_pct_max": 0.0108,
    "retest_tol_atr": 0.32,
    "min_body_ratio_breakout": 0.38,
    "min_body_ratio_continuation": 0.28,
    "score": 100.0,
    "components": [
        ("trend_structure", 25),
        ("adx_strength", 20),
        ("atr_valid", 15),
        ("breakout_retest", 25),
        ("continuation_candle", 15),
    ],
}

PLUS_PROFILE = {
    "name": "plus",
    "adx_min": 17.0,
    "atr_pct_min": 0.0024,
    "atr_pct_max": 0.0125,
    "retest_tol_atr": 0.48,
    "min_body_ratio_breakout": 0.30,
    "min_body_ratio_continuation": 0.22,
    "score": 86.0,
    "components": [
        ("trend_structure", 24),
        ("adx_strength", 18),
        ("atr_valid", 14),
        ("breakout_retest", 18),
        ("continuation_candle", 12),
    ],
}

FREE_PROFILE = {
    "name": "free",
    "adx_min": 15.0,
    "atr_pct_min": 0.0020,
    "atr_pct_max": 0.0140,
    "retest_tol_atr": 0.62,
    "min_body_ratio_breakout": 0.22,
    "min_body_ratio_continuation": 0.16,
    "score": 78.0,
    "components": [
        ("trend_structure", 22),
        ("adx_strength", 16),
        ("atr_valid", 12),
        ("breakout_retest", 16),
        ("continuation_candle", 12),
    ],
}

PROFILES = [PREMIUM_PROFILE, PLUS_PROFILE, FREE_PROFILE]

# =======================================
# PERFILES DE TRADING POR APALANCAMIENTO
# =======================================

TRADING_PROFILES = {
    "conservador": {
        "leverage": "20x-30x",
        "sl_pct": 0.0080,
        "tp1_pct": 0.0090,
        "tp2_pct": 0.0160,
    },
    "moderado": {
        "leverage": "30x-40x",
        "sl_pct": 0.0068,
        "tp1_pct": 0.0080,
        "tp2_pct": 0.0140,
    },
    "agresivo": {
        "leverage": "40x-50x",
        "sl_pct": 0.0058,
        "tp1_pct": 0.0070,
        "tp2_pct": 0.0120,
    },
}

# =======================================
# INDICADORES
# =======================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema20"] = ta.trend.ema_indicator(df["close"], EMA_FAST)
    df["ema50"] = ta.trend.ema_indicator(df["close"], EMA_MID)
    df["ema200"] = ta.trend.ema_indicator(df["close"], EMA_SLOW)

    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], ADX_PERIOD)

    atr = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], ATR_PERIOD
    )
    df["atr"] = atr
    df["atr_pct"] = df["atr"] / df["close"]

    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, 1e-9)
    df["body_ratio"] = df["body"] / df["range"]

    return df

# =======================================
# BREAKOUT LEVEL
# =======================================

def breakout_level(df: pd.DataFrame, direction: str) -> float:
    ref = df.iloc[-(BREAKOUT_LOOKBACK + 2):-2]

    if direction == "LONG":
        return float(ref["high"].max())

    return float(ref["low"].min())

# =======================================
# HELPERS
# =======================================

def _trend_direction(last: pd.Series) -> Optional[str]:
    if float(last["ema20"]) > float(last["ema50"]) > float(last["ema200"]):
        return "LONG"
    if float(last["ema20"]) < float(last["ema50"]) < float(last["ema200"]):
        return "SHORT"
    return None


def _confirm_breakout_retest(df: pd.DataFrame, direction: str, profile: Dict) -> bool:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    level = breakout_level(df, direction)
    atr = float(last["atr"])

    if atr <= 0:
        return False

    if direction == "LONG":
        breakout = (
            float(prev["close"]) > level
            and float(prev["high"]) > level
            and float(prev["body_ratio"]) >= profile["min_body_ratio_breakout"]
        )

        retest = (
            float(last["low"]) <= level + (atr * profile["retest_tol_atr"])
            and float(last["close"]) >= level
        )

        return breakout and retest

    breakout = (
        float(prev["close"]) < level
        and float(prev["low"]) < level
        and float(prev["body_ratio"]) >= profile["min_body_ratio_breakout"]
    )

    retest = (
        float(last["high"]) >= level - (atr * profile["retest_tol_atr"])
        and float(last["close"]) <= level
    )

    return breakout and retest


def _continuation_ok(last: pd.Series, direction: str, profile: Dict) -> bool:
    if direction == "LONG":
        if float(last["close"]) <= float(last["open"]):
            return False
    else:
        if float(last["close"]) >= float(last["open"]):
            return False

    if float(last["body_ratio"]) < profile["min_body_ratio_continuation"]:
        return False

    return True


def _build_trade_profiles(entry_price: float, direction: str) -> Dict[str, Dict]:
    profiles: Dict[str, Dict] = {}

    for name, cfg in TRADING_PROFILES.items():
        sl_pct = float(cfg["sl_pct"])
        tp1_pct = float(cfg["tp1_pct"])
        tp2_pct = float(cfg["tp2_pct"])

        if direction == "LONG":
            stop_loss = round(entry_price * (1 - sl_pct), 4)
            tp1 = round(entry_price * (1 + tp1_pct), 4)
            tp2 = round(entry_price * (1 + tp2_pct), 4)
        else:
            stop_loss = round(entry_price * (1 + sl_pct), 4)
            tp1 = round(entry_price * (1 - tp1_pct), 4)
            tp2 = round(entry_price * (1 - tp2_pct), 4)

        profiles[name] = {
            "stop_loss": stop_loss,
            "take_profits": [tp1, tp2],
            "leverage": cfg["leverage"],
        }

    return profiles


def _evaluate_profile(df: pd.DataFrame, profile: Dict) -> Optional[Tuple[str, float, float, List, Dict[str, Dict]]]:
    last = df.iloc[-1]

    direction = _trend_direction(last)
    if not direction:
        return None

    if float(last["adx"]) < profile["adx_min"]:
        return None

    atr_pct = float(last["atr_pct"])
    if not (profile["atr_pct_min"] <= atr_pct <= profile["atr_pct_max"]):
        return None

    if not _confirm_breakout_retest(df, direction, profile):
        return None

    if not _continuation_ok(last, direction, profile):
        return None

    level = breakout_level(df, direction)
    # Entrada menos perseguida:
    # en lugar de usar el punto medio exacto entre breakout y cierre,
    # nos quedamos más cerca del nivel de retest para no entrar tan tarde.
    close_price = float(last["close"])
    level_price = float(level)
    entry_price = level_price + ((close_price - level_price) * 0.25)
    trade_profiles = _build_trade_profiles(entry_price, direction)

    return (
        direction,
        entry_price,
        float(profile["score"]),
        list(profile["components"]),
        trade_profiles,
    )

# =======================================
# ESTRATEGIA 5M POR NIVELES
# =======================================

def mtf_strategy(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
) -> Optional[Dict]:
    if len(df_5m) < BREAKOUT_LOOKBACK + 30:
        return None

    df = add_indicators(df_5m)

    if len(df) < BREAKOUT_LOOKBACK + 30:
        return None

    for profile in PROFILES:
        result = _evaluate_profile(df, profile)
        if result:
            direction, entry_price, score, components, trade_profiles = result
            return {
                "direction": direction,
                "entry_price": round(float(entry_price), 4),
                "stop_loss": trade_profiles["conservador"]["stop_loss"],
                "take_profits": list(trade_profiles["conservador"]["take_profits"]),
                "profiles": trade_profiles,
                "score": round(float(score), 2),
                "components": components,
                "timeframes": ["5M"],
            }

    return None
