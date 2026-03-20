from typing import Dict, List, Optional, Tuple

import pandas as pd

# =======================================
# CONFIGURACIÓN BASE — LIQUIDITY SWEEP REVERSAL
# =======================================

ATR_PERIOD = 14
VOLUME_PERIOD = 20
LIQUIDITY_LOOKBACK = 36
TARGET_LOOKBACK = 30
PIVOT_WINDOW = 3
MIN_HISTORY_BARS = max(LIQUIDITY_LOOKBACK + 8, ATR_PERIOD + VOLUME_PERIOD + 8)

# =======================================
# PERFILES POR PLAN
# Premium más estricto; Free más permisivo.
# Los scores respetan los umbrales del scanner.
# =======================================

PREMIUM_PROFILE = {
    "name": "premium",
    "score": 94.0,
    "atr_pct_min": 0.0025,
    "atr_pct_max": 0.0120,
    "liquidity_tolerance_atr": 0.18,
    "min_sweep_atr": 0.18,
    "min_rel_volume": 1.30,
    "min_wick_body_ratio": 1.45,
    "min_wick_range_ratio": 0.42,
    "min_confirm_body_ratio": 0.30,
    "entry_offset_atr": 0.08,
    "sl_buffer_atr": 0.12,
    "min_rr": 1.70,
    "min_pivots": 2,
    "min_sweep_range_atr": 0.85,
    "max_sweep_range_atr": 2.35,
    "max_risk_pct": 0.0090,
    "components": [
        "liquidity_zone",
        "minimum_sweep",
        "recovery_close",
        "relative_volume",
        "confirmation_candle",
        "rr_filter",
    ],
}

PLUS_PROFILE = {
    "name": "plus",
    "score": 86.0,
    "atr_pct_min": 0.0022,
    "atr_pct_max": 0.0132,
    "liquidity_tolerance_atr": 0.22,
    "min_sweep_atr": 0.14,
    "min_rel_volume": 1.15,
    "min_wick_body_ratio": 1.20,
    "min_wick_range_ratio": 0.36,
    "min_confirm_body_ratio": 0.24,
    "entry_offset_atr": 0.10,
    "sl_buffer_atr": 0.14,
    "min_rr": 1.55,
    "min_pivots": 2,
    "min_sweep_range_atr": 0.70,
    "max_sweep_range_atr": 2.70,
    "max_risk_pct": 0.0105,
    "components": [
        "liquidity_zone",
        "minimum_sweep",
        "recovery_close",
        "relative_volume",
        "confirmation_candle",
        "rr_filter",
    ],
}

FREE_PROFILE = {
    "name": "free",
    "score": 78.0,
    "atr_pct_min": 0.0019,
    "atr_pct_max": 0.0145,
    "liquidity_tolerance_atr": 0.28,
    "min_sweep_atr": 0.10,
    "min_rel_volume": 1.00,
    "min_wick_body_ratio": 1.00,
    "min_wick_range_ratio": 0.30,
    "min_confirm_body_ratio": 0.18,
    "entry_offset_atr": 0.12,
    "sl_buffer_atr": 0.16,
    "min_rr": 1.40,
    "min_pivots": 2,
    "min_sweep_range_atr": 0.55,
    "max_sweep_range_atr": 3.00,
    "max_risk_pct": 0.0125,
    "components": [
        "liquidity_zone",
        "minimum_sweep",
        "recovery_close",
        "relative_volume",
        "confirmation_candle",
        "rr_filter",
    ],
}

PROFILES = [PREMIUM_PROFILE, PLUS_PROFILE, FREE_PROFILE]

# =======================================
# PERFILES DE TRADING
# Mismo SL estructural; cambian objetivos y apalancamiento.
# =======================================

TRADING_PROFILES = {
    "conservador": {
        "leverage": "20x-30x",
        "tp1_rr": 1.50,
        "tp2_rr": 1.95,
    },
    "moderado": {
        "leverage": "30x-40x",
        "tp1_rr": 1.75,
        "tp2_rr": 2.35,
    },
    "agresivo": {
        "leverage": "40x-50x",
        "tp1_rr": 2.05,
        "tp2_rr": 2.75,
    },
}


# =======================================
# INDICADORES Y HELPERS DE VELA
# =======================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["close"].shift(1)
    tr_components = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    df["atr"] = tr_components.max(axis=1).rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / df["close"].clip(lower=1e-9)

    df["vol_sma"] = df["volume"].rolling(VOLUME_PERIOD).mean()
    df["rel_volume"] = df["volume"] / df["vol_sma"].clip(lower=1e-9)

    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).clip(lower=1e-9)
    df["body_ratio"] = df["body"] / df["range"]

    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    return df


def _round_price(value: float) -> float:
    return round(float(value), 4)


def _find_pivots(series: pd.Series, mode: str, window: int) -> List[Tuple[int, float]]:
    values = [float(x) for x in series.tolist()]
    pivots: List[Tuple[int, float]] = []

    if len(values) < (window * 2) + 1:
        return pivots

    for idx in range(window, len(values) - window):
        left = values[idx - window:idx]
        center = values[idx]
        right = values[idx + 1:idx + 1 + window]

        if mode == "high":
            if center > max(left) and center >= max(right):
                pivots.append((idx, center))
        else:
            if center < min(left) and center <= min(right):
                pivots.append((idx, center))

    return pivots


def _cluster_pivots(pivots: List[Tuple[int, float]], tolerance: float) -> List[Dict]:
    zones: List[Dict] = []

    for idx, price in sorted(pivots, key=lambda item: item[1]):
        matched = False

        for zone in zones:
            if abs(price - zone["price"]) <= tolerance:
                zone["prices"].append(price)
                zone["indices"].append(idx)
                zone["price"] = sum(zone["prices"]) / len(zone["prices"])
                matched = True
                break

        if not matched:
            zones.append(
                {
                    "price": price,
                    "prices": [price],
                    "indices": [idx],
                }
            )

    for zone in zones:
        zone["count"] = len(zone["prices"])
        zone["latest_index"] = max(zone["indices"])

    return zones


def _select_liquidity_zone(
    historical: pd.DataFrame,
    direction: str,
    sweep_candle: pd.Series,
    profile: Dict,
) -> Optional[Dict]:
    atr = float(sweep_candle["atr"])
    if atr <= 0:
        return None

    tolerance = atr * float(profile["liquidity_tolerance_atr"])
    min_sweep = atr * float(profile["min_sweep_atr"])

    if direction == "SHORT":
        pivots = _find_pivots(historical["high"], "high", PIVOT_WINDOW)
    else:
        pivots = _find_pivots(historical["low"], "low", PIVOT_WINDOW)

    if len(pivots) < int(profile["min_pivots"]):
        return None

    candidates: List[Dict] = []
    for zone in _cluster_pivots(pivots, tolerance):
        if zone["count"] < int(profile["min_pivots"]):
            continue

        zone_price = float(zone["price"])
        if direction == "SHORT":
            sweep_ok = (
                float(sweep_candle["high"]) >= zone_price + min_sweep
                and float(sweep_candle["close"]) < zone_price
            )
            distance = abs(float(sweep_candle["high"]) - zone_price)
        else:
            sweep_ok = (
                float(sweep_candle["low"]) <= zone_price - min_sweep
                and float(sweep_candle["close"]) > zone_price
            )
            distance = abs(zone_price - float(sweep_candle["low"]))

        if not sweep_ok:
            continue

        zone["distance"] = distance
        candidates.append(zone)

    if not candidates:
        return None

    candidates.sort(
        key=lambda zone: (
            -int(zone["count"]),
            float(zone["distance"]),
            -int(zone["latest_index"]),
        )
    )
    return candidates[0]


def _recovery_candle_ok(
    sweep_candle: pd.Series,
    direction: str,
    profile: Dict,
    zone_price: float,
) -> bool:
    body = max(float(sweep_candle["body"]), 1e-9)
    candle_range = float(sweep_candle["range"])

    if direction == "SHORT":
        wick = float(sweep_candle["upper_wick"])
        if float(sweep_candle["close"]) >= zone_price:
            return False
    else:
        wick = float(sweep_candle["lower_wick"])
        if float(sweep_candle["close"]) <= zone_price:
            return False

    if wick < body * float(profile["min_wick_body_ratio"]):
        return False

    if (wick / max(candle_range, 1e-9)) < float(profile["min_wick_range_ratio"]):
        return False

    return True


def _confirmation_candle_ok(confirm_candle: pd.Series, sweep_candle: pd.Series, direction: str, profile: Dict) -> bool:
    if float(confirm_candle["body_ratio"]) < float(profile["min_confirm_body_ratio"]):
        return False

    if direction == "SHORT":
        return (
            float(confirm_candle["close"]) < float(confirm_candle["open"])
            and float(confirm_candle["close"]) < float(sweep_candle["close"])
            and float(confirm_candle["low"]) < float(sweep_candle["low"])
        )

    return (
        float(confirm_candle["close"]) > float(confirm_candle["open"])
        and float(confirm_candle["close"]) > float(sweep_candle["close"])
        and float(confirm_candle["high"]) > float(sweep_candle["high"])
    )


def _room_to_target(entry_price: float, stop_loss: float, structure_target: float, direction: str) -> float:
    risk = abs(stop_loss - entry_price)
    if risk <= 0:
        return 0.0

    if direction == "SHORT":
        room = entry_price - structure_target
    else:
        room = structure_target - entry_price

    return max(0.0, room / risk)


def _tp_from_rr(entry_price: float, risk: float, rr: float, direction: str) -> float:
    if direction == "LONG":
        return entry_price + (risk * rr)
    return entry_price - (risk * rr)


def _build_trade_profiles(
    entry_price: float,
    direction: str,
    stop_loss: float,
    max_room_rr: float,
) -> Dict[str, Dict]:
    risk = abs(stop_loss - entry_price)
    profiles: Dict[str, Dict] = {}

    capped_max_rr = max(1.20, max_room_rr - 0.05)

    for name, cfg in TRADING_PROFILES.items():
        tp1_rr = min(float(cfg["tp1_rr"]), capped_max_rr)
        tp2_rr = min(float(cfg["tp2_rr"]), capped_max_rr)

        if tp2_rr <= tp1_rr:
            tp2_rr = min(capped_max_rr, tp1_rr + 0.25)

        profiles[name] = {
            "stop_loss": _round_price(stop_loss),
            "take_profits": [
                _round_price(_tp_from_rr(entry_price, risk, tp1_rr, direction)),
                _round_price(_tp_from_rr(entry_price, risk, tp2_rr, direction)),
            ],
            "leverage": cfg["leverage"],
        }

    return profiles


def _evaluate_direction(df: pd.DataFrame, direction: str, profile: Dict) -> Optional[Tuple[Dict, Tuple]]:
    sweep_candle = df.iloc[-2]
    confirm_candle = df.iloc[-1]
    historical = df.iloc[:-2].tail(LIQUIDITY_LOOKBACK)

    if len(historical) < LIQUIDITY_LOOKBACK:
        return None

    atr = float(sweep_candle["atr"])
    atr_pct = float(sweep_candle["atr_pct"])
    if atr <= 0 or not (float(profile["atr_pct_min"]) <= atr_pct <= float(profile["atr_pct_max"])):
        return None

    rel_volume = max(float(sweep_candle["rel_volume"]), float(confirm_candle["rel_volume"]))
    if rel_volume < float(profile["min_rel_volume"]):
        return None

    sweep_range_atr = float(sweep_candle["range"]) / atr
    if not (float(profile["min_sweep_range_atr"]) <= sweep_range_atr <= float(profile["max_sweep_range_atr"])):
        return None

    zone = _select_liquidity_zone(historical, direction, sweep_candle, profile)
    if not zone:
        return None

    zone_price = float(zone["price"])

    if not _recovery_candle_ok(sweep_candle, direction, profile, zone_price):
        return None

    if not _confirmation_candle_ok(confirm_candle, sweep_candle, direction, profile):
        return None

    entry_offset = atr * float(profile["entry_offset_atr"])
    if direction == "SHORT":
        entry_price = zone_price - entry_offset
        stop_loss = float(sweep_candle["high"]) + (atr * float(profile["sl_buffer_atr"]))
        structure_target = float(historical.tail(TARGET_LOOKBACK)["low"].min())
    else:
        entry_price = zone_price + entry_offset
        stop_loss = float(sweep_candle["low"]) - (atr * float(profile["sl_buffer_atr"]))
        structure_target = float(historical.tail(TARGET_LOOKBACK)["high"].max())

    risk = abs(stop_loss - entry_price)
    if risk <= 0:
        return None

    risk_pct = risk / max(entry_price, 1e-9)
    if risk_pct > float(profile["max_risk_pct"]):
        return None

    room_rr = _room_to_target(entry_price, stop_loss, structure_target, direction)
    if room_rr < float(profile["min_rr"]):
        return None

    trade_profiles = _build_trade_profiles(entry_price, direction, stop_loss, room_rr)
    result = {
        "direction": direction,
        "entry_price": _round_price(entry_price),
        "stop_loss": trade_profiles["conservador"]["stop_loss"],
        "take_profits": list(trade_profiles["conservador"]["take_profits"]),
        "profiles": trade_profiles,
        "score": round(float(profile["score"]), 2),
        "components": list(profile["components"]),
        "timeframes": ["15M"],
        "atr_pct": round(atr_pct, 6),
    }

    ranking = (
        int(zone["count"]),
        round(room_rr, 4),
        round(rel_volume, 4),
    )
    return result, ranking


def _evaluate_profile(df: pd.DataFrame, profile: Dict) -> Optional[Dict]:
    best_result: Optional[Dict] = None
    best_rank: Optional[Tuple] = None

    for direction in ("SHORT", "LONG"):
        evaluated = _evaluate_direction(df, direction, profile)
        if not evaluated:
            continue

        result, rank = evaluated
        if best_rank is None or rank > best_rank:
            best_result = result
            best_rank = rank

    return best_result


# =======================================
# ESTRATEGIA PRINCIPAL
# =======================================

def liquidity_sweep_reversal_strategy(df_15m: pd.DataFrame) -> Optional[Dict]:
    if len(df_15m) < MIN_HISTORY_BARS:
        return None

    df = add_indicators(df_15m)
    if len(df) < MIN_HISTORY_BARS:
        return None

    if df[["atr", "atr_pct", "rel_volume", "body_ratio"]].tail(5).isnull().any().any():
        return None

    for profile in PROFILES:
        result = _evaluate_profile(df, profile)
        if result:
            return result

    return None


# =======================================
# COMPATIBILIDAD HACIA ATRÁS
# El scanner viejo sigue llamando mtf_strategy().
# Internamente ya no usa MTF; opera solo la lógica de barrida de liquidez en 15M.
# =======================================

def mtf_strategy(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
) -> Optional[Dict]:
    _ = df_1h, df_5m
    return liquidity_sweep_reversal_strategy(df_15m)
