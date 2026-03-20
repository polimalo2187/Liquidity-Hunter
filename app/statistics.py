from datetime import datetime, timedelta
from typing import Dict

from app.database import (
    signal_results_collection,
    signals_collection,
    user_signals_collection,
)
from app.signals import evaluate_expired_signals


# ======================================================
# UTILIDADES DE FECHAS
# ======================================================

def _start_of_day(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day)


def _start_of_week(dt: datetime) -> datetime:
    start = dt - timedelta(days=dt.weekday())
    return datetime(start.year, start.month, start.day)


def _start_of_month(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1)


# ======================================================
# RESET COMPLETO DESDE CERO
# ======================================================

def reset_statistics() -> None:
    """
    Reinicia TODO desde cero:
    - señales base
    - señales usuario
    - resultados evaluados
    """
    signals_collection().delete_many({})
    user_signals_collection().delete_many({})
    signal_results_collection().delete_many({})


# ======================================================
# CURSOR DE RESULTADOS BASE
# ======================================================

def _results_cursor(from_date: datetime, extra_query: Dict = None):
    query = {
        "evaluated_at": {"$gte": from_date},
        "evaluation_scope": "base",
    }

    if extra_query:
        query.update(extra_query)

    return signal_results_collection().find(query)


# ======================================================
# CÁLCULO DE ESTADÍSTICAS
# ======================================================

def _calculate_stats(from_date: datetime) -> Dict:
    evaluate_expired_signals()

    results = _results_cursor(from_date)

    total = 0
    won = 0
    lost = 0
    expired = 0

    for r in results:
        total += 1

        if r["result"] == "won":
            won += 1
        elif r["result"] == "lost":
            lost += 1
        elif r["result"] == "expired":
            expired += 1

    effective_trades = won + lost
    winrate = round((won / effective_trades) * 100, 2) if effective_trades > 0 else 0.0

    return {
        "total": total,
        "won": won,
        "lost": lost,
        "expired": expired,
        "winrate": winrate,
    }


def _calculate_stats_by_plan(from_date: datetime, plan: str) -> Dict:
    evaluate_expired_signals()

    results = _results_cursor(
        from_date,
        {"visibility": plan}
    )

    total = 0
    won = 0
    lost = 0
    expired = 0

    for r in results:
        total += 1

        if r["result"] == "won":
            won += 1
        elif r["result"] == "lost":
            lost += 1
        elif r["result"] == "expired":
            expired += 1

    effective_trades = won + lost
    winrate = round((won / effective_trades) * 100, 2) if effective_trades > 0 else 0.0

    return {
        "total": total,
        "won": won,
        "lost": lost,
        "expired": expired,
        "winrate": winrate,
    }


# ======================================================
# ESTADÍSTICAS GENERALES
# ======================================================

def get_daily_stats() -> Dict:
    now = datetime.utcnow()
    return _calculate_stats(_start_of_day(now))


def get_weekly_stats() -> Dict:
    now = datetime.utcnow()
    return _calculate_stats(_start_of_week(now))


def get_monthly_stats() -> Dict:
    now = datetime.utcnow()
    return _calculate_stats(_start_of_month(now))


# ======================================================
# ESTADÍSTICAS POR DÍAS
# ======================================================

def get_last_days_stats(days: int) -> Dict:
    now = datetime.utcnow()
    from_date = now - timedelta(days=days)
    return _calculate_stats(from_date)


def get_last_days_stats_by_plan(days: int) -> Dict:
    now = datetime.utcnow()
    from_date = now - timedelta(days=days)

    return {
        "free": _calculate_stats_by_plan(from_date, "free"),
        "plus": _calculate_stats_by_plan(from_date, "plus"),
        "premium": _calculate_stats_by_plan(from_date, "premium"),
    }


# ======================================================
# ACTIVIDAD DEL SCANNER
# ======================================================

def get_signal_activity_stats(days: int) -> Dict:
    now = datetime.utcnow()
    from_date = now - timedelta(days=days)

    cur = signals_collection().find({"created_at": {"$gte": from_date}}, {"score": 1})
    total = 0
    score_sum = 0.0
    score_n = 0

    for s in cur:
        total += 1
        sc = s.get("score")
        if isinstance(sc, (int, float)):
            score_sum += float(sc)
            score_n += 1

    avg_score = round(score_sum / score_n, 2) if score_n > 0 else "—"
    return {"signals_total": total, "avg_score": avg_score}


def get_signal_activity_stats_by_plan(days: int) -> Dict:
    now = datetime.utcnow()
    from_date = now - timedelta(days=days)

    out = {}

    for plan in ["free", "plus", "premium"]:
        cur = signals_collection().find(
            {
                "created_at": {"$gte": from_date},
                "visibility": plan,
            },
            {"score": 1}
        )

        total = 0
        score_sum = 0.0
        score_n = 0

        for s in cur:
            total += 1
            sc = s.get("score")
            if isinstance(sc, (int, float)):
                score_sum += float(sc)
                score_n += 1

        avg_score = round(score_sum / score_n, 2) if score_n > 0 else "—"
        out[plan] = {
            "signals_total": total,
            "avg_score": avg_score,
        }

    return out


# ======================================================
# WIN RATE POR SCORE
# ======================================================

def get_winrate_by_score(days: int, buckets=None) -> Dict:
    if buckets is None:
        buckets = [
            (0, 70, "<70"),
            (70, 80, "70–79"),
            (80, 90, "80–89"),
            (90, 101, "90+"),
        ]

    evaluate_expired_signals()

    now = datetime.utcnow()
    from_date = now - timedelta(days=days)

    results = list(_results_cursor(from_date))

    if not results:
        return {"days": days, "buckets": []}

    out = []
    for lo, hi, label in buckets:
        won = 0
        lost = 0
        for r in results:
            sc = r.get("score")
            if sc is None:
                continue
            try:
                sc = float(sc)
            except Exception:
                continue
            if lo <= sc < hi:
                if r.get("result") == "won":
                    won += 1
                elif r.get("result") == "lost":
                    lost += 1
        n = won + lost
        winrate = round((won / n) * 100, 2) if n > 0 else 0.0
        out.append({"label": label, "winrate": winrate, "n": n})

    return {"days": days, "buckets": out}
