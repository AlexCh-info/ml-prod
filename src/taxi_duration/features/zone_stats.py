"""Признаки зоны посадки за прошлые дни без утечки будущего (спека zone-features).

Строка ``(pu_zone, event_timestamp = D 00:00)`` содержит агрегаты по поездкам
дней ``[D - window_days, D - 1]``. День D в окно не входит: в момент D 00:00 о поездках
дня D ещё ничего не известно. ``event_timestamp`` — момент, когда значение становится
доступно; по нему Feast делает point-in-time join (этап 05).

Как считается:

1. дневные суммы по ``(pu_zone, день)``: число поездок, сумма длительностей и расстояний;
2. полная сетка «зона × каждый день» с нулями, чтобы ``rolling(window)`` по строкам
   означал ровно ``window`` календарных дней, даже если в какие-то дни поездок не было;
3. скользящая сумма за ``window`` дней и ``shift(1)``: строка дня D видит только прошлое;
4. средние = сумма / число поездок за окно (взвешенно по поездкам); без поездок — NaN.

Сетка идёт от первого дня данных до дня, следующего за последним: последняя строка
даёт признаки «на завтра» — то, что нужно сервису в проде.
"""

from __future__ import annotations

import pandas as pd

from taxi_duration.errors import DataError

TRIPS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "pickup_datetime",
    "pu_zone",
    "duration_min",
    "trip_distance",
)


def zone_stats_schema(window_days: int) -> dict[str, str]:
    """Схема результата: колонки и dtype. Суффикс ``_{window_days}d`` — часть имени признака."""
    suffix = f"{window_days}d"
    return {
        "pu_zone": "int32",
        "event_timestamp": "datetime64[us]",
        f"zone_trips_{suffix}": "int32",
        f"zone_mean_duration_{suffix}": "float64",
        f"zone_mean_distance_{suffix}": "float64",
    }


def build_zone_stats(trips: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Посчитать признаки зон по подготовленным поездкам (одного или нескольких месяцев).

    Args:
        trips: результат ``prepare_trips`` (нужны ``TRIPS_REQUIRED_COLUMNS``).
        window_days: длина окна в днях.

    Returns:
        Таблица со схемой ``zone_stats_schema(window_days)``, отсортированная по
        ``pu_zone, event_timestamp``, с ``RangeIndex``. Пара ``(pu_zone, event_timestamp)``
        уникальна.

    Raises:
        DataError: нет обязательных колонок или ``window_days < 1``.
    """
    if window_days < 1:
        raise DataError(f"window_days must be >= 1, got {window_days}")
    missing = [col for col in TRIPS_REQUIRED_COLUMNS if col not in trips.columns]
    if missing:
        raise DataError(f"Missing columns for zone stats: {', '.join(missing)}")

    schema = zone_stats_schema(window_days)
    if trips.empty:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})

    day = trips["pickup_datetime"].dt.floor("D")
    daily = (
        trips.assign(day=day)
        .groupby(["pu_zone", "day"])
        .agg(
            trips=("duration_min", "size"),
            duration_sum=("duration_min", "sum"),
            distance_sum=("trip_distance", "sum"),
        )
    )

    # Полная сетка: каждый день от первого до (последний + 1) для каждой зоны.
    days = pd.date_range(day.min(), day.max() + pd.Timedelta(days=1), freq="D")
    zones = daily.index.get_level_values("pu_zone").unique().sort_values()
    grid = pd.MultiIndex.from_product([zones, days], names=["pu_zone", "day"])
    daily = daily.reindex(grid, fill_value=0)

    by_zone = daily.groupby(level="pu_zone", group_keys=False)
    window_sum = by_zone.rolling(window_days, min_periods=1).sum().droplevel(0)
    # shift(1) внутри зоны: строка дня D получает сумму по D-window..D-1.
    past = window_sum.groupby(level="pu_zone").shift(1, fill_value=0)

    # Деление 0/0 даёт NaN: среднее не определено, если в окне не было поездок.
    suffix = f"{window_days}d"
    result = pd.DataFrame(
        {
            "pu_zone": past.index.get_level_values("pu_zone"),
            "event_timestamp": past.index.get_level_values("day"),
            f"zone_trips_{suffix}": past["trips"].to_numpy(),
            f"zone_mean_duration_{suffix}": (past["duration_sum"] / past["trips"]).to_numpy(),
            f"zone_mean_distance_{suffix}": (past["distance_sum"] / past["trips"]).to_numpy(),
        }
    )
    return result.astype(schema).sort_values(["pu_zone", "event_timestamp"], ignore_index=True)
