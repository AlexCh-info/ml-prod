"""Подготовка поездок: таргет ``duration_min``, фильтры выбросов, фиксированная схема.

Спека trip-preparation. Функция ``prepare_trips`` чистая (без I/O), поэтому её легко
тестировать, а читать и писать файлы будет команда ``taxi prepare`` (задача 2.3).
"""

from __future__ import annotations

import pandas as pd

from taxi_duration.config import FilterSettings, parse_month
from taxi_duration.errors import DataError

# Колонки сырого файла TLC, без которых подготовка невозможна.
RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
)

# Схема результата: порядок колонок и dtype фиксированы (контракт для этапов 03–05).
PREPARED_SCHEMA: dict[str, str] = {
    "pickup_datetime": "datetime64[us]",
    "pu_zone": "int32",
    "do_zone": "int32",
    "trip_distance": "float64",
    "pickup_hour": "int32",
    "pickup_weekday": "int32",
    "duration_min": "float64",
}

# Переименование сырых колонок TLC в имена схемы результата.
_RAW_TO_PREPARED: dict[str, str] = {
    "lpep_pickup_datetime": "pickup_datetime",
    "PULocationID": "pu_zone",
    "DOLocationID": "do_zone",
}


def prepare_trips(raw: pd.DataFrame, month: str, filters: FilterSettings) -> pd.DataFrame:
    """Очистить сырые поездки одного месяца и посчитать таргет.

    Контракт:

    1. Если в ``raw`` нет хотя бы одной колонки из ``RAW_REQUIRED_COLUMNS`` —
       ``DataError``, в сообщении перечислены недостающие колонки.
       Неверный ``month`` (не ``YYYY-MM``) — тоже ``DataError``.
    2. Строки с пропуском (NaN/NaT) в любой из ``RAW_REQUIRED_COLUMNS`` отбрасываются.
    3. ``duration_min`` = (``lpep_dropoff_datetime`` − ``lpep_pickup_datetime``) в минутах,
       дробное число (15 мин 30 с → 15.5).
    4. Остаются только поездки с
       ``filters.min_duration <= duration_min <= filters.max_duration`` (границы включены).
    5. Остаются только поездки, у которых посадка попадает в ``month``:
       ``начало месяца <= pickup < начало следующего месяца``.
    6. Колонки результата — ровно ``PREPARED_SCHEMA``, в том же порядке и с теми же dtype:
       ``pickup_datetime`` ← ``lpep_pickup_datetime``; ``pu_zone`` ← ``PULocationID``;
       ``do_zone`` ← ``DOLocationID``; ``pickup_hour`` — час посадки 0..23;
       ``pickup_weekday`` — день недели посадки, понедельник = 0 … воскресенье = 6.
       Остальные колонки ``raw`` в результат не попадают.
    7. Порядок оставшихся строк сохраняется, индекс — ``RangeIndex`` 0..n-1.
    8. Если после фильтров ничего не осталось — пустой DataFrame с той же схемой.
    9. ``raw`` не изменяется (функция чистая).

    Args:
        raw: сырые поездки из ``green_tripdata_YYYY-MM.parquet``.
        month: обрабатываемый месяц ``YYYY-MM``.
        filters: пороги длительности из ``data.filters`` конфига.

    Returns:
        Очищенные поездки со схемой ``PREPARED_SCHEMA``.

    Raises:
        DataError: нет обязательных колонок или неверный ``month``.
    """
    # Контракт п.1: сначала проверяем вход, до любых вычислений.
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise DataError(f"Missing required raw columns: {', '.join(missing)}")
    try:
        # Один источник правды для формата месяца — тот же, что валидирует конфиг и CLI.
        month_start = pd.Timestamp(parse_month(month))
    except ValueError as exc:
        raise DataError(str(exc)) from exc
    next_month_start = month_start + pd.DateOffset(months=1)

    # Берём только нужные колонки (п.2): остальные колонки TLC не копируем и не проверяем
    # на пропуски. raw не меняется: в pandas 3 Copy-on-Write, присваивание ниже — в копию.
    trips = raw.loc[:, list(RAW_REQUIRED_COLUMNS)].dropna()
    pickup = trips["lpep_pickup_datetime"]
    trips["duration_min"] = (trips["lpep_dropoff_datetime"] - pickup).dt.total_seconds() / 60

    # Одна маска на все условия (п.4–5): таргет лежит колонкой в той же таблице,
    # поэтому строки всех колонок отбрасываются синхронно, индексы не разъезжаются.
    in_month = (pickup >= month_start) & (pickup < next_month_start)
    in_range = trips["duration_min"].between(filters.min_duration, filters.max_duration)
    trips = trips.loc[in_month & in_range].rename(columns=_RAW_TO_PREPARED)

    # Признаки времени считаем уже на отфильтрованных строках — меньше работы.
    trips["pickup_hour"] = trips["pickup_datetime"].dt.hour
    trips["pickup_weekday"] = trips["pickup_datetime"].dt.day_of_week  # пн = 0 … вс = 6

    # Порядок колонок и dtype берём из PREPARED_SCHEMA (п.6): схема описана в одном месте.
    return trips.loc[:, list(PREPARED_SCHEMA)].astype(PREPARED_SCHEMA).reset_index(drop=True)
