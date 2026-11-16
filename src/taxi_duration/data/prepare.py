"""Подготовка поездок: таргет ``duration_min``, фильтры выбросов, фиксированная схема.

Спека trip-preparation. Функция ``prepare_trips`` чистая (без I/O), поэтому её легко
тестировать, а читать и писать файлы будет команда ``taxi prepare`` (задача 2.3).
"""

from __future__ import annotations

import re

import pandas as pd

from taxi_duration.config import FilterSettings
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
    # Пропущенные колонки, сравниваем колонки из списка нужных и тех что в датасете
    missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in raw.columns]
    # Если есть пропущенные колонки выводим ошибку
    if missing_cols:
        raise DataError(f"Отсутствуют обязательные колонки: {', '.join(missing_cols)}")

    # Проверка на правильный формат Datetime
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise DataError(f"Неверный формат месяца (ожидается YYYY-MM): {month!r}")

    # Определеяем границы нужного месяца
    period = pd.Period(month, freq="M")
    month_start = period.start_time
    next_month_start = (period + 1).start_time

    # Выбираем нужные колонки
    df = raw.loc[:, list(RAW_REQUIRED_COLUMNS)].dropna(subset=list(RAW_REQUIRED_COLUMNS))

    df["pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"]).astype("datetime64[us]")
    dropoff = pd.to_datetime(df["lpep_dropoff_datetime"]).astype("datetime64[us]")

    # Считаем длительность и сохраняем её колонкой в тот же df
    df["duration_min"] = (dropoff - df["pickup_datetime"]).dt.total_seconds() / 60.0

    # Фильтруем сам df по месяцу и длительности — все колонки фильтруются синхронно
    mask = (
        (df["pickup_datetime"] >= month_start)
        & (df["pickup_datetime"] < next_month_start)
        & (df["duration_min"] >= filters.min_duration)
        & (df["duration_min"] <= filters.max_duration)
    )
    df = df.loc[mask].reset_index(drop=True)

    # Вычисляем признаки из уже отфильтрованного pickup_datetime и собираем итог
    result = pd.DataFrame(
        {
            "pickup_datetime": df["pickup_datetime"],
            "pu_zone": df["PULocationID"],
            "do_zone": df["DOLocationID"],
            "trip_distance": df["trip_distance"],
            "pickup_hour": df["pickup_datetime"].dt.hour,
            "pickup_weekday": df["pickup_datetime"].dt.day_of_week,
            "duration_min": df["duration_min"],
        }
    )

    return result.astype(PREPARED_SCHEMA)
