"""Контракт данных: наборы ожиданий GX, описанные кодом.

Источник правды контракта — этот файл: он проходит ревью как код, а JSON в
``gx/expectations/`` — производный (runner пересоздаёт его при каждом запуске).

Общие правила для обоих наборов:

- имя suite стабильно и равно ``Stage.suite_name`` (``raw_trips`` / ``prepared_trips``),
  иначе runner откажется его запускать;
- у КАЖДОГО ожидания severity указан явно (``severity="critical"`` или ``"warning"``):
  без него GX молча ставит ``critical``;
- границы месяца не зашиваются в код: используй параметры suite
  ``{"$PARAMETER": "month_start"}`` и ``{"$PARAMETER": "month_end"}``
  (ISO-строки, ``month_end`` — начало следующего месяца, его не включаем: ``strict_max=True``).
  Значения подставляет runner (``runner.suite_parameters``) в момент запуска.

Политика severity — design.md, решение 3.
"""

from __future__ import annotations

import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.expectations.metadata_types import FailureSeverity

from taxi_duration.data.prepare import PREPARED_SCHEMA, RAW_REQUIRED_COLUMNS

RAW_TRIPS_SUITE = "raw_trips"
PREPARED_TRIPS_SUITE = "prepared_trips"


def build_raw_trips_suite() -> gx.ExpectationSuite:
    """Задача 2.2 [YOU]: контракт сырого месячного файла TLC Green Taxi.

    Контракт (проверяется ``tests/unit/test_suites.py``):

    1. Имя ``RAW_TRIPS_SUITE``; не меньше 7 ожиданий; у каждого severity явно.
    2. **critical:**
       - схема: обязательные колонки сырого файла присутствуют
         (минимум ``taxi_duration.data.prepare.RAW_REQUIRED_COLUMNS``);
       - нет пропусков в ключевых полях (время посадки/высадки, зоны);
       - ``lpep_dropoff_datetime >= lpep_pickup_datetime`` (с ``mostly``: в данных TLC
         такие строки встречаются, prepare их отфильтрует; подумай, какой порог честный);
       - число строк в разумных пределах (пустой или подозрительно маленький файл — стоп).
    3. **warning:**
       - доля посадок вне месяца (``month_start`` ≤ pickup < ``month_end``) с ``mostly``;
       - диапазон ``trip_distance`` (выбросы вроде 35 000 миль) с ``mostly``;
       - диапазон ``passenger_count`` с ``mostly``.

    Проверка задачи: ``uv run taxi validate --month 2024-01 --stage raw`` → код 0.
    """
    # severity=critical
    # Создаем ожидание
    suite = gx.ExpectationSuite(name=RAW_TRIPS_SUITE)

    # Добавляем первое ожидание схема, присутствие колонок
    for col in RAW_REQUIRED_COLUMNS:
        suite.add_expectation(
            gxe.ExpectColumnToExist(column=col, severity=FailureSeverity.CRITICAL)
        )

    # Проверка на нули в ключевых полях
    # Ключевые поля
    main_cols = ["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID"]
    for col in main_cols:
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(column=col, severity=FailureSeverity.CRITICAL)
        )

    # Сравнение lpep_dropoff_datetime и lpep_pickup_datetime
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="lpep_dropoff_datetime",
            column_B="lpep_pickup_datetime",
            or_equal=True,
            mostly=0.999,
            severity=FailureSeverity.CRITICAL,
        )
    )

    # Число строк в файле (таблице)
    suite.add_expectation(
        gxe.ExpectTableRowCountToBeBetween(
            min_value=10_000, max_value=500_000, severity=FailureSeverity.CRITICAL
        )
    )

    # severity=warning
    # Доля посадок вне месяца
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="lpep_pickup_datetime",
            min_value={"$PARAMETER": "month_start"},
            max_value={"$PARAMETER": "month_end"},
            strict_max=True,
            mostly=0.999,
            severity=FailureSeverity.WARNING,
        )
    )

    # Выбросы в trip_distance
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="trip_distance",
            max_value=100.0,
            min_value=0.0,
            mostly=0.995,
            severity=FailureSeverity.WARNING,
        )
    )

    # Выбросы с passenger_count
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="passenger_count",
            min_value=1,
            max_value=6,
            mostly=0.98,
            severity=FailureSeverity.WARNING,
        )
    )

    return suite


def build_prepared_trips_suite() -> gx.ExpectationSuite:
    """Задача 2.3 [YOU]: контракт результата ``taxi prepare`` (``PREPARED_SCHEMA``).

    Контракт (проверяется ``tests/unit/test_suites.py``):

    1. Имя ``PREPARED_TRIPS_SUITE``; у каждого ожидания severity явно.
    2. **critical:**
       - схема: колонки ровно ``taxi_duration.data.prepare.PREPARED_SCHEMA``, в том же порядке;
       - типы колонок как в ``PREPARED_SCHEMA``;
       - ``duration_min`` в 1..60 (без ``mostly``: prepare это гарантирует, любое
         нарушение — баг в коде, а не шум в данных);
       - ``pickup_hour`` в 0..23.
    3. Уникальность строк/ключа не проверяем — объясни ментору почему.

    Проверка задачи: ``uv run taxi validate --month 2024-01 --stage prepared`` → код 0.
    """

    # Объявляем ожидание
    suite = gx.ExpectationSuite(name=PREPARED_TRIPS_SUITE)

    # severity=critical
    # Сравнение с финальной схемой (колонки на выходе)
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchOrderedList(
            column_list=list(PREPARED_SCHEMA.keys()), severity=FailureSeverity.CRITICAL
        )
    )

    # Проверка типов
    for key, item in PREPARED_SCHEMA.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeOfType(
                column=key, type_=item, severity=FailureSeverity.CRITICAL
            )
        )

    # Проверка отрезка значений для duration_min
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="duration_min", min_value=1, max_value=60, severity=FailureSeverity.CRITICAL
        )
    )

    # Проверка отрезка значений для pickup_hour
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="pickup_hour", min_value=0, max_value=23, severity=FailureSeverity.CRITICAL
        )
    )

    return suite
