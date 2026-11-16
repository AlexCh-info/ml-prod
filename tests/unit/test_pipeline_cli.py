"""Команды `taxi prepare` и `taxi build-features` и хелперы parquet из storage."""

from __future__ import annotations

from pathlib import Path

import fsspec
import pandas as pd
import pytest
from typer.testing import CliRunner

from taxi_duration.cli import app
from taxi_duration.data.prepare import PREPARED_SCHEMA
from taxi_duration.errors import DataError
from taxi_duration.features.zone_stats import zone_stats_schema
from taxi_duration.storage import read_parquet, write_parquet

runner = CliRunner()


def _raw_month(month: str, trips: int = 4) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-05 10:00:00")
    pickups = [start + pd.Timedelta(days=i) for i in range(trips)]
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": [*pickups, pd.Timestamp("2009-01-01 10:00:00")],
            "lpep_dropoff_datetime": [p + pd.Timedelta(minutes=12) for p in pickups]
            + [pd.Timestamp("2009-01-01 10:10:00")],
            "PULocationID": [74] * (trips + 1),
            "DOLocationID": [75] * (trips + 1),
            "trip_distance": [2.0] * (trips + 1),
            "fare_amount": [10.0] * (trips + 1),
        }
    ).astype(
        {
            "lpep_pickup_datetime": "datetime64[us]",
            "lpep_dropoff_datetime": "datetime64[us]",
            "PULocationID": "int32",
            "DOLocationID": "int32",
        }
    )


@pytest.fixture
def data_root(monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path) -> Path:
    root = tmp_path / "data"
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  data_root: {root}\ndata:\n  months: ["2024-01", "2024-02"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TAXI_CONFIG", str(config))
    monkeypatch.setenv("TAXI__LOGGING__CONFIG_PATH", str(repo_root / "configs" / "logging.yaml"))
    return root


def _write_raw(root: Path, month: str) -> None:
    path = root / "raw" / f"green_tripdata_{month}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    _raw_month(month).to_parquet(path)


# --- taxi prepare -------------------------------------------------------------------------------


def test_prepare_writes_processed_month(data_root: Path) -> None:
    _write_raw(data_root, "2024-01")

    result = runner.invoke(app, ["prepare", "--month", "2024-01"])

    assert result.exit_code == 0, result.output
    trips = pd.read_parquet(data_root / "processed" / "trips_2024-01.parquet")
    assert len(trips) == 4  # поездка из 2009 отфильтрована
    assert {col: str(dtype) for col, dtype in trips.dtypes.items()} == PREPARED_SCHEMA


def test_prepare_without_raw_file_exits_1_with_hint(data_root: Path) -> None:
    result = runner.invoke(app, ["prepare", "--month", "2024-01"])
    assert result.exit_code == 1
    assert "taxi ingest --month 2024-01" in result.stderr
    assert not (data_root / "processed").exists()


def test_prepare_rejects_bad_month(data_root: Path) -> None:
    result = runner.invoke(app, ["prepare", "--month", "2024-13"])
    assert result.exit_code == 2


# --- taxi build-features ------------------------------------------------------------------------


def test_build_features_uses_all_configured_months(data_root: Path) -> None:
    for month in ("2024-01", "2024-02"):
        _write_raw(data_root, month)
        assert runner.invoke(app, ["prepare", "--month", month]).exit_code == 0

    result = runner.invoke(app, ["build-features"])

    assert result.exit_code == 0, result.output
    stats = pd.read_parquet(data_root / "features" / "zone_stats.parquet")
    assert {col: str(dtype) for col, dtype in stats.dtypes.items()} == zone_stats_schema(7)
    assert stats["event_timestamp"].min() == pd.Timestamp("2024-01-05")
    assert stats["event_timestamp"].max() == pd.Timestamp("2024-02-09")
    # Строка 01-09: окно 01-02..01-08 содержит все 4 январские поездки (01-05..01-08).
    jan9 = stats[stats["event_timestamp"] == pd.Timestamp("2024-01-09")].iloc[0]
    assert jan9["zone_trips_7d"] == 4


def test_build_features_without_processed_month_exits_1_with_hint(data_root: Path) -> None:
    _write_raw(data_root, "2024-01")
    assert runner.invoke(app, ["prepare", "--month", "2024-01"]).exit_code == 0

    result = runner.invoke(app, ["build-features"])

    assert result.exit_code == 1
    assert "taxi prepare --month 2024-02" in result.stderr
    assert not (data_root / "features").exists()


# --- storage: read_parquet / write_parquet ------------------------------------------------------


def test_write_parquet_is_readable_and_world_readable(tmp_path: Path) -> None:
    uri = str(tmp_path / "nested" / "out.parquet")
    df = pd.DataFrame({"a": [1, 2]})

    write_parquet(df, uri)

    pd.testing.assert_frame_equal(read_parquet(uri), df)
    assert Path(uri).stat().st_mode & 0o777 == 0o644
    assert list(Path(uri).parent.glob("*.part")) == []


def test_write_parquet_failure_keeps_old_file(tmp_path: Path) -> None:
    uri = str(tmp_path / "out.parquet")
    write_parquet(pd.DataFrame({"a": [1]}), uri)

    bad = pd.DataFrame({"a": [object()]})  # pyarrow не умеет сериализовать object()
    with pytest.raises(Exception):  # noqa: B017 - тип ошибки pyarrow тут не важен
        write_parquet(bad, uri)

    assert pd.read_parquet(uri)["a"].tolist() == [1]
    assert list(tmp_path.glob("*.part")) == []


def test_read_parquet_missing_file_is_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="run something"):
        read_parquet(str(tmp_path / "nope.parquet"), hint="run something")


def test_parquet_roundtrip_on_remote_filesystem() -> None:
    uri = "memory://lake/features/zone_stats.parquet"
    df = pd.DataFrame({"a": [1, 2, 3]})

    write_parquet(df, uri)

    pd.testing.assert_frame_equal(read_parquet(uri), df)
    fsspec.filesystem("memory").rm("/lake", recursive=True)
