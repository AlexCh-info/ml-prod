"""taxi-duration: прогноз длительности поездки NYC Green Taxi."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Единый источник версии — [project].version в pyproject.toml,
    # попадает в метаданные установленного пакета.
    __version__: str = version("taxi-duration")
except PackageNotFoundError:  # pragma: no cover - пакет импортирован без установки
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
