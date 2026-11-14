import logging.config

import yaml

from taxi_duration.config import Settings
from taxi_duration.errors import ConfigError


def setup_logging(settings: Settings) -> None:
    """Настроить логирование процесса по settings.logging.

    - читает YAML-шаблон из settings.logging.config_path;
    - handler `console` использует formatter с именем settings.logging.format (text|json);
    - логгер `taxi_duration` получает уровень settings.logging.level;
    - применяет результат через logging.config.dictConfig;
    - повторный вызов не дублирует вывод.

    Raises:
        ConfigError: файл шаблона не найден (в сообщении — путь).
    """

    # Получаем путь к YAML шаблону
    logging_path = settings.logging.config_path

    # Проверяем что по полученому пути лежит файл и он существует
    if not logging_path.is_file():
        raise ConfigError(f"Logging config not found: {logging_path}")

    # Получаем конфигурацию YAML в виде Dict
    dict_text = logging_path.read_text(encoding="utf-8")
    logging_dict = yaml.safe_load(dict_text)

    # Применяем полученную конфигурацию из класса
    logging_dict["handlers"]["console"]["formatter"] = settings.logging.format
    logging_dict["loggers"]["taxi_duration"]["level"] = settings.logging.level

    # Записываем изменения
    logging.config.dictConfig(logging_dict)
