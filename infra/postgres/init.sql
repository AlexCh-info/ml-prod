-- Инициализация Postgres для стенда core: отдельная база и владелец для каждого сервиса.
--
-- Когда выполняется: entrypoint образа postgres запускает файлы из
-- /docker-entrypoint-initdb.d/ ТОЛЬКО при первом старте, когда volume с данными пуст.
-- Правка этого файла на живом стенде ничего не изменит: нужен `docker compose down -v`
-- (удалит ВСЕ данные MLflow) или ручной psql.
--
-- Секреты: пароли не захардкожены, а читаются из env контейнера через \getenv
-- (мета-команда psql 15+). Сами переменные приходят из .env через docker-compose.yml.
-- ON_ERROR_STOP: при любой ошибке скрипт падает, и контейнер не стартует с полусозданной БД.

\set ON_ERROR_STOP on

\getenv mlflow_user MLFLOW_DB_USER
\getenv mlflow_password MLFLOW_DB_PASSWORD
\getenv airflow_user AIRFLOW_DB_USER
\getenv airflow_password AIRFLOW_DB_PASSWORD

-- :"var" подставляет идентификатор в кавычках, :'var' — строковый литерал.
-- Так значения из env экранируются и не ломают SQL.
CREATE USER :"mlflow_user" WITH PASSWORD :'mlflow_password';
CREATE DATABASE mlflow OWNER :"mlflow_user";

CREATE USER :"airflow_user" WITH PASSWORD :'airflow_password';
CREATE DATABASE airflow OWNER :"airflow_user";

-- С Postgres 15 схема public доступна на запись только владельцу базы.
-- Владельцы выше уже совпадают с пользователями сервисов, поэтому GRANT не нужен.
