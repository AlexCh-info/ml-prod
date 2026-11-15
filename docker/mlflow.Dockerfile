# syntax=docker/dockerfile:1
# MLflow tracking server для стенда core.
# Официальный образ содержит только mlflow: драйвера Postgres и клиента S3 в нём нет,
# а запускается он от root. Добавляем зависимости и непривилегированного пользователя.
# Версии зафиксированы: обновление MLflow — осознанное действие (миграции схемы БД!).
ARG MLFLOW_VERSION=3.18.0
FROM ghcr.io/mlflow/mlflow:v${MLFLOW_VERSION}

# psycopg2-binary — драйвер для --backend-store-uri postgresql://...
# boto3 — клиент S3: сервер сам пишет артефакты в RustFS (--artifacts-destination s3://mlflow).
RUN pip install --no-cache-dir \
        psycopg2-binary==2.9.13 \
        boto3==1.44.3 \
    && useradd --system --uid 10001 --no-create-home mlflow

USER mlflow
EXPOSE 5000

# Аргументы сервера (БД, S3, allowed-hosts) задаются в docker-compose.yml:
# так один образ работает и в compose, и в Kubernetes с другими адресами.
ENTRYPOINT ["mlflow", "server"]
CMD ["--host", "0.0.0.0", "--port", "5000"]
