# MLflow Infrastructure

Локальный MLflow tracking server с MinIO в качестве S3-совместимого хранилища артефактов и Postgres для метаданных.

## Быстрый старт

```bash
# Запустить сервисы
docker compose -f mlflow/docker-compose.mlflow.yaml up -d

# Проверить, что всё поднялось
docker compose -f mlflow/docker-compose.mlflow.yaml ps
```

После запуска:

- **MLflow UI** — http://localhost:5001  *(порт 5000 занят macOS AirPlay)*
- **MinIO UI** — http://localhost:9001 (login: `minioadmin` / `minioadmin`)
- **S3 endpoint** — http://localhost:9000

## Подключение из Python

```python
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("resume-ner")
```

## Зачем какие компоненты

| Компонент | Назначение |
|---|---|
| **MLflow server** | Tracking server: эксперименты, runs, метрики, параметры |
| **MinIO** | S3-совместимое хранилище артефактов (модели, графики, примеры) |
| **Postgres** | Backend store для метаданных MLflow |

## Остановка / очистка

```bash
docker compose -f mlflow/docker-compose.mlflow.yaml down          # stop
docker compose -f mlflow/docker-compose.mlflow.yaml down -v       # stop + remove volumes
```
