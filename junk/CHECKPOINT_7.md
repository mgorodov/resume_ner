# Чекпоинт 7 — MLflow, анализ ошибок, robustness

Финальный чекпоинт: модель и пайплайн становятся **наблюдаемыми и воспроизводимыми**. Используем MLflow для версионирования экспериментов, MinIO для S3-совместимого хранения артефактов.

## TL;DR

```bash
# 1. Поднять инфраструктуру (один раз)
docker-compose -f mlflow/docker-compose.mlflow.yaml up -d

# 2a. Запустить исследовательский ноутбук
jupyter notebook notebooks/DL_Experiments.ipynb     # секции 7.* — Checkpoint 7

# 2b. ИЛИ через CLI (бонус — позволяет повысить оценку прошлого чекпоинта)
python train_cli.py                                  # дефолтные параметры
python train_cli.py training.n_epochs=50            # переопределение
python train_cli.py data.use_augmentation=false     # без аугментации

# 3. Демонстрация инференса PRD-модели
jupyter notebook notebooks/DL_Demonstration.ipynb
# или
python demo_cli.py
```

UI:
- **MLflow** — http://localhost:5001
- **MinIO** — http://localhost:9001 (`minioadmin` / `minioadmin`)

## Что было сделано

### Инфраструктура — `mlflow/`

```
mlflow/
├── docker-compose.mlflow.yaml   # MLflow + MinIO + Postgres
└── README.md                    # инструкция и схема компонентов
```

Сервисы:
- **MLflow** (порт 5001, потому что 5000 на macOS занят AirPlay) — tracking server
- **MinIO** (9000 API / 9001 UI) — S3-совместимое хранилище артефактов
- **Postgres** (5433 host -> 5432 container) — backend store для метаданных MLflow

### Эксперимент — `notebooks/DL_Experiments.ipynb`

В существующий ноутбук добавлены секции **7.1–7.6**:

| Секция | Что делает |
|---|---|
| 7.1 Выбор финальной модели | Обоснование выбора `B+aug` (pretrained + augmentation, Entity-F1=0.604) |
| 7.2 Переобучение + MLflow | Полный run: параметры, метрики (per-epoch и финальные), артефакты (модель, learning curve, confusion matrix, prediction examples) — всё в MinIO. Тег `stage=PRD` |
| 7.3 Анализ ошибок | Категоризация (boundary / wrong label / missed / hallucinated) + 20 конкретных примеров с разбором причин |
| 7.4 Сравнение с baseline | Regex+dict baseline vs final model |
| 7.5 Robustness | 5 видов возмущений (lowercase, uppercase, typos, case_swap, no_punct) + наблюдения |
| 7.6 Итог | Сводная таблица: что сделано / где |

### Демонстрация — `notebooks/DL_Demonstration.ipynb`

Чистый ноутбук, **без обучения**. Показывает production-сценарий:

1. Подключается к MLflow tracking server
2. Находит последний run с тегом `stage=PRD`
3. Загружает модель из S3 (MinIO)
4. Применяет к произвольному резюме
5. Визуализирует результат через `spacy.displacy`

### Бонус — CLI с Hydra

`train_cli.py` + `demo_cli.py` + `conf/config.yaml`

Hydra даёт человекочитаемый конфиг и удобный override параметров с командной строки:

```bash
# Переопределение глубоко в иерархии
python train_cli.py model.type=baseline training.dropout=0.5

# Изменение MLflow назначения
python train_cli.py mlflow.experiment_name=my_exp mlflow.run_name=alt_run
```

## Файлы, добавленные/изменённые в чекпоинте 7

| Файл | Назначение |
|---|---|
| `mlflow/docker-compose.mlflow.yaml` | стек MLflow + MinIO + Postgres |
| `mlflow/README.md` | инструкции |
| `conf/config.yaml` | Hydra конфиг для CLI |
| `train_cli.py` | CLI обучение с MLflow |
| `demo_cli.py` | CLI инференс PRD-модели |
| `notebooks/DL_Experiments.ipynb` | + секции 7.1–7.6 |
| `notebooks/DL_Demonstration.ipynb` | новый ноутбук |
| `pyproject.toml` | + `mlflow`, `boto3`, `hydra-core` |
| `CHECKPOINT_7.md` | этот файл |

## Воспроизводимость

Зафиксировано:
- `random.seed(42)` + `np.random.seed(42)` перед обучением
- Все гиперпараметры → `conf/config.yaml` (или `CFG` в ноутбуке) → MLflow params
- Размеры split-ов (train=179, dev=19, test=22) фиксируются стратегией split-а с одинаковым seed
- Версия датасета фиксируется через путь и название (`dataturks/Entity Recognition in Resumes`)
- Версии библиотек — `pyproject.toml`

## Метрики финального run-а (типичные)

| Метрика | Значение |
|---|---|
| Test Entity-F1 | ~0.58–0.60 |
| Test Entity-P | ~0.60–0.65 |
| Test Entity-R | ~0.55–0.58 |
| Test Token Accuracy | ~0.92 |
| Время обучения | ~6–8 минут |

Конкретное значение видно в MLflow UI после run-а — лучшее, что удавалось на этом датасете с такой архитектурой (pretrained tok2vec + augmentation).
