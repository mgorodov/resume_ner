# Как запустить обучение модели и подсчет метрик

## ✅ Простой способ (рекомендуется)

### Вариант 1: Jupyter Notebook (уже готово!)

Я уже добавил все необходимые ячейки в **`notebooks/ML.ipynb`**:

1. Откройте файл `notebooks/ML.ipynb` в Jupyter
2. Запустите все ячейки по порядку (или просто Cell 11-15, если модель уже обучена)
3. Получите результаты!

**Новые ячейки:**
- **Cell 11**: Подсчет token-level метрик (Accuracy, Precision, Recall, F1)
- **Cell 12**: Детальный отчет по классам
- **Cell 13**: Entity-level метрики (strict matching)
- **Cell 14**: Итоговая таблица всех метрик
- **Cell 15**: Примеры предсказаний модели

### Вариант 2: Python скрипт

Если у вас настроено окружение с Python 3.10-3.12 и установлены зависимости:

```bash
# Оценка уже обученной модели
python3 evaluate_model.py

# Или полное обучение с нуля
python3 train_and_evaluate.py
```

## 📊 Что вы получите

### Token-level метрики
- **Accuracy** - общая точность классификации токенов
- **Precision** - точность (weighted average)
- **Recall** - полнота (weighted average) 
- **F1-Score** - гармоническое среднее

### Entity-level метрики (Strict Matching)
- **Precision** - точность распознавания целых сущностей
- **Recall** - полнота распознавания
- **F1-Score** - общая оценка качества
- **TP/FP/FN** - детальная статистика

### Дополнительно
- Детальный classification report по каждому классу
- Примеры работы модели на реальных резюме
- Сравнительная таблица метрик

## 🔧 Технические детали

**Параметры обучения:**
- Модель: SpaCy Blank Model
- Train/Test split: 90% / 10% (198 / 22 резюме)
- Эпохи: 10
- Dropout: 0.2
- Схема разметки: BILOU (Begin, Inside, Last, Outside, Unit)

**Датасет:**
- Источник: Dataturks Resume Entities for NER (Kaggle)
- Объем: 220 резюме
- Сущностей: 3,556
- Типов сущностей: 11

## 🐛 Устранение проблем

### Если Python 3.14

SpaCy не полностью совместим с Python 3.14. Используйте Python 3.10-3.12:

```bash
# Проверьте доступные версии
ls /opt/homebrew/bin/ | grep python

# Используйте конкретную версию
/opt/homebrew/bin/python3.13 evaluate_model.py
```

### Если нет библиотек

Установите зависимости из pyproject.toml или напрямую:

```bash
pip install spacy scikit-learn pandas
```

Или используйте uv (если установлен):

```bash
uv pip install spacy scikit-learn pandas
```

### Если модель не найдена

Сначала обучите модель, запустив ячейки 0-8 в `notebooks/ML.ipynb`.

## 📝 Файлы в проекте

- `train_and_evaluate.py` - Полное обучение и оценка
- `evaluate_model.py` - Оценка готовой модели
- `notebooks/ML.ipynb` - Jupyter notebook (рекомендуется!)
- `resume_ner_model/` - Директория с обученной моделью

## 💡 Рекомендация

**Самый простой и надежный способ** - использовать Jupyter notebook (`notebooks/ML.ipynb`), 
где уже все настроено и работает! 🚀
