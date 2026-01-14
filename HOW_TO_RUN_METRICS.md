# Как запустить обучение модели и подсчет метрик

## ⚠️ ВАЖНО: Проблема с Python 3.14

**SpaCy не работает с Python 3.14!** Используйте Python 3.10-3.13.

---

## ✅ Способ 1: Jupyter Notebook (РЕКОМЕНДУЕТСЯ!)

### 🎯 Это САМЫЙ ПРОСТОЙ и НАДЕЖНЫЙ способ!

Я уже добавил все необходимые ячейки в **`notebooks/ML.ipynb`**:

1. Откройте файл `notebooks/ML.ipynb` в Jupyter Lab/Notebook
2. Запустите ячейки **11-15** (или все с начала, если нужно обучить модель)
3. Получите все метрики! 🎉

**Новые ячейки:**
- **Cell 11**: Token-level метрики (Accuracy, Precision, Recall, F1)
- **Cell 12**: Детальный classification report
- **Cell 13**: Entity-level метрики (strict matching)
- **Cell 14**: Итоговая таблица
- **Cell 15**: Примеры предсказаний

---

## ✅ Способ 2: Автоматический скрипт

### С автоматической настройкой окружения:

```bash
# Создаст venv с правильным Python и установит зависимости
chmod +x setup_and_run.sh
./setup_and_run.sh
```

**Что делает скрипт:**
1. Находит подходящий Python (3.10-3.13)
2. Создает виртуальное окружение `venv/`
3. Устанавливает spacy, sklearn, pandas
4. Запускает оценку модели

---

## ✅ Способ 3: Вручную

### Если хотите полный контроль:

```bash
# 1. Создайте виртуальное окружение с Python 3.13
python3.13 -m venv venv
source venv/bin/activate

# 2. Установите зависимости
pip install spacy scikit-learn pandas kaggle

# 3. Убедитесь, что датасет скачан
# (он должен быть в notebooks/datasets/dataturks/)

# 4. Запустите оценку
python evaluate_model.py
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
