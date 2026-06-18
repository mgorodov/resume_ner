# Архитектура моделей Resume NER — Чекпоинт 7

## 📋 Общая схема пайплайна

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CHECKPOINT 7 ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │   Dataset    │───▶│  Preprocess  │───▶│   Model Training        │   │
│  │  (Dataturks) │    │  (cleaning)  │    │   + MLflow Logging      │   │
│  └──────────────┘    └──────────────┘    └─────────────────────────┘   │
│                                              │                          │
│                    ┌─────────────────────────┼──────────────────────┐   │
│                    │                         ▼                      │   │
│                    │              ┌─────────────────────┐           │   │
│                    │              │   MLflow Tracking   │           │   │
│                    │              │   Server (5001)     │           │   │
│                    │              └─────────────────────┘           │   │
│                    │                       │                         │   │
│                    │         ┌─────────────┴─────────────┐           │   │
│                    │         ▼                           ▼           │   │
│                    │  ┌──────────────┐          ┌──────────────┐     │   │
│                    │  │   MinIO S3   │          │  PostgreSQL  │     │   │
│                    │  │  Artifacts   │          │  Metadata    │     │   │
│                    │  │  (9000/9001) │          │  (5433)      │     │   │
│                    │  └──────────────┘          └──────────────┘     │   │
│                    │                                                  │   │
│                    ▼                                                  │   │
│         ┌──────────────────────┐                                     │   │
│         │  Production Model    │                                     │   │
│         │  (tagged: stage=PRD) │                                     │   │
│         └──────────────────────┘                                     │   │
│                    │                                                  │   │
│         ┌──────────┴──────────┐                                       │   │
│         ▼                     ▼                                       │   │
│  ┌──────────────┐     ┌──────────────┐                               │   │
│  │  demo_cli.py │     │  ASGI API    │                               │   │
│  │  (inference) │     │  (serving)   │                               │   │
│  └──────────────┘     └──────────────┘                               │   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Архитектура моделей (по возрастанию сложности)

### 1️⃣ **A: Baseline (SpaCy Blank HashEmbedCNN)**

```
┌────────────────────────────────────────────────────────────┐
│                    BASELINE ARCHITECTURE                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Tokenizer (SpaCy en)                     │  │
│  │  - Word segmentation                                  │  │
│  │  - Whitespace rules                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         HashEmbedCNN (tok2vec)                        │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Embedding Layer (HashEmbed)                   │  │  │
│  │  │  - Input: token orth, shape, prefix, suffix    │  │  │
│  │  │  - 5000 hash buckets × 128 dim                 │  │  │
│  │  │  - No pretrained vectors (learns from scratch) │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │      │                                                 │  │
│  │      ▼                                                 │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  CNN Layers (3-4 convolutional layers)         │  │  │
│  │  │  - Kernel sizes: 3, 4, 5                       │  │  │
│  │  │  - Output: 128-dim context vectors             │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              NER Tagging Layer                        │  │
│  │  - BIO scheme (Begin, Inside, Outside)               │  │
│  │  - 10 entity types + O                               │  │
│  │  - Softmax classification per token                  │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  Output: [(start, end, label), ...]                         │
│                                                             │
│  Entity-F1: 0.5789                                          │
│  Parameters: ~1-2M (learned from scratch)                   │
└────────────────────────────────────────────────────────────┘
```

**Ключевые особенности:**
- **Пустая модель** (`spacy.blank("en")`) — обучается с нуля
- **HashEmbed** — хеширование признаков вместо lookup-таблицы
- **CNN** — захват локального контекста (n-gram паттерны)
- **Преимущества:** быстрое обучение, нет зависимости от внешних векторов
- **Недостатки:** требует больше данных для качественного обучения

---

### 2️⃣ **B: Pretrained (GloVe 300d + CNN)**

```
┌────────────────────────────────────────────────────────────┐
│                 PRETRAINED ARCHITECTURE                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Tokenizer (SpaCy en_core_web_lg)              │  │
│  │  - Same as baseline                                   │  │
│  │  - + Lemmatization rules available                    │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Pretrained tok2vec (en_core_web_lg)              │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Embedding Layer (Lookup + CNN)                │  │  │
│  │  │  - 685k words × 300 dim (GloVe vectors)        │  │  │
│  │  │  - OOV words: CNN fallback                     │  │  │
│  │  │  - Semantic similarity baked in                │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │      │                                                 │  │
│  │      ▼                                                 │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  CNN Layers (same as baseline)                 │  │  │
│  │  │  - Contextualizes pretrained embeddings        │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              NER Tagging Layer                        │  │
│  │  - Same as baseline                                   │  │
│  │  - But learns faster with better features            │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  Output: [(start, end, label), ...]                         │
│                                                             │
│  Entity-F1: 0.5417 (clean) / 0.6040 (augmented)             │
│  Parameters: ~20M (frozen GloVe) + ~1-2M (trainable)        │
└────────────────────────────────────────────────────────────┘
```

**Ключевые особенности:**
- **Предобученные векторы** — GloVe 300d на 685k словах
- **Transfer learning** — использует семантические связи из GloVe
- **OOV handling** — CNN для слов вне словаря
- **Преимущества:** лучше обобщает, требует меньше данных
- **Недостатки:** больше память, статические эмбеддинги

---

### 3️⃣ **C: Hybrid (Pretrained + Entity Ruler)**

```
┌────────────────────────────────────────────────────────────┐
│                  HYBRID ARCHITECTURE                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text                                                 │
│      │                                                      │
│      ├─────────────────────────────┐                        │
│      │                             │                        │
│      ▼                             ▼                        │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Entity Ruler    │    │  Pretrained NER  │              │
│  │  (Rule-based)    │    │  (ML model)      │              │
│  │                  │    │                  │              │
│  │  Patterns:       │    │  en_core_web_lg  │              │
│  │  - Email regex   │    │  + CNN + NER     │              │
│  │  - Year regex    │    │                  │              │
│  │  - Experience    │    │                  │              │
│  │    patterns      │    │                  │              │
│  │                  │    │                  │              │
│  │  [Email]         │    │  [Skills]        │              │
│  │  [Year]          │    │  [Location]      │              │
│  │  [Experience]    │    │  [Companies]     │              │
│  └──────────────────┘    └──────────────────┘              │
│            │                       │                        │
│            └───────────┬───────────┘                        │
│                        │                                    │
│                        ▼                                    │
│              ┌──────────────────────┐                      │
│              │  Merge Results       │                      │
│              │  (Entity Ruler has   │                      │
│              │   priority)          │                      │
│              └──────────────────────┘                      │
│                        │                                    │
│                        ▼                                    │
│  Output: [(start, end, label), ...]                         │
│                                                             │
│  Entity-F1: 0.5682                                          │
│  Entity-Precision: 0.6944 (highest)                         │
│  Entity-Recall: 0.4808 (lower due to limited rules)         │
└────────────────────────────────────────────────────────────┘
```

**Entity Ruler Patterns:**
```python
[
    {"label": "Email Address", "pattern": [{"REGEX": r"[\w.+-]+@[\w-]+\.[\w.-]+"}]},
    {"label": "Graduation Year", "pattern": [{"REGEX": r"\b(19|20)\d{2}\b"}]},
    {"label": "Years of Experience", "pattern": [
        {"LOWER": {"REGEX": r"\d+"}}, {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}}
    ]},
]
```

**Ключевые особенности:**
- **Дуальная система** — правила + ML
- **Высокая precision** — правила никогда не ошибаются на своих паттернах
- **Низкая recall** — правила покрывают только 3 из 10 типов сущностей
- **Use case:** когда критична точность, а не полнота

---

### 4️⃣ **D: Transformer (RoBERTa-base) — экспериментальная**

```
┌────────────────────────────────────────────────────────────┐
│                TRANSFORMER ARCHITECTURE                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         RoBERTa Tokenizer (Byte-Pair Encoding)        │  │
│  │  - 50k vocab subword tokens                           │  │
│  │  - Handles OOV gracefully                             │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         RoBERTa-base Encoder (12 layers)              │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Self-Attention (Multi-head)                   │  │  │
│  │  │  - Global context for each token               │  │  │
│  │  │  - 12 layers × 12 heads × 768 dim              │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │      │                                                 │  │
│  │      ▼                                                 │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Contextual Token Representations              │  │  │
│  │  │  - 768-dim per token                           │  │  │
│  │  │  - Bidirectional context                       │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         NER Classification Head                       │  │
│  │  - Linear layer on top of RoBERTa                    │  │
│  │  - BIO tagging per token                             │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  Output: [(start, end, label), ...]                         │
│                                                             │
│  Entity-F1: ~0.55-0.60 (typical for this dataset size)      │
│  Parameters: 125M (RoBERTa-base) + ~1M (head)               │
│  Inference: ~100-500ms per resume                           │
└────────────────────────────────────────────────────────────┘
```

**Ключевые особенности:**
- **Contextual embeddings** — динамические векторы в зависимости от контекста
- **Self-attention** — глобальный контекст для каждого токена
- **Fine-tuning** — дообучение всего трансформера на NER
- **Преимущества:** лучшее качество на сложных случаях
- **Недостатки:** медленно, требует GPU для обучения

---

### 5️⃣ **E: LLM Zero-Shot (Qwen3) — экспериментальная**

```
┌────────────────────────────────────────────────────────────┐
│                  LLM ZERO-SHOT ARCHITECTURE                 │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text (truncated to 3000 chars)                       │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Prompt Template                                      │  │
│  │  "Extract named entities from the following resume   │  │
│  │   text. Return ONLY a valid JSON object with these   │  │
│  │   entity types as keys: {entity_types}..."           │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Qwen3 LLM (0.6B / 1.7B / 4B params)           │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Transformer Decoder (causal attention)        │  │  │
│  │  │  - Autoregressive generation                   │  │  │
│  │  │  - No fine-tuning (zero-shot)                  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Generated JSON Response                       │  │
│  │  {"Name": ["John Smith"],                            │  │
│  │   "Email Address": ["john@example.com"], ...}        │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Post-processing                                │  │
│  │  - Parse JSON                                          │  │
│  │  - Match extracted strings back to original text     │  │
│  │  - Convert to (start, end, label) spans              │  │
│  └──────────────────────────────────────────────────────┘  │
│      │                                                      │
│      ▼                                                      │
│  Output: [(start, end, label), ...]                         │
│                                                             │
│  Entity-F1: 0.17-0.23 (Qwen3-0.6B → 4B)                     │
│  Inference: 10-25 seconds per resume (on MPS)               │
│  Parse failures: 32% → 0% (0.6B → 4B)                       │
└────────────────────────────────────────────────────────────┘
```

**Ключевые особенности:**
- **Zero-shot** — нет обучения на наших данных
- **Generative approach** — модель генерирует JSON, а не теги
- **Semantic understanding** — LLM "понимает" семантику сущностей
- **Проблемы:**
  - Неточные границы (fuzzy matching vs strict)
  - Hallucinated entity types (не из нашей схемы)
  - Медленный инференс
  - Parse failures JSON

---

## 📊 Сравнительная таблица архитектур

| Модель | Архитектура | Params | Entity-F1 | Precision | Recall | Inference | Training |
|--------|-------------|--------|-----------|-----------|--------|-----------|----------|
| **A: Baseline** | HashEmbedCNN | ~2M | 0.5789 | 0.5946 | 0.5641 | ~5ms | 4 min |
| **A+aug** | HashEmbedCNN + aug | ~2M | 0.5816 | 0.6196 | 0.5481 | ~5ms | 10 min |
| **B: Pretrained** | GloVe+CNN | ~20M | 0.5417 | 0.6620 | 0.4583 | ~5ms | 4 min |
| **B+aug (PRD)** | GloVe+CNN + aug | ~20M | **0.6040** | 0.6338 | 0.5769 | ~5ms | 10 min |
| **C: Hybrid** | GloVe+CNN+Ruler | ~20M | 0.5682 | **0.6944** | 0.4808 | ~5ms | 4 min |
| **D: Transformer** | RoBERTa-base | 125M | ~0.58 | ~0.62 | ~0.55 | ~100ms | 30+ min |
| **E: LLM 0.6B** | Qwen3 decoder | 0.6B | 0.1690 | 0.1828 | 0.1571 | ~11s | N/A |
| **E: LLM 1.7B** | Qwen3 decoder | 1.7B | 0.2194 | 0.1822 | 0.2756 | ~14s | N/A |
| **E: LLM 4B** | Qwen3 decoder | 4B | 0.2302 | 0.1839 | 0.3077 | ~25s | N/A |
| **Baseline (regex)** | Heuristics | — | 0.05-0.15 | 0.80+ | 0.03-0.10 | <1ms | N/A |

---

## 🏆 Production модель (B+aug) — детальная схема

```
┌─────────────────────────────────────────────────────────────────┐
│            PRODUCTION MODEL: B+aug (PRD)                         │
│            Entity-F1 = 0.6040                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CONFIGURATION (from conf/config.yaml)                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  model.type: pretrained                                     │ │
│  │  model.base_spacy_model: en_core_web_lg                     │ │
│  │  training.n_epochs: 30                                      │ │
│  │  training.dropout: 0.3                                      │ │
│  │  training.patience: 5                                       │ │
│  │  data.use_augmentation: true                                │ │
│  │  data.random_state: 42                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  TRAINING DATA                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Original train: 179 resumes                                │ │
│  │  Augmented (4x): 716 resumes                                │ │
│  │  Augmentation techniques:                                   │ │
│  │    - Entity swap (same-type replacement)                    │ │
│  │    - Case variation (random casing)                         │ │
│  │    - Skill synonyms (ML ↔ Machine Learning)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PIPELINE ARCHITECTURE                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. Tokenizer (en_core_web_lg)                              │ │
│  │     - Whitespace rules + prefix/suffix/infix               │ │
│  │     - Exception list for special tokens                    │ │
│  │                                                             │ │
│  │  2. tok2vec (Pretrained)                                    │ │
│  │     - Lookup table: 685k words × 300d (GloVe)              │ │
│  │     - CNN for OOV words                                    │ │
│  │     - Frozen weights (not updated during training)         │ │
│  │                                                             │ │
│  │  3. NER (Trainable)                                         │ │
│  │     - Input: 300d token vectors from tok2vec               │ │
│  │     - CNN layers: kernel sizes [3, 4, 5]                   │ │
│  │     - Output: BIO tags for 10 entity types                 │ │
│  │     - Labels: Name, Email, Location, College, Degree,      │ │
│  │               GradYear, Companies, Designation, Skills,    │ │
│  │               YearsOfExperience                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  MLFLOW INTEGRATION                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Logged params: 15+ hyperparameters                         │ │
│  │  Logged metrics: per-epoch + final test metrics             │ │
│  │  Artifacts (in MinIO S3):                                   │ │
│  │    - model/ (SpaCy model directory)                         │ │
│  │    - learning_curve.png                                     │ │
│  │    - confusion_matrix.png                                   │ │
│  │    - prediction_examples.json                               │ │
│  │    - config.yaml                                            │ │
│  │  Tags: stage=PRD, model_type=pretrained, augmentation=true  │ │
│  │  Registered model: resume_ner_prod                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  INFERENCE (via demo_cli.py or ASGI API)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. Load model from MLflow: mlflow.spacy.load_model(...)   │ │
│  │  2. Process text: nlp(text)                                 │ │
│  │  3. Extract entities: doc.ents                              │ │
│  │  4. Return: [(text, label, start, end), ...]               │ │
│  │                                                             │ │
│  │  Example output:                                            │ │
│  │  [                                                          │ │
│  │    {"text": "John Smith", "label": "Name",                 │ │
│  │     "start": 1, "end": 11},                                │ │
│  │    {"text": "Senior Software Engineer",                    │ │
│  │     "label": "Designation", "start": 12, "end": 38},       │ │
│  │    ...                                                      │ │
│  │  ]                                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 MLflow пайплайн (Checkpoint 7)

```
┌──────────────────────────────────────────────────────────────────┐
│                    MLFLOW EXPERIMENT FLOW                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐                                              │
│  │ train_cli.py   │                                              │
│  │ (or notebook)  │                                              │
│  └───────┬────────┘                                              │
│          │                                                        │
│          │ 1. Start MLflow run                                    │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  mlflow.start_run(run_name="B+aug_pretrained_final")         ││
│  └───────┬──────────────────────────────────────────────────────┘│
│          │                                                        │
│          │ 2. Log parameters                                      │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  mlflow.log_params({                                         ││
│  │    "model_type": "pretrained",                               ││
│  │    "n_epochs": 30, "dropout": 0.3, ...                       ││
│  │  })                                                          ││
│  └───────┬──────────────────────────────────────────────────────┘│
│          │                                                        │
│          │ 3. Train model + log per-epoch metrics                 │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  for epoch in range(n_epochs):                               ││
│  │    train(...)                                                ││
│  │    mlflow.log_metric("train_loss", loss, step=epoch)         ││
│  │    mlflow.log_metric("dev_f1", dev_f1, step=epoch)           ││
│  └───────┬──────────────────────────────────────────────────────┘│
│          │                                                        │
│          │ 4. Evaluate on test                                    │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  test_metrics = calculate_entity_metrics(nlp, test_data)     ││
│  │  mlflow.log_metrics({                                        ││
│  │    "test_entity_f1": test_metrics["f1"],                     ││
│  │    "test_entity_precision": test_metrics["precision"],       ││
│  │    "test_entity_recall": test_metrics["recall"],             ││
│  │    ...                                                       ││
│  │  })                                                          ││
│  └───────┬──────────────────────────────────────────────────────┘│
│          │                                                        │
│          │ 5. Log artifacts (to MinIO S3)                         │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  mlflow.log_artifact("learning_curve.png")                   ││
│  │  mlflow.log_artifact("confusion_matrix.png")                 ││
│  │  mlflow.log_artifact("prediction_examples.json")             ││
│  └───────┬──────────────────────────────────────────────────────┘│
│          │                                                        │
│          │ 6. Log & register model                                │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  mlflow.spacy.log_model(                                     ││
│  │    spacy_model=nlp,                                          ││
│  │    artifact_path="model",                                    ││
│  │    registered_model_name="resume_ner_prod"                   ││
│  │  )                                                           ││
│  └───────┬──────────────────────────────────────────────────────┘│
│          │                                                        │
│          │ 7. Tag as production                                   │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  mlflow.set_tag("stage", "PRD")                              ││
│  │  mlflow.set_tag("model_type", "pretrained")                  ││
│  └───────┬──────────────────────────────────────────────────────┘│
│          │                                                        │
│          │ 8. Run complete                                        │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  Run ID: abc123...                                           ││
│  │  URL: http://localhost:5001/#/experiments/0/runs/abc123...   ││
│  │  Model: s3://mlflow-artifacts/0/abc123.../model              ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Структура проекта (MLflow-интеграция)

```
resume_ner/
├── mlflow/
│   ├── docker-compose.mlflow.yaml    # MLflow + MinIO + Postgres
│   └── README.md                      # Инструкции по запуску
│
├── conf/
│   └── config.yaml                    # Hydra конфиг для train_cli.py
│
├── train_cli.py                       # CLI обучение с MLflow
├── demo_cli.py                        # CLI инференс PRD-модели
│
├── notebooks/
│   ├── DL_Experiments.ipynb          # + секции 7.1-7.6 (MLflow, анализ ошибок)
│   └── DL_Demonstration.ipynb        # Чистый ноутбук для демо
│
├── app/
│   └── ml/
│       ├── ner_service.py            # NERService для ASGI API
│       └── schema.py                 # Pydantic схемы
│
└── pyproject.toml                     # + mlflow, boto3, hydra-core
```

---

## 🎯 Ключевые улучшения Чекпоинта 7

| Улучшение | Описание | Benefit |
|-----------|----------|---------|
| **MLflow Tracking** | Централизованное логирование экспериментов | Воспроизводимость, сравнение runs |
| **MinIO S3** | Хранение артефактов (модели, графики) | Версионирование, доступность |
| **PostgreSQL** | Backend store для метаданных | Быстрый поиск, фильтрация runs |
| **Model Registry** | Регистрация модели с тегом `PRD` | Production-ready deployment |
| **Hydra CLI** | Конфигурация через YAML + overrides | Гибкость, документирование params |
| **Анализ ошибок** | 4 категории ошибок + 20 примеров | Понимание failure modes |
| **Robustness check** | 5 видов возмущений входа | Оценка устойчивости модели |

---

## 🚀 Как использовать

```bash
# 1. Поднять инфраструктуру
docker-compose -f mlflow/docker-compose.mlflow.yaml up -d

# 2. Обучить модель с MLflow логированием
python train_cli.py
python train_cli.py training.n_epochs=50  # override параметров

# 3. Демонстрация инференса
python demo_cli.py

# 4. UI для просмотра результатов
# MLflow: http://localhost:5001
# MinIO:  http://localhost:9001 (minioadmin/minioadmin)
```

---

**Итог:** В Чекпоинте 7 модель эволюционирует от простого ML-эксперимента до **production-ready решения** с полным трекингом экспериментов, версионированием артефактов и воспроизводимостью обучения. Лучшая модель (**B+aug**) достигает **Entity-F1 = 0.6040** и регистрируется в MLflow Model Registry с тегом `stage=PRD`.