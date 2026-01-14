#!/usr/bin/env python3
"""
Evaluate pre-trained SpaCy NER model
Loads existing model and calculates comprehensive metrics
"""

from pathlib import Path
import json
import re
import math
import random

print("="*70)
print("ОЦЕНКА ОБУЧЕННОЙ SPACY NER МОДЕЛИ")
print("="*70)

try:
    import spacy
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report
    )
    import pandas as pd
except ImportError as e:
    print(f"\n❌ Ошибка импорта: {e}")
    print("\nУстановите зависимости:")
    print("  pip install spacy scikit-learn pandas")
    exit(1)

# Configuration
MODEL_PATH = Path("resume_ner_model")
DATASET_DIR = Path("datasets/dataturks")
DATASET_PATH = DATASET_DIR / "Entity Recognition in Resumes.json"
TEST_SIZE = 0.1
RANDOM_STATE = 42

# ============================================================================
# DATA LOADING
# ============================================================================

def convert_dataturks_to_spacy(dataturks_json_filepath):
    """Convert Dataturks JSON format to SpaCy training format"""
    nlp = spacy.blank("en")
    training_data = []
    
    with open(dataturks_json_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        data = json.loads(line)
        text = data['content'].replace("\n", " ").replace("\t", " ").strip()
        doc = nlp.make_doc(text)
        
        entities = []
        data_annotations = data['annotation']
        
        if data_annotations is not None:
            for annotation in data_annotations:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]
                
                for label in labels:
                    point_text = point['text'].strip()
                    if not point_text:
                        continue
                    
                    matches = list(re.finditer(re.escape(point_text), text))
                    if matches:
                        match_start, match_end = matches[0].span()
                        entities.append((match_start, match_end, label))
        
        training_data.append((text, {"entities": entities}))
    
    return training_data


def trim_entity_spans(data):
    """Trim whitespace from entity spans"""
    invalid_span_tokens = re.compile(r'\s')
    cleaned_data = []
    
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    
    return cleaned_data


def train_test_split(data, test_size, random_state):
    """Split data into train and test sets"""
    random.Random(random_state).shuffle(data)
    test_idx = len(data) - math.floor(test_size * len(data))
    return data[test_idx:]


print("\n📂 Загрузка данных...")
data = convert_dataturks_to_spacy(DATASET_PATH)
data = trim_entity_spans(data)
test_data = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"  ✓ Тестовая выборка: {len(test_data)} резюме")

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"\n🤖 Загрузка модели из {MODEL_PATH}...")
try:
    nlp = spacy.load(MODEL_PATH)
    print(f"  ✓ Модель загружена!")
    print(f"  ✓ Распознаваемых типов сущностей: {len(nlp.get_pipe('ner').labels)}")
    print(f"  ✓ Типы: {', '.join(nlp.get_pipe('ner').labels)}")
except Exception as e:
    print(f"  ❌ Ошибка загрузки модели: {e}")
    print("\n  Запустите сначала обучение модели в notebooks/ML.ipynb")
    exit(1)

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_token_level_metrics(nlp, test_data):
    """Calculate token-level metrics"""
    all_true_labels = []
    all_pred_labels = []
    
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        true_entities = annotations.get("entities", [])
        
        true_labels = ['O'] * len(doc)
        for start, end, label in true_entities:
            entity_tokens = []
            for token in doc:
                if token.idx >= start and token.idx + len(token.text) <= end:
                    entity_tokens.append(token)
            
            if entity_tokens:
                for i, token in enumerate(entity_tokens):
                    if len(entity_tokens) == 1:
                        true_labels[token.i] = f'U-{label}'
                    elif i == 0:
                        true_labels[token.i] = f'B-{label}'
                    elif i == len(entity_tokens) - 1:
                        true_labels[token.i] = f'L-{label}'
                    else:
                        true_labels[token.i] = f'I-{label}'
        
        pred_doc = nlp(text)
        pred_labels = [token.ent_iob_ + ('-' + token.ent_type_ if token.ent_type_ else '')
                      for token in pred_doc]
        
        min_len = min(len(true_labels), len(pred_labels))
        all_true_labels.extend(true_labels[:min_len])
        all_pred_labels.extend(pred_labels[:min_len])
    
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels,
        all_pred_labels,
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_labels': all_true_labels,
        'pred_labels': all_pred_labels
    }


def calculate_entity_level_metrics(nlp, test_data):
    """Calculate entity-level metrics (strict matching)"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for text, annotations in test_data:
        true_entities = annotations.get("entities", [])
        true_entity_set = set([(start, end, label) for start, end, label in true_entities])
        
        pred_doc = nlp(text)
        pred_entity_set = set([(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents])
        
        true_positives += len(true_entity_set & pred_entity_set)
        false_positives += len(pred_entity_set - true_entity_set)
        false_negatives += len(true_entity_set - pred_entity_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


print("\n📊 Расчет метрик качества...")
token_metrics = calculate_token_level_metrics(nlp, test_data)
entity_metrics = calculate_entity_level_metrics(nlp, test_data)

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("МЕТРИКИ КАЧЕСТВА (Token-level)")
print("="*70)
print(f"Accuracy:  {token_metrics['accuracy']:.4f} ({token_metrics['accuracy']*100:.2f}%)")
print(f"Precision: {token_metrics['precision']:.4f} ({token_metrics['precision']*100:.2f}%)")
print(f"Recall:    {token_metrics['recall']:.4f} ({token_metrics['recall']*100:.2f}%)")
print(f"F1-Score:  {token_metrics['f1']:.4f} ({token_metrics['f1']*100:.2f}%)")

print("\n" + "="*70)
print("МЕТРИКИ КАЧЕСТВА (Entity-level - Strict Matching)")
print("="*70)
print(f"Precision: {entity_metrics['precision']:.4f} ({entity_metrics['precision']*100:.2f}%)")
print(f"Recall:    {entity_metrics['recall']:.4f} ({entity_metrics['recall']*100:.2f}%)")
print(f"F1-Score:  {entity_metrics['f1']:.4f} ({entity_metrics['f1']*100:.2f}%)")
print(f"\nДеталировка:")
print(f"  True Positives:  {entity_metrics['true_positives']}")
print(f"  False Positives: {entity_metrics['false_positives']}")
print(f"  False Negatives: {entity_metrics['false_negatives']}")

# Summary table
print("\n" + "="*70)
print("ИТОГОВАЯ ТАБЛИЦА МЕТРИК")
print("="*70)

summary_data = {
    'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Token-level': [
        f"{token_metrics['accuracy']:.4f}",
        f"{token_metrics['precision']:.4f}",
        f"{token_metrics['recall']:.4f}",
        f"{token_metrics['f1']:.4f}"
    ],
    'Entity-level': [
        'N/A',
        f"{entity_metrics['precision']:.4f}",
        f"{entity_metrics['recall']:.4f}",
        f"{entity_metrics['f1']:.4f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("ВЫВОДЫ")
print("="*70)
total_entities = entity_metrics['true_positives'] + entity_metrics['false_negatives']
print(f"✓ Протестировано на {len(test_data)} резюме")
print(f"✓ Token-level accuracy: {token_metrics['accuracy']*100:.2f}%")
print(f"✓ Entity-level F1: {entity_metrics['f1']*100:.2f}%")
print(f"✓ Корректно распознано: {entity_metrics['true_positives']} из {total_entities} сущностей")
print(f"✓ Процент правильных: {entity_metrics['true_positives']/total_entities*100:.1f}%")

# Examples
print("\n" + "="*70)
print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("="*70)

test_samples = random.sample(test_data, min(2, len(test_data)))
for idx, (text, annotations) in enumerate(test_samples, 1):
    print(f"\nПРИМЕР {idx}")
    print("-" * 70)
    text_preview = text[:150] + "..." if len(text) > 150 else text
    print(f"Текст: {text_preview}")
    
    true_entities = annotations.get("entities", [])
    print(f"\n📌 Истинные ({len(true_entities)} сущностей, показано 5):")
    for start, end, label in true_entities[:5]:
        entity_text = text[start:end]
        print(f"  • {label}: '{entity_text[:40]}{'...' if len(entity_text) > 40 else ''}'")
    
    pred_doc = nlp(text)
    print(f"\n🤖 Предсказано ({len(pred_doc.ents)} сущностей, показано 5):")
    for ent in list(pred_doc.ents)[:5]:
        print(f"  • {ent.label_}: '{ent.text[:40]}{'...' if len(ent.text) > 40 else ''}'")

print("\n" + "="*70)
print("✅ ОЦЕНКА ЗАВЕРШЕНА!")
print("="*70)
