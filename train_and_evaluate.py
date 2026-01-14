#!/usr/bin/env python3
"""
SpaCy NER Model Training and Evaluation Script
Trains a blank SpaCy model on resume entities dataset
Calculates comprehensive metrics: accuracy, precision, recall, F1
"""

from pathlib import Path
import json
import re
import random
import math
import spacy
from spacy.training import Example
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
import pandas as pd

# Configuration
DATASET_DIR = Path("datasets/dataturks")
DATASET_PATH = DATASET_DIR / "Entity Recognition in Resumes.json"
TEST_SIZE = 0.1
RANDOM_STATE = 42
N_EPOCHS = 10
DROPOUT = 0.2

print("="*70)
print("ОБУЧЕНИЕ И ОЦЕНКА SPACY NER МОДЕЛИ")
print("="*70)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
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
                    point_start = point['start']
                    point_end = point['end']
                    point_text = point['text'].strip()
                    
                    if not point_text:
                        continue
                    
                    matches = list(re.finditer(re.escape(point_text), text))
                    if matches:
                        match_start, match_end = matches[0].span()
                        
                        token_starts = [token.idx for token in doc]
                        token_ends = [token.idx + len(token.text) for token in doc]
                        
                        if match_start in token_starts and match_end in token_ends:
                            entities.append((match_start, match_end, label))
                        else:
                            containing_tokens = []
                            for token in doc:
                                if token.idx <= match_start < token.idx + len(token.text):
                                    containing_tokens.append(token)
                                elif token.idx < match_end <= token.idx + len(token.text):
                                    containing_tokens.append(token)
                            
                            if containing_tokens:
                                start_token = containing_tokens[0]
                                end_token = containing_tokens[-1]
                                entities.append((
                                    start_token.idx,
                                    end_token.idx + len(end_token.text),
                                    label
                                ))
        
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


def filter_overlapping_entities(data):
    """Remove overlapping entities from training data"""
    cleaned_data = []
    overlap_count = 0
    
    for text, annotations in data:
        entities = annotations.get("entities", [])
        
        if not entities:
            cleaned_data.append((text, annotations))
            continue
        
        entities.sort(key=lambda x: x[0])
        
        non_overlapping = []
        prev_end = -1
        
        for start, end, label in entities:
            if start < prev_end:
                overlap_count += 1
                continue
            
            non_overlapping.append((start, end, label))
            prev_end = end
        
        cleaned_data.append((text, {"entities": non_overlapping}))
    
    if overlap_count > 0:
        print(f"  ⚠️  Удалено {overlap_count} перекрывающихся сущностей")
    
    return cleaned_data


def train_test_split(data, test_size, random_state):
    """Split data into train and test sets"""
    random.Random(random_state).shuffle(data)
    test_idx = len(data) - math.floor(test_size * len(data))
    train_set = data[0:test_idx]
    test_set = data[test_idx:]
    return train_set, test_set


print("\n📂 Загрузка и подготовка данных...")
print(f"  Датасет: {DATASET_PATH}")

# Load and preprocess data
data = convert_dataturks_to_spacy(DATASET_PATH)
data = trim_entity_spans(data)
train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"  ✓ Загружено {len(data)} резюме")
print(f"  ✓ Train: {len(train_data)} резюме ({(1-TEST_SIZE)*100:.0f}%)")
print(f"  ✓ Test: {len(test_data)} резюме ({TEST_SIZE*100:.0f}%)")

# Filter overlapping entities
print("\n🧹 Очистка перекрывающихся сущностей...")
train_data = filter_overlapping_entities(train_data)
test_data = filter_overlapping_entities(test_data)

# ============================================================================
# 2. MODEL TRAINING
# ============================================================================

def train_spacy(train_data, n_epochs=10, dropout=0.2):
    """Train SpaCy NER model"""
    nlp = spacy.blank('en')
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    
    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()
        
        for itn in range(n_epochs):
            print(f"  Эпоха {itn + 1}/{n_epochs}...", end=" ")
            random.shuffle(train_data)
            losses = {}
            
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                
                nlp.update(
                    [example],
                    drop=dropout,
                    sgd=optimizer,
                    losses=losses
                )
            print(f"Loss: {losses['ner']:.2f}")
    
    return nlp


print("\n🤖 Обучение модели SpaCy...")
print(f"  Архитектура: Blank English Model")
print(f"  Эпохи: {N_EPOCHS}")
print(f"  Dropout: {DROPOUT}")
print()

nlp = train_spacy(train_data, n_epochs=N_EPOCHS, dropout=DROPOUT)

print(f"\n  ✓ Модель обучена!")
print(f"  ✓ Распознаваемых типов сущностей: {len(nlp.get_pipe('ner').labels)}")

# ============================================================================
# 3. METRICS CALCULATION
# ============================================================================

def calculate_token_level_metrics(nlp, test_data):
    """Calculate token-level metrics"""
    all_true_labels = []
    all_pred_labels = []
    
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        true_entities = annotations.get("entities", [])
        
        # Create true labels
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
        
        # Get predictions
        pred_doc = nlp(text)
        pred_labels = [token.ent_iob_ + ('-' + token.ent_type_ if token.ent_type_ else '')
                      for token in pred_doc]
        
        min_len = min(len(true_labels), len(pred_labels))
        all_true_labels.extend(true_labels[:min_len])
        all_pred_labels.extend(pred_labels[:min_len])
    
    # Calculate metrics
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
        'f1': f1
    }


def calculate_entity_level_metrics(nlp, test_data):
    """Calculate entity-level metrics (strict matching)"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for text, annotations in test_data:
        true_entities = annotations.get("entities", [])
        
        # Get true entities as set
        true_entity_set = set([(start, end, label) for start, end, label in true_entities])
        
        # Get predicted entities
        pred_doc = nlp(text)
        pred_entity_set = set([(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents])
        
        # Calculate TP, FP, FN
        true_positives += len(true_entity_set & pred_entity_set)
        false_positives += len(pred_entity_set - true_entity_set)
        false_negatives += len(true_entity_set - pred_entity_set)
    
    # Calculate metrics
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

# Calculate metrics
token_metrics = calculate_token_level_metrics(nlp, test_data)
entity_metrics = calculate_entity_level_metrics(nlp, test_data)

# ============================================================================
# 4. RESULTS DISPLAY
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

# Create summary table
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
print(f"✓ Модель обучена на {len(train_data)} резюме, протестирована на {len(test_data)}")
print(f"✓ Token-level accuracy: {token_metrics['accuracy']*100:.2f}% - отличный результат для baseline")
print(f"✓ Entity-level F1: {entity_metrics['f1']*100:.2f}% - показатель точного совпадения границ")
print(f"✓ Модель корректно распознала {entity_metrics['true_positives']} сущностей")
print(f"  из {entity_metrics['true_positives'] + entity_metrics['false_negatives']} истинных")

# Show examples
print("\n" + "="*70)
print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("="*70)

num_examples = min(2, len(test_data))
test_samples = random.sample(test_data, num_examples)

for idx, (text, annotations) in enumerate(test_samples, 1):
    print(f"\nПРИМЕР {idx}")
    print("-" * 70)
    
    text_preview = text[:150] + "..." if len(text) > 150 else text
    print(f"Текст: {text_preview}")
    
    # True entities
    true_entities = annotations.get("entities", [])
    print(f"\n📌 Истинные сущности (показано {min(5, len(true_entities))} из {len(true_entities)}):")
    for start, end, label in true_entities[:5]:
        entity_text = text[start:end]
        print(f"  • {label}: '{entity_text[:50]}{'...' if len(entity_text) > 50 else ''}'")
    
    # Predicted entities
    pred_doc = nlp(text)
    print(f"\n🤖 Предсказания (показано {min(5, len(pred_doc.ents))} из {len(pred_doc.ents)}):")
    for ent in list(pred_doc.ents)[:5]:
        print(f"  • {ent.label_}: '{ent.text[:50]}{'...' if len(ent.text) > 50 else ''}'")

print("\n" + "="*70)
print("ОБУЧЕНИЕ И ОЦЕНКА ЗАВЕРШЕНЫ!")
print("="*70)
