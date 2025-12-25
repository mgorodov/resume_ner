"""
Модель для NER из резюме
"""
import json
import re
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import spacy
from spacy.training import Example


class ResumeNERModel:
    def __init__(self, model_path: str = None):
        self.nlp = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.train_model()
    
    def load_model(self, model_path: str):
        """Загрузка предобученной модели"""
        self.nlp = spacy.load(model_path)
    
    def save_model(self, model_path: str):
        """Сохранение модели"""
        if self.nlp:
            self.nlp.to_disk(model_path)
    
    def _load_dataset(self):
        """Загрузка датасета"""
        DATASET_DIR = Path.cwd() / "datasets" / "dataturks"
        DATASET_PATH = DATASET_DIR / "Entity Recognition in Resumes.json"
        
        # Скачивание датасета если его нет
        if not DATASET_PATH.exists():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'dataturks/resume-entities-for-ner', 
                path=DATASET_DIR, 
                quiet=False, 
                unzip=True
            )
        
        return DATASET_PATH
    
    def _convert_dataturks_to_spacy(self, dataturks_JSON_FilePath):
        """Конвертация данных из формата Dataturks в Spacy"""
        nlp = spacy.blank("en")
        training_data = []
        
        with open(dataturks_JSON_FilePath, 'r', encoding='utf-8') as f:
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
    
    def _trim_entity_spans(self, data: list) -> list:
        """Очистка сущностей от пробелов"""
        invalid_span_tokens = re.compile(r'\s')
        cleaned_data = []
        for text, annotations in data:
            entities = annotations['entities']
            valid_entities = []
            for start, end, label in entities:
                valid_start = start
                valid_end = end
                while valid_start < len(text) and invalid_span_tokens.match(
                        text[valid_start]):
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(
                        text[valid_end - 1]):
                    valid_end -= 1
                valid_entities.append([valid_start, valid_end, label])
            cleaned_data.append([text, {'entities': valid_entities}])
        return cleaned_data
    
    def _train_test_split(self, data, test_size, random_state):
        """Разделение на train/test"""
        random.Random(random_state).shuffle(data)
        test_idx = len(data) - math.floor(test_size * len(data))
        train_set = data[0: test_idx]
        test_set = data[test_idx: ]
        return train_set, test_set
    
    def _filter_overlapping_entities(self, data):
        """Удаление перекрывающихся сущностей"""
        cleaned_data = []
        
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
                    print(f"Warning: Skipping overlapping entity '{text[start:end]}' ({label})")
                    continue
                
                non_overlapping.append((start, end, label))
                prev_end = end
            
            cleaned_data.append((text, {"entities": non_overlapping}))
        
        return cleaned_data
    
    def train_model(self, n_iter: int = 10, test_size: float = 0.1):
        """Обучение модели"""
        # Загрузка и подготовка данных
        dataset_path = self._load_dataset()
        data = self._convert_dataturks_to_spacy(dataset_path)
        data = self._trim_entity_spans(data)
        train_data, test_data = self._train_test_split(data, test_size=test_size, random_state=42)
        train_data = self._filter_overlapping_entities(train_data)
        
        # Создание и обучение модели
        nlp = spacy.blank('en')
        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner', last=True)
        
        # Добавление лейблов
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])
        
        # Обучение
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.initialize()
            
            for itn in range(n_iter):
                print(f"Starting iteration {itn + 1}/{n_iter}")
                random.shuffle(train_data)
                losses = {}
                
                for text, annotations in train_data:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    
                    nlp.update(
                        [example],
                        drop=0.2,
                        sgd=optimizer,
                        losses=losses
                    )
                print(f"Losses: {losses}")
        
        self.nlp = nlp
        return nlp
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Предсказание сущностей из текста"""
        if not self.nlp:
            raise ValueError("Модель не обучена. Сначала вызовите train_model()")
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def evaluate(self, test_data=None):
        """Оценка модели"""
        if not self.nlp:
            raise ValueError("Модель не обучена")
        
        if test_data is None:
            dataset_path = self._load_dataset()
            data = self._convert_dataturks_to_spacy(dataset_path)
            data = self._trim_entity_spans(data)
            _, test_data = self._train_test_split(data, test_size=0.1, random_state=42)
            test_data = self._filter_overlapping_entities(test_data)
        
        from sklearn.metrics import accuracy_score
        
        all_true_labels = []
        all_pred_labels = []
        
        for text, annotations in test_data:
            doc = self.nlp.make_doc(text)
            true_entities = annotations.get("entities", [])
            
            true_labels = ['O'] * len(doc)
            for start, end, label in true_entities:
                entity_text = text[start:end]
                
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
            
            pred_doc = self.nlp(text)
            pred_labels = [
                token.ent_iob_ + ('-' + token.ent_type_ if token.ent_type_ else '') 
                for token in pred_doc
            ]
            
            pred_labels = [
                label.replace('B-', 'B-').replace('I-', 'I-').replace('O', 'O') 
                for label in pred_labels
            ]
            
            min_len = min(len(true_labels), len(pred_labels))
            all_true_labels.extend(true_labels[:min_len])
            all_pred_labels.extend(pred_labels[:min_len])
        
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        return accuracy