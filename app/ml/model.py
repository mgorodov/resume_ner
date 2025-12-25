"""
Модель для NER из резюме с простым сохранением весов
"""
import json
import re
import random
import math
from pathlib import Path
from typing import List, Dict, Any

import spacy
from spacy.training import Example


class ResumeNERModel:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.nlp = None
        
        # Пытаемся загрузить модель
        self.load_model()
    
    def load_model(self) -> bool:
        """Загрузка модели из файла"""
        model_path = self.model_dir / "resume_ner_model"
        
        # Проверяем, есть ли сохраненная модель
        if model_path.exists():
            try:
                print(f"Загрузка модели из {model_path}")
                self.nlp = spacy.load(model_path)
                print("Модель загружена успешно")
                return True
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                return False
        else:
            print("Модель не найдена, будет создана новая")
            return False
    
    def save_model(self) -> bool:
        """Сохранение модели"""
        if not self.nlp:
            print("Модель не обучена, нечего сохранять")
            return False
        
        try:
            model_path = self.model_dir / "resume_ner_model"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.nlp.to_disk(model_path)
            print(f"Модель сохранена в {model_path}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения модели: {e}")
            return False
    
    def train_model(self, n_iter: int = 10) -> bool:
        """Обучение модели (только если еще не обучена)"""
        
        # Если модель уже загружена, не переобучаем
        if self.nlp:
            print("Модель уже загружена, обучение не требуется")
            return True
        
        print("Обучение новой модели...")
        
        try:
            # Загрузка датасета
            dataset_path = self._load_dataset()
            
            # Подготовка данных
            training_data = self._prepare_training_data(dataset_path)
            
            # Создание модели
            self.nlp = spacy.blank('en')
            if 'ner' not in self.nlp.pipe_names:
                ner = self.nlp.add_pipe('ner', last=True)
            
            # Добавление лейблов
            for _, annotations in training_data:
                for ent in annotations.get("entities"):
                    ner.add_label(ent[2])
            
            # Обучение
            optimizer = self.nlp.initialize()
            
            for itn in range(n_iter):
                print(f"Итерация {itn + 1}/{n_iter}")
                random.shuffle(training_data)
                losses = {}
                
                for text, annotations in training_data:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    
                    self.nlp.update(
                        [example],
                        drop=0.2,
                        sgd=optimizer,
                        losses=losses
                    )
                print(f"Потери: {losses}")
            
            # Сохранение модели после обучения
            self.save_model()
            print("Модель обучена и сохранена")
            return True
            
        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            return False
    
    def _load_dataset(self) -> Path:
        """Загрузка датасета"""
        DATASET_DIR = Path.cwd() / "datasets" / "dataturks"
        DATASET_PATH = DATASET_DIR / "Entity Recognition in Resumes.json"
        
        if not DATASET_PATH.exists():
            print("Скачивание датасета...")
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
    
    def _prepare_training_data(self, dataset_path: Path):
        """Подготовка тренировочных данных"""
        training_data = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            data = json.loads(line)
            text = data['content'].replace("\n", " ").replace("\t", " ").strip()
            
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
                        
                        if point_text:
                            entities.append((point_start, point_end, label))
            
            training_data.append((text, {"entities": entities}))
        
        return training_data
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Предсказание сущностей"""
        if not self.nlp:
            # Пытаемся загрузить модель
            if not self.load_model():
                raise ValueError("Модель не обучена и не может быть загружена")
        
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
    
    def is_ready(self) -> bool:
        """Проверка, готова ли модель"""
        return self.nlp is not None