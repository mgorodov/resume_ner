#!/usr/bin/env python
"""
Проверка модели
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.ml.model import ResumeNERModel


def main():
    print("Проверка модели...")
    
    model = ResumeNERModel()
    
    if model.is_ready():
        print("Модель загружена и готова к использованию")
        test_text = "John Smith worked at Google from 2018 to 2023 as a Software Engineer."
        entities = model.predict(test_text)
        
        print(f"\nТестовый текст: {test_text}")
        print(f"Найдено сущностей: {len(entities)}")
        
        for ent in entities:
            print(f"  - {ent['text']} ({ent['label']})")
    else:
        print("Модель не загружена")
        
        print("Попытка обучения...")
        if model.train_model(n_iter=20):
            print("Модель обучена и сохранена")
        else:
            print("Не удалось обучить модель")


if __name__ == "__main__":
    main()