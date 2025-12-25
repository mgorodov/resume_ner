#!/usr/bin/env python
"""
Скрипт для обучения модели
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.ml.model import ResumeNERModel


def main():
    """Основная функция обучения"""
    print("Обучение модели NER для резюме...")
    
    # Инициализация модели
    model = ResumeNERModel()
    
    # Проверяем, есть ли уже модель
    if model.load_model():
        print("Модель уже существует. Для переобучения удалите папку models/")
        response = input("Хотите переобучить модель? (y/N): ")
        if response.lower() == 'y':
            print("Переобучение модели...")
            # Для переобучения нужно удалить модель
            import shutil
            model_path = Path("models/resume_ner_model")
            if model_path.exists():
                shutil.rmtree(model_path)
            # Создаем новую модель
            model = ResumeNERModel()
            model.train_model(n_iter=10)
            print("Модель переобучена")
        else:
            print("Используется существующая модель")
    else:
        # Обучаем новую модель
        print("Обучение новой модели...")
        model.train_model(n_iter=10)
        print("Модель обучена")
    
    return model


if __name__ == "__main__":
    main()