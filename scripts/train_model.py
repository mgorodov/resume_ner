#!/usr/bin/env python
"""
Скрипт для обучения модели
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.ml.model import ResumeNERModel


def main():
    """Обучение модели"""
    print("Обучение модели NER для резюме...")
    
    model = ResumeNERModel()
    
    # Проверяем, есть ли уже модель
    if model.load_model():
        print("Модель уже существует.")
        response = input("Хотите переобучить модель? (y/N): ")
        if response.lower() == 'y':
            # Для переобучения нужно удалить старую модель
            import shutil
            model_path = Path("models/resume_ner_model")
            if model_path.exists():
                shutil.rmtree(model_path)
                print("Старая модель удалена")
            
            # Создаем новую модель
            model = ResumeNERModel()
            success = model.train_model(n_iter=15)
            if success:
                print("Модель успешно переобучена")
            else:
                print("Ошибка при переобучении модели")
        else:
            print("Используется существующая модель")
    else:
        # Обучаем новую модель
        success = model.train_model(n_iter=15)
        if success:
            print("Модель успешно обучена")
        else:
            print("Ошибка при обучении модели")


if __name__ == "__main__":
    main()