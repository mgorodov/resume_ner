#!/usr/bin/env python
"""
Скрипт для обучения модели
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.ml.model import ResumeNERModel


def main():
    print("Обучение модели NER для резюме...")
    
    model = ResumeNERModel()
    if model.load_model():
        print("Модель уже существует.")
        response = input("Хотите переобучить модель? (y/N): ")
        if response.lower() == 'y':
            import shutil
            model_path = Path("models/resume_ner_model")
            if model_path.exists():
                shutil.rmtree(model_path)
                print("Старая модель удалена")
            
            model = ResumeNERModel()
            success = model.train_model(n_iter=20)
            if success:
                print("Модель успешно переобучена")
            else:
                print("Ошибка при переобучении модели")
        else:
            print("Используется существующая модель")
    else:
        success = model.train_model(n_iter=20)
        if success:
            print("Модель успешно обучена")
        else:
            print("Ошибка при обучении модели")


if __name__ == "__main__":
    main()