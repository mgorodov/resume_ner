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
    print("🎯 Обучение модели NER для резюме...")
    
    # Инициализация модели
    model = ResumeNERModel()
    
    # Обучение
    print("📚 Начало обучения...")
    nlp = model.train_model(n_iter=10)
    
    # Сохранение модели
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "resume_ner_model"
    model.save_model(str(model_path))
    
    print(f"💾 Модель сохранена в {model_path}")
    
    # Оценка
    accuracy = model.evaluate()
    print(f"📊 Точность модели: {accuracy:.4f}")
    
    return model


if __name__ == "__main__":
    main()