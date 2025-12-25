#!/usr/bin/env python

import shutil
from pathlib import Path

model_path = Path("models/resume_ner_model")
if model_path.exists():
    shutil.rmtree(model_path)
    print(f"Модель удалена: {model_path}")
else:
    print("Модель не найдена")

print("При следующем запуске модель будет обучена заново")