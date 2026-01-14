#!/bin/bash
# Setup virtual environment with Python 3.13 and run evaluation

set -e  # Exit on error

echo "======================================================================"
echo "НАСТРОЙКА ОКРУЖЕНИЯ И ЗАПУСК ОЦЕНКИ МОДЕЛИ"
echo "======================================================================"

# Check if Python 3.13 is available
if command -v python3.13 &> /dev/null; then
    PYTHON_CMD="python3.13"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
else
    echo "❌ Не найден подходящий Python (нужен 3.10-3.13)"
    echo "   Python 3.14 слишком новый для SpaCy!"
    exit 1
fi

echo "✓ Используется: $PYTHON_CMD ($(${PYTHON_CMD} --version))"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Создание виртуального окружения..."
    ${PYTHON_CMD} -m venv venv
    echo "✓ Виртуальное окружение создано"
else
    echo "✓ Виртуальное окружение уже существует"
fi

# Activate virtual environment
echo ""
echo "🔌 Активация виртуального окружения..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Обновление pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "📚 Установка зависимостей..."
echo "   Это может занять несколько минут..."
pip install spacy scikit-learn pandas kaggle --quiet

echo ""
echo "✓ Все зависимости установлены!"

# Download dataset if it doesn't exist
if [ ! -f "datasets/dataturks/Entity Recognition in Resumes.json" ]; then
    echo ""
    echo "📥 Загрузка датасета с Kaggle..."
    
    # Check if Kaggle credentials exist
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo ""
        echo "⚠️  Не найдены учетные данные Kaggle!"
        echo "   Для загрузки датасета нужно:"
        echo "   1. Зарегистрироваться на kaggle.com"
        echo "   2. Создать API токен (Account -> Create New API Token)"
        echo "   3. Сохранить kaggle.json в ~/.kaggle/"
        echo ""
        echo "   Или скачайте датасет вручную из notebooks/EDA.ipynb"
        exit 1
    fi
    
    mkdir -p datasets/dataturks
    python -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('dataturks/resume-entities-for-ner', path='datasets/dataturks', quiet=False, unzip=True)
print('✓ Датасет загружен!')
"
else
    echo "✓ Датасет уже существует"
fi

# Run evaluation
echo ""
echo "======================================================================"
echo "🚀 ЗАПУСК ОЦЕНКИ МОДЕЛИ"
echo "======================================================================"
echo ""

python evaluate_model.py

echo ""
echo "======================================================================"
echo "✅ ГОТОВО!"
echo "======================================================================"
echo ""
echo "💡 Для повторного запуска:"
echo "   source venv/bin/activate"
echo "   python evaluate_model.py"
