#!/bin/bash
# Quick evaluation script - runs evaluation from notebook directory

echo "======================================================================"
echo "🚀 БЫСТРАЯ ОЦЕНКА МОДЕЛИ (из Jupyter окружения)"
echo "======================================================================"

cd notebooks

# Check if jupyter/ipython kernel is available  
if command -v jupyter &> /dev/null; then
    echo ""
    echo "✓ Jupyter найден, запускаем оценку..."
    echo ""
    
    # Run the evaluation cells from ML.ipynb
    jupyter nbconvert --to notebook --execute ML.ipynb --output ML_evaluated.ipynb 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Оценка завершена!"
        echo "📊 Результаты сохранены в: notebooks/ML_evaluated.ipynb"
    else
        echo "⚠️  Ошибка при запуске через jupyter"
        echo ""
        echo "💡 Откройте вручную: notebooks/ML.ipynb"
        echo "   И запустите ячейки 11-15 для получения метрик"
    fi
else
    echo ""
    echo "⚠️  Jupyter не найден"
    echo ""
    echo "📝 ИНСТРУКЦИЯ:"
    echo "1. Откройте notebooks/ML.ipynb в Jupyter"
    echo "2. Запустите ячейки 11-15"
    echo "3. Получите все метрики!"
    echo ""
    echo "Или используйте: ./setup_and_run.sh"
    echo "(создаст виртуальное окружение и установит зависимости)"
fi

echo ""
echo "======================================================================"
