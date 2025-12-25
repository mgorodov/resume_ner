"""
Сервисный слой для работы с NER моделью
"""
import time
import base64
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image

from app.ml.model import ResumeNERModel


class NERService:
    def __init__(self, model_path: str = None):
        self.model = ResumeNERModel(model_path)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Обработка текстового запроса"""
        start_time = time.time()
        
        try:
            # Извлечение сущностей
            entities = self.model.predict(text)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "entities": entities,
                "processing_time": processing_time,
                "text_length": len(text),
                "entity_count": len(entities)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_image(self, image_bytes: bytes, description: str = None) -> Dict[str, Any]:
        """Обработка изображения"""
        start_time = time.time()
        
        try:
            # Для демонстрации - просто анализируем размер и формат
            image = Image.open(BytesIO(image_bytes))
            
            # Создаем обработанное изображение
            processed_image = self._add_watermark(image)
            
            # Конвертируем в base64
            buffered = BytesIO()
            processed_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "image_result": img_str,
                "processing_time": processing_time,
                "image_size": len(image_bytes),
                "dimensions": f"{image.width}x{image.height}",
                "format": image.format,
                "description": description
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _add_watermark(self, image: Image.Image) -> Image.Image:
        """Добавление водяного знака на изображение"""
        from PIL import ImageDraw, ImageFont
        
        # Создаем копию изображения
        result = image.copy()
        
        # Добавляем водяной знак
        try:
            draw = ImageDraw.Draw(result)
            font = ImageFont.load_default()
            
            watermark_text = "Resume NER Service"
            text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Позиционируем водяной знак
            position = (
                image.width - text_width - 10,
                image.height - text_height - 10
            )
            
            draw.text(position, watermark_text, fill=(255, 255, 255, 128), font=font)
        except:
            # Если не удалось добавить водяной знак, просто возвращаем оригинал
            pass
        
        return result