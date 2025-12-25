import time
from typing import Dict, Any, Optional
from app.ml.model import ResumeNERModel


class NERService:
    def __init__(self):
        self.model = ResumeNERModel()
        
        if not self.model.is_ready():
            print("Модель не найдена, начинаем обучение...")
            self.model.train_model(n_iter=20)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            entities = self.model.predict(text)
            
            return {
                "success": True,
                "entities": entities,
                "processing_time": time.time() - start_time,
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
        start_time = time.time()
        
        try:
            # TODO: Реализовать обработку изображения
            print("Image processing not implemented yet")
            return {
                "success": True,
                "message": "Image processing not implemented yet",
                "description": description,
                "processing_time": time.time() - start_time,
                "image_size": len(image_bytes)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }