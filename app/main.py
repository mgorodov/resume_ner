from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import os

from app.core.config import settings
from app.database.session import engine, Base
from app.api.endpoints import router as api_router
from app.ml.ner_service import NERService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    # Startup
    print("Запуск Resume NER Service...")
    
    try:
        # Создание таблиц если их нет
        Base.metadata.create_all(bind=engine)
        print("База данных готова")
    except Exception as e:
        print(f"Ошибка базы данных: {e}")
    
    # Инициализация NER сервиса (модель загрузится или обучится автоматически)
    try:
        print("Инициализация NER сервиса...")
        app.state.ner_service = NERService()
        print("NER сервис готов")
    except Exception as e:
        print(f"Ошибка NER сервиса: {e}")
        app.state.ner_service = None
    
    yield
    
    # Shutdown
    print("Остановка сервиса...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware для измерения времени выполнения запросов
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Подключение роутеров
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Добро пожаловать в Resume NER Service",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    status = {
        "status": "healthy",
        "service": "Resume NER Service",
        "timestamp": time.time(),
        "database": "connected" if os.path.exists("resume_ner.db") else "not_connected"
    }
    
    # Проверка NER сервиса
    try:
        from app.ml.ner_service import NERService
        status["ner_service"] = "initialized"
    except:
        status["ner_service"] = "error"
    
    return status


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений"""
    import traceback
    print(f"Ошибка: {exc}")
    print(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Внутренняя ошибка сервера",
            "error": str(exc),
            "type": type(exc).__name__
        }
    )