from sqlalchemy.orm import Session
from sqlalchemy import func, and_, text
import datetime
import json
from typing import List, Optional
from app.core import models


def create_request_history(
    db: Session,
    endpoint: str,
    method: str,
    processing_time: float,
    input_size: Optional[int] = None,
    input_type: Optional[str] = None,
    status: str = "success",
    response_data: Optional[dict] = None,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None
):
    # Для SQLite конвертируем dict в JSON строку
    response_data_json = json.dumps(response_data) if response_data else None
    
    db_history = models.RequestHistory(
        endpoint=endpoint,
        method=method,
        processing_time=processing_time,
        input_size=input_size,
        input_type=input_type,
        status=status,
        response_data=response_data_json,  # Используем JSON строку
        user_agent=user_agent,
        ip_address=ip_address
    )
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history


def get_all_history(db: Session, skip: int = 0, limit: int = 100):
    results = db.query(models.RequestHistory).order_by(models.RequestHistory.timestamp.desc()).offset(skip).limit(limit).all()
    
    # Конвертируем JSON строки обратно в dict
    for result in results:
        if result.response_data and isinstance(result.response_data, str):
            try:
                result.response_data = json.loads(result.response_data)
            except:
                pass
    
    return results


def delete_all_history(db: Session):
    db.query(models.RequestHistory).delete()
    db.commit()


def get_history_stats(db: Session, days: int = 7):
    """Get statistics for the last N days"""
    # Для SQLite используем другой синтаксис для даты
    if db.bind.dialect.name == 'sqlite':
        since_date = datetime.datetime.now() - datetime.timedelta(days=days)
        since_timestamp = since_date.timestamp()
        
        # Используем текстовый запрос для SQLite
        stats = db.execute(text("""
            SELECT 
                AVG(processing_time) as avg_time,
                AVG(input_size) as avg_input_size,
                COUNT(*) as request_count
            FROM request_history 
            WHERE status = 'success' 
            AND strftime('%s', timestamp) >= :since_timestamp
        """), {'since_timestamp': since_timestamp}).first()
        
        if stats:
            # Для перцентилей в SQLite нужен более сложный запрос
            # Здесь возвращаем только средние значения
            return type('Stats', (), {
                'avg_time': stats.avg_time,
                'p50_time': None,  # Не поддерживается в простой версии
                'p95_time': None,
                'p99_time': None,
                'avg_input_size': stats.avg_input_size,
                'request_count': stats.request_count
            })()
        return None
    else:
        # PostgreSQL версия
        since_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        stats = db.query(
            func.avg(models.RequestHistory.processing_time).label('avg_time'),
            func.percentile_cont(0.5).within_group(
                models.RequestHistory.processing_time).label('p50_time'),
            func.percentile_cont(0.95).within_group(
                models.RequestHistory.processing_time).label('p95_time'),
            func.percentile_cont(0.99).within_group(
                models.RequestHistory.processing_time).label('p99_time'),
            func.avg(models.RequestHistory.input_size).label('avg_input_size'),
            func.count(models.RequestHistory.id).label('request_count')
        ).filter(
            models.RequestHistory.timestamp >= since_date,
            models.RequestHistory.status == 'success'
        ).first()
        
        return stats


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def create_user(db: Session, username: str, hashed_password: str, is_admin: bool = False):
    db_user = models.User(
        username=username,
        hashed_password=hashed_password,
        is_admin=is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user