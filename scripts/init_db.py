#!/usr/bin/env python
"""
Инициализация базы данных
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.database.session import engine, Base
from app.database.crud import create_user
from app.core.security import get_password_hash
from app.database.session import SessionLocal


def init_database():
    print("Создание таблиц в базе данных...")
    Base.metadata.create_all(bind=engine)
    print("Таблицы созданы успешно!")


def create_admin_user():
    db = SessionLocal()
    try:
        from app.core.config import settings
        
        create_user(
            db=db,
            username=settings.ADMIN_USERNAME,
            hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
            is_admin=True
        )
        print(f"Пользователь администратора '{settings.ADMIN_USERNAME}' создан")
    except Exception as e:
        print(f"Ошибка при создании администратора: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    init_database()
    create_admin_user()