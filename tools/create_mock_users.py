import asyncio
from app.config.settings import settings
from app.db.helper import DatabaseHelper
from app.db.models import UserOrm
from app.asgi.auth.utils import hash_password
from loguru import logger


async def main():
    db_helper = DatabaseHelper(settings.db.url, echo=True)

    logger.info("Создание пользователей в базе данных...")
    async with db_helper.session_factory() as db_session:
        db_session.add(UserOrm(username="user", hashed_password=hash_password("user")))
        db_session.add(UserOrm(username="admin", hashed_password=hash_password("admin"), roles=["user", "admin"]))
        db_session.add(UserOrm(username="only_admin", hashed_password=hash_password("only_admin"), roles=["admin"]))
        await db_session.commit()
    logger.info("Пользователи созданы успешно!")

    await db_helper.dispose()


if __name__ == "__main__":
    asyncio.run(main())
