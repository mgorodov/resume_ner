from fastapi import APIRouter, HTTPException

from app.asgi.auth.utils import hash_password
from app.asgi.state import DBSessionDep
from app.db.models import UserOrm
from fastapi import status
from app.asgi.schemas.users import CreateUser
import sqlalchemy.dialects.sqlite as sqlite_sa
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/")
async def create_user(db_session: DBSessionDep, request: CreateUser):
    sql = (
        sqlite_sa.insert(UserOrm)
        .values(username=request.username, hashed_password=hash_password(request.password), roles=["user"])
        .on_conflict_do_nothing(index_elements=[UserOrm.username])
        .returning(UserOrm.id)
    )

    query_result = await db_session.execute(sql)
    user_id = query_result.scalar_one_or_none()
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")

    try:
        await db_session.commit()
    except SQLAlchemyError:
        await db_session.rollback()
        raise

    return {"msg": "success", "user_id": user_id}
