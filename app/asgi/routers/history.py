from typing import cast
from fastapi import APIRouter
import sqlalchemy as sa

from app.asgi.auth.deps import UserRoleDep

from app.asgi.state import DBSessionDep
from app.db.models import ForwardHistoryOrm
from app.asgi.schemas.history import GetHistoryResponse, HistoryRecord
from app.ml.schema import NamedEntity
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/")
async def get_history(db_session: DBSessionDep, user_info: UserRoleDep, only_mine: bool = True) -> GetHistoryResponse:
    if not only_mine:
        user_info.verify_role("admin")

    sql = sa.select(ForwardHistoryOrm)
    if only_mine:
        sql = sql.where(ForwardHistoryOrm.actor == user_info.actor)

    query_result = await db_session.execute(sql)
    fwd_history_orms = query_result.scalars().all()
    return GetHistoryResponse(
        records=[
            HistoryRecord(
                started_at=fwd_history_orm.started_at,
                actor=fwd_history_orm.actor,
                entities=cast(list[NamedEntity], fwd_history_orm.entities),
                text=fwd_history_orm.text,
                error=fwd_history_orm.error,
                duration_us=fwd_history_orm.duration_us,
            )
            for fwd_history_orm in fwd_history_orms
        ]
    )


@router.delete("/")
async def delete_history(db_session: DBSessionDep, user_info: UserRoleDep, only_mine: bool = True):
    if not only_mine:
        user_info.verify_role("admin")

    sql = sa.delete(ForwardHistoryOrm)
    if only_mine:
        sql = sql.where(ForwardHistoryOrm.actor == user_info.actor)

    await db_session.execute(sql)
    try:
        await db_session.commit()
    except SQLAlchemyError:
        await db_session.rollback()
        raise
    return {"msg": "success"}
