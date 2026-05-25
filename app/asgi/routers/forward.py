import time
from fastapi import APIRouter, HTTPException, status

from app.asgi.auth.deps import UserRoleDep
from app.asgi.state import NERServiceDep
from app.asgi.state import DBSessionDep
from app.asgi.schemas.forward import ForwardResponse, ForwardRequest
from app.db.models import ForwardHistoryOrm
import datetime as dt
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

router = APIRouter(prefix="/forward", tags=["Forward"])


@router.post("/")
async def forward(
    db_session: DBSessionDep,
    ner_service: NERServiceDep,
    request: ForwardRequest,
    user_info: UserRoleDep,
) -> ForwardResponse:
    history_record = ForwardHistoryOrm(
        actor=user_info.actor,
        text=request.text,
        started_at=dt.datetime.now(tz=dt.timezone.utc),
    )
    error: str | None = None

    start_counter_ns = time.perf_counter_ns()
    try:
        entities = ner_service.infer(request.text)
        history_record.entities = [ent.model_dump() for ent in ner_service.infer(request.text)]
    except Exception as e:
        logger.exception("NER service error")
        error = str(e)

    end_counter_ns = time.perf_counter_ns()
    duration_us = (end_counter_ns - start_counter_ns) // 1000

    history_record.error = error
    history_record.duration_us = duration_us
    db_session.add(history_record)
    try:
        await db_session.commit()
    except SQLAlchemyError:
        await db_session.rollback()
        raise

    if error is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Модель не смогла обработать данные",
        )

    return ForwardResponse(entities=entities, duration_us=duration_us)
