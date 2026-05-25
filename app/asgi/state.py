from typing import Annotated

from fastapi import Depends, Request

from collections.abc import AsyncIterator
from app.db.helper import DatabaseHelper
from app.ml.ner_service import NERService


from sqlalchemy.ext.asyncio import AsyncSession


async def gen_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    db_helper: DatabaseHelper = request.app.state.db_helper
    async with db_helper.session_factory() as db_session:
        yield db_session


def get_ner_service(request: Request) -> NERService:
    return request.app.state.ner_service


DBSessionDep = Annotated[AsyncSession, Depends(gen_db_session)]
NERServiceDep = Annotated[NERService, Depends(get_ner_service)]
