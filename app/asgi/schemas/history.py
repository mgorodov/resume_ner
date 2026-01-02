from pydantic import BaseModel
import datetime as dt
from app.ml.schema import NamedEntity


class HistoryRecord(BaseModel):
    started_at: dt.datetime
    actor: str | None
    entities: list[NamedEntity] | None
    text: str
    error: str | None
    duration_us: int


class GetHistoryResponse(BaseModel):
    records: list[HistoryRecord]
