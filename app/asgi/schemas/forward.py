from pydantic import BaseModel
from app.ml.schema import NamedEntity


class ForwardRequest(BaseModel):
    text: str


class ForwardResponse(BaseModel):
    entities: list[NamedEntity]
    duration_us: int
