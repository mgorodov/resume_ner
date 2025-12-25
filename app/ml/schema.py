from pydantic import BaseModel


class NamedEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
