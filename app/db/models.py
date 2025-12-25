from sqlalchemy import JSON, LargeBinary
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

import datetime as dt
from app.db.mixins.created_at import CreatedAtMixin
from app.db.mixins.int_id_pk import IntIdPkMixin

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class UserOrm(IntIdPkMixin, CreatedAtMixin, Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(unique=True)
    hashed_password: Mapped[bytes] = mapped_column(LargeBinary)
    roles: Mapped[list[str]] = mapped_column(JSON, default=lambda: ["user"])


class ForwardHistoryOrm(IntIdPkMixin, Base):
    __tablename__ = "forward_history"

    actor: Mapped[str | None]
    text: Mapped[str]

    entities: Mapped[list[dict] | None] = mapped_column(JSON)
    error: Mapped[str | None]

    started_at: Mapped[dt.datetime]
    duration_us: Mapped[int]
