import datetime as dt

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column


def get_current_dt() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0, tzinfo=None)


class CreatedAtMixin:
    created_at: Mapped[dt.datetime] = mapped_column(default=get_current_dt, server_default=func.now())
