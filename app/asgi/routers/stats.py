from fastapi import APIRouter
import sqlalchemy as sa

from app.asgi.auth.deps import UserRoleDep

from app.asgi.state import DBSessionDep
from app.db.models import ForwardHistoryOrm
from app.asgi.schemas.stats import GetStatsResponse, DistributionSummary

import numpy as np

router = APIRouter(prefix="/stats", tags=["Stats"])


def summarize_distribution(arr: np.ndarray) -> DistributionSummary:
    avg = float(np.mean(arr))
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return DistributionSummary(avg=avg, p50=p50, p95=p95, p99=p99)


@router.get("/")
async def get_stats(db_session: DBSessionDep, user_info: UserRoleDep, only_mine: bool = True) -> GetStatsResponse:
    sql = sa.select(ForwardHistoryOrm.duration_us, sa.func.length(ForwardHistoryOrm.text))
    if only_mine:
        sql = sql.where(ForwardHistoryOrm.actor == user_info.actor)

    query_result = await db_session.execute(sql)
    rows = query_result.all()

    dur_arr = np.fromiter((x[0] for x in rows), dtype=np.int64, count=len(rows))
    text_len_arr = np.fromiter((x[1] for x in rows), dtype=np.int64, count=len(rows))

    return GetStatsResponse(
        count=len(rows),
        duration_us=summarize_distribution(dur_arr),
        text_length=summarize_distribution(text_len_arr),
    )
