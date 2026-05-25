from pydantic import BaseModel


class DistributionSummary(BaseModel):
    avg: float
    p50: float
    p95: float
    p99: float


class GetStatsResponse(BaseModel):
    count: int
    duration_us: DistributionSummary
    text_length: DistributionSummary
