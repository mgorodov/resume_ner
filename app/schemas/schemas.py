from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class TextRequest(BaseModel):
    text: str = Field(..., description="Text for NER processing")


class ImageRequest(BaseModel):
    description: Optional[str] = Field(None, description="Optional description for image")


class ForwardResponse(BaseModel):
    success: bool
    entities: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    processing_time: float
    image_result: Optional[str] = None


class HistoryResponse(BaseModel):
    id: int
    endpoint: str
    method: str
    timestamp: datetime
    processing_time: float
    input_type: Optional[str]
    status: str
    
    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    avg_processing_time: Optional[float]
    p50_processing_time: Optional[float]
    p95_processing_time: Optional[float]
    p99_processing_time: Optional[float]
    avg_input_size: Optional[float]
    request_count: Optional[int]


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    password: str
    is_admin: bool = False


class UserLogin(BaseModel):
    username: str
    password: str