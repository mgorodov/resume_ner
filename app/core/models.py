from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, Boolean
from sqlalchemy.sql import func, text
from app.database.session import Base


class RequestHistory(Base):
    __tablename__ = "request_history"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, nullable=False)
    method = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    processing_time = Column(Float, nullable=False)
    input_size = Column(Integer)
    input_type = Column(String)
    status = Column(String, nullable=False)
    response_data = Column(JSON, nullable=True)
    user_agent = Column(String)
    ip_address = Column(String)


class RequestStats(Base):
    __tablename__ = "request_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    endpoint = Column(String, nullable=False)
    avg_processing_time = Column(Float)
    p50_processing_time = Column(Float)
    p95_processing_time = Column(Float)
    p99_processing_time = Column(Float)
    avg_input_size = Column(Float)
    request_count = Column(Integer, default=0)


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())