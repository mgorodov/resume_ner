from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)
from app.config.auth import AuthJWTConfig
from app.config.db import DatabaseConfig
from app.config.ml import MLConfig


class UvicornConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__",
        env_prefix="APP_CONFIG__",
    )

    auth_jwt: AuthJWTConfig = AuthJWTConfig()
    db: DatabaseConfig = DatabaseConfig()
    ml: MLConfig = MLConfig()
    uvicorn: UvicornConfig = UvicornConfig()


settings = Settings()
