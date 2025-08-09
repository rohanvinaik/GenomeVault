from __future__ import annotations

"""Settings module."""
"""Settings module."""
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Settings implementation."""
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("INFO", env="GENOMEVAULT_LOG_LEVEL")
    rate_limit_rps: float = Field(5.0, env="RATE_LIMIT_RPS")
    rate_limit_burst: int = Field(10, env="RATE_LIMIT_BURST")

    class Config:
        """Config implementation."""
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
