"""Application settings and logging configuration."""

from __future__ import annotations

import logging
from functools import lru_cache

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed runtime settings."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    agent_llm_model: str = Field(default="gpt-5.1", alias="AGENT_LLM_MODEL")
    supervisor_llm_model: str = Field(default="gpt-5.2", alias="SUPERVISOR_LLM_MODEL")
    embeddings_model: str = Field(default="text-embedding-3-small", alias="EMBEDDINGS_MODEL")
    faiss_index_path: str = Field(default=".cache/faiss_news")

    finnhub_api_key: str | None = Field(default=None, alias="FINNHUB_API_KEY")
    rss_feeds: str = Field(
        default=(
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US,"
            "https://www.marketwatch.com/rss/topstories"
        ),
        alias="RSS_FEEDS",
    )

    max_position_weight: float = Field(default=0.15)
    max_gross_exposure: float = Field(default=1.0)
    max_drawdown_pct: float = Field(default=0.2)
    default_target_weight: float = Field(default=0.05)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def parsed_rss_feeds(self) -> list[str]:
        """Return cleaned list of RSS feed urls."""
        return [item.strip() for item in self.rss_feeds.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""
    return Settings()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging."""
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )
