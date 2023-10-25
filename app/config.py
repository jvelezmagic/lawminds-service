from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    OPENAI_API_KEY: str
    OPENAI_API_ORGANIZATION_ID: str
    AI21_API_KEY: str
    WEAVIATE_URL: str
    REDIS_URL: str
    INDEX_DATABASE_URL: str
    INDEX_NAME: str


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore
