"""Runtime settings from the environment and `.env`.

Classes:
    Settings: OpenRouter key and base URL, data/config directories, HTTP budgets.
Functions:
    load_settings: read Settings from the current environment.
"""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JEV_BENCH_",
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    openrouter_base_url: str = "https://openrouter.ai/api"
    data_dir: Path = Path("data")
    config_dir: Path = Path("config")
    request_timeout_s: float = Field(default=60.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_base_delay_s: float = Field(default=0.5, ge=0)

    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    def server_api_key(self) -> str | None:
        return self.openrouter_api_key.get_secret_value() if self.openrouter_api_key else None


def load_settings() -> Settings:
    return Settings()
