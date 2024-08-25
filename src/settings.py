from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from api.paths import root_dir

env_file_path = root_dir / ".env"


class LLMProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8", extra="ignore")

    temperature: float = 0.0
    max_tokens: int | None = None
    max_retries: int = 3


class OpenAISettings(LLMProviderSettings):
    api_key: str | None = Field(alias="OPENAI_API_KEY", default=None)
    default_model: str = "gpt-4o"


class AnthropicSettings(LLMProviderSettings):
    api_key: str | None = Field(alias="ANTHROPIC_API_KEY", default=None)
    default_model: str = "claude-3-5-sonnet-20240620"
    max_tokens: int | None = 1024


class OllamaSettings(LLMProviderSettings):
    api_key: str = "key"  # required, but not used
    default_model: str = "llama3.1"
    base_url: str = "http://localhost:11434/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8")

    app_name: str = "GenAI Project Template"
    openai: OpenAISettings = OpenAISettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    ollama: OllamaSettings = OllamaSettings()


@lru_cache
def get_settings():
    return Settings()
