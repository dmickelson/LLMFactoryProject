from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# from api.paths import root_dir
from pathlib import Path
root_dir = Path.cwd()
env_file_path = root_dir / ".env"


class LLMProviderSettings(BaseSettings):
    # Read the settings from the .env file
    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    # Base settings for all LLM providers
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


class CohereSettings(LLMProviderSettings):
    api_key: str | None = Field(alias="COHERE_API_KEY", default=None)
    default_model: str = "command-r-plus"
    max_tokens: int | None = 1024


class OllamaSettings(LLMProviderSettings):
    api_key: str = "key"  # required, but not used
    default_model: str = "llama3.1"
    base_url: str = "http://localhost:11434/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8", populate_by_name=True)

    app_name: str = "GenAI Project Template"
    openai: OpenAISettings = OpenAISettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    ollama: OllamaSettings = OllamaSettings()
    cohere: CohereSettings = CohereSettings()


"""
By applying lru_cache to the get_settings() function, the project ensures that:
1. The settings are only loaded and processed once, then cached for subsequent calls.
2. Accessing the settings becomes much faster after the initial call, as the cached result is returned.
3. If the settings rarely change during runtime, this caching mechanism significantly reduces overhead.
4. If the settings do change, the cache is invalidated, and the new settings are loaded and cached.

This setup is particularly beneficial for LLM provider configurations, as these settings are likely to be 
accessed frequently throughout the application's lifecycle. The caching helps maintain consistent performance
 when interacting with various LLM providers, reducing unnecessary recomputation of settings.

It's a great example of optimizing resource usage and improving overall application performance in the context of working with language models.
"""


@lru_cache
def get_settings():
    return Settings()
