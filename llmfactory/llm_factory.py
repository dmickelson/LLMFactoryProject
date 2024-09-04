from collections.abc import Generator
from typing import Any, Literal, Protocol, overload

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

# from api.custom_types import LiteralFalse, LiteralTrue
from settings import AnthropicSettings, OllamaSettings, OpenAISettings, get_settings

type LLMProviders = Literal["ollama", "openai", "anthropic"]
type LLMSettings = OpenAISettings | AnthropicSettings | OllamaSettings


class ClientInitializerCallback(Protocol):
    def __call__(self, settings: LLMSettings) -> instructor.Instructor: ...


type ClientInitializer = dict[LLMProviders, ClientInitializerCallback]


class LLMFactory:
    def __init__(self, provider: LLMProviders) -> None:
        self.provider: LLMProviders = provider
        self.settings: LLMSettings = getattr(get_settings(), provider)
        self.client: instructor.Instructor = self._initialize_client()

    def _initialize_client(self) -> instructor.Instructor:
        client_initializers: ClientInitializer = {
            "openai": lambda settings: instructor.from_openai(OpenAI(api_key=settings.api_key)),
            "anthropic": lambda settings: instructor.from_anthropic(Anthropic(api_key=settings.api_key)),
            "ollama": lambda settings: instructor.from_openai(
                # type: ignore Ollama setting will have `settings.base_url`
                OpenAI(base_url=settings.base_url, api_key=settings.api_key),
                mode=instructor.Mode.JSON,
            ),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)

        err_msg = f"Unsupported LLM provider: {self.provider}"
        raise ValueError(err_msg)

    def create_completion[T: type[BaseModel]](
        self,
        response_model: T,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> T | Generator[T, None, None]:
        completion_params = {
            "model": kwargs.get("model") or self.settings.default_model,
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }

        # type: ignore Instructor needs to improve type hints
        return self.client.chat.completions.create(**completion_params)
