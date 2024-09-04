"""
LLM Factory Module

This module provides a factory class for initializing and interacting with various Language Model (LLM) providers.

Key components:
- LLMProviders: Type alias for supported LLM providers
- LLMSettings: Type alias for provider-specific settings
- ClientInitializerCallback: Protocol for client initialization functions
- ClientInitializer: Type alias for a dictionary of initialization callbacks
- LLMFactory: Main class for LLM client management and interaction

The module supports OpenAI, Anthropic, and Ollama as LLM providers.
"""

# Imports
from collections.abc import Generator
from typing import Any, Literal, Protocol, Union, Dict, List
import instructor
from anthropic import Anthropic
from openai import OpenAI
import cohere
from pydantic import BaseModel, Field

# from api.custom_types import LiteralFalse, LiteralTrue

# from settings import AnthropicSettings, OllamaSettings, OpenAISettings, get_settings
from llmfactory.settings import AnthropicSettings, OllamaSettings, OpenAISettings, CohereSettings, get_settings

LLMProviders = Literal["ollama", "openai", "anthropic", "cohere"]
LLMSettings = Union[OpenAISettings,
                    AnthropicSettings, OllamaSettings, CohereSettings]


# Protocol for client initialization
class ClientInitializerCallback(Protocol):
    def __call__(self, settings: LLMSettings) -> instructor.Instructor: ...


ClientInitializer = Dict[LLMProviders, ClientInitializerCallback]


class LLMFactory:
    """
    Factory class for initializing and managing LLM clients.

    Attributes:
        provider (LLMProviders): The selected LLM provider.
        settings (LLMSettings): Provider-specific settings.
        client (instructor.Instructor): Initialized LLM client.

    Methods:
        __init__: Initialize the LLMFactory with a specific provider.
        _initialize_client: Set up the LLM client based on the provider.
        create_completion: Generate completions using the LLM.
    """

    def __init__(self, provider: LLMProviders) -> None:
        """
        Initialize the LLMFactory with a specific LLM provider.

        Args:
            provider (LLMProviders): The LLM provider to use.
        """
        self.provider: LLMProviders = provider
        self.settings: LLMSettings = getattr(get_settings(), provider)
        self.client: instructor.Instructor = self._initialize_client()

    def _initialize_client(self) -> instructor.Instructor:
        """
        Initialize the LLM client based on the selected provider.

        Returns:
            instructor.Instructor: Initialized LLM client.

        Raises:
            ValueError: If an unsupported LLM provider is specified.
        """
        client_initializers: ClientInitializer = {
            "openai": lambda settings: instructor.from_openai(OpenAI(api_key=settings.api_key)),
            "anthropic": lambda settings: instructor.from_anthropic(Anthropic(api_key=settings.api_key)),
            "cohere": lambda settings: instructor.from_cohere(cohere.Client(api_key=settings.api_key)),
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

    def create_completion(
        self,
        response_model: type[BaseModel],
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        Generate completions using the initialized LLM client.

        Args:
            response_model (T): Pydantic model for structuring the response.
            messages (list[dict[str, str]]): List of message dictionaries for the conversation.
            **kwargs: Additional parameters for the completion.

        Returns:
            T | Generator[T, None, None]: Structured response or a generator of responses.
        """
        completion_params = {
            "model": kwargs.get("model") or self.settings.default_model,
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }

        return self.client.chat.completions.create(**completion_params)


class CompletionModel(BaseModel):
    response: str = Field(description="Your response to the user.")
    reasoning: str = Field(
        description="Explain your reasoning for the response.")
