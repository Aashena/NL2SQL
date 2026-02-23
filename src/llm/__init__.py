"""
LLM abstraction layer.

Usage:
    from src.llm import get_client, ToolParam, CacheableText, ThinkingConfig, LLMResponse

    client = get_client()
    response = await client.generate(
        model=settings.model_fast,
        system=[CacheableText(text="You are a SQL expert.", cache=True)],
        messages=[{"role": "user", "content": "..."}],
        tools=[ToolParam(name="...", description="...", input_schema={...})],
        tool_choice_name="...",
    )
    result = response.tool_inputs[0]  # dict with tool output

Provider is selected via LLM_PROVIDER env var ("anthropic" or "gemini").
Model tier defaults are applied automatically; override with MODEL_FAST,
MODEL_POWERFUL, MODEL_REASONING env vars.
"""

from src.llm.base import (
    CacheableText,
    LLMClient,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    ThinkingConfig,
    ToolParam,
)
from src.config.settings import settings


def get_client() -> LLMClient:
    """Return an LLMClient for the configured provider.

    Imports are lazy so that google-genai is not required when using Anthropic.
    """
    if settings.llm_provider == "anthropic":
        from src.llm.anthropic_client import AnthropicClient
        return AnthropicClient(api_key=settings.anthropic_api_key)
    elif settings.llm_provider == "gemini":
        from src.llm.gemini_client import GeminiClient
        return GeminiClient(api_key=settings.gemini_api_key)
    else:
        raise ValueError(
            f"Unknown LLM provider: {settings.llm_provider!r}. "
            "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=gemini in .env"
        )


__all__ = [
    "get_client",
    "LLMClient",
    "LLMResponse",
    "LLMError",
    "LLMRateLimitError",
    "ToolParam",
    "CacheableText",
    "ThinkingConfig",
]
