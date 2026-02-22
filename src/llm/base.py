"""
Abstract base class and shared dataclasses for the LLM abstraction layer.

All LLM interactions in the pipeline go through LLMClient.generate(). Each
provider (Anthropic, Gemini) implements this interface so the rest of the
codebase stays provider-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolParam:
    """Provider-agnostic tool definition (JSON Schema style)."""
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema object


@dataclass
class CacheableText:
    """
    A system prompt block with an optional caching hint.

    When cache=True, the Anthropic client applies prompt caching
    (cache_control: ephemeral). The Gemini client ignores this hint.
    """
    text: str
    cache: bool = True


@dataclass
class ThinkingConfig:
    """
    Configuration for extended thinking / reasoning mode.

    budget_tokens maps to Anthropic's budget_tokens and Gemini's thinking_budget.
    When enabled=False this config is ignored entirely.
    """
    enabled: bool = False
    budget_tokens: int = 8000


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    tool_inputs: list[dict[str, Any]]  # one dict per tool call (usually just one)
    text: Optional[str] = None         # non-tool text content, if any
    thinking: Optional[str] = None     # reasoning trace when thinking is enabled
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(ABC):
    """
    Minimal async interface for LLM providers.

    All structured outputs use tool_use. Retry logic is handled internally
    by each implementation — callers do not need to retry.
    """

    @abstractmethod
    async def generate(
        self,
        *,
        model: str,
        system: list[CacheableText],
        messages: list[dict[str, Any]],
        tools: list[ToolParam],
        tool_choice_name: Optional[str] = None,
        thinking: Optional[ThinkingConfig] = None,
        max_tokens: int = 2000,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """
        Send a structured LLM request and return a normalized response.

        Parameters
        ----------
        model:
            Model identifier string (e.g. "claude-sonnet-4-6" or "gemini-2.5-pro").
        system:
            Ordered list of system prompt blocks. Use CacheableText(cache=True)
            on large blocks to enable prompt caching where supported.
        messages:
            Conversation turns in [{"role": "user"|"assistant", "content": "..."}] format.
        tools:
            List of tool definitions the model may call.
        tool_choice_name:
            If set, forces the model to call exactly this tool. If None, the model
            may call any tool or none (auto).
        thinking:
            Extended thinking configuration. Only meaningful for model_reasoning tier.
        max_tokens:
            Maximum output tokens.
        temperature:
            Sampling temperature. Ignored when thinking is enabled on Anthropic
            (API forces temperature=1).

        Raises
        ------
        LLMError
            When the call fails after all internal retries are exhausted.
        """
        ...


class LLMError(Exception):
    """Raised when an LLM call fails after all retries are exhausted."""
    pass
