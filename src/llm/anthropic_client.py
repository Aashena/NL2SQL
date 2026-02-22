"""
Anthropic (Claude) implementation of LLMClient.

Features:
- Uses anthropic.AsyncAnthropic for async calls
- Prompt caching via cache_control: ephemeral on CacheableText(cache=True) blocks
- Extended thinking via ThinkingConfig (forces temperature=1 per API constraint)
- Tenacity retry: 3 attempts, exponential backoff 2–30 seconds
"""

from typing import Any, Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.base import CacheableText, LLMClient, LLMError, LLMResponse, ThinkingConfig, ToolParam


class AnthropicClient(LLMClient):

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

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
        system_blocks = _build_system_blocks(system)
        tool_defs = [_to_anthropic_tool(t) for t in tools]
        tool_choice = (
            {"type": "tool", "name": tool_choice_name}
            if tool_choice_name
            else {"type": "auto"}
        )

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            tools=tool_defs,
            tool_choice=tool_choice,
            messages=messages,
        )

        if thinking and thinking.enabled:
            # Extended thinking requires temperature=1 (Anthropic API constraint)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking.budget_tokens}
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        try:
            raw = await self._call_with_retry(**kwargs)
        except Exception as exc:
            raise LLMError(f"Anthropic API call failed after retries: {exc}") from exc

        return _parse_response(raw)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_with_retry(self, **kwargs: Any) -> Any:
        return await self._client.messages.create(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_blocks(system: list[CacheableText]) -> list[dict[str, Any]]:
    blocks = []
    for block in system:
        entry: dict[str, Any] = {"type": "text", "text": block.text}
        if block.cache:
            entry["cache_control"] = {"type": "ephemeral"}
        blocks.append(entry)
    return blocks


def _to_anthropic_tool(t: ToolParam) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    }


def _parse_response(raw: Any) -> LLMResponse:
    tool_inputs: list[dict[str, Any]] = []
    text_parts: list[str] = []
    thinking_text: Optional[str] = None

    for block in raw.content:
        if block.type == "tool_use":
            tool_inputs.append(block.input)
        elif block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            text_parts.append(block.text)

    return LLMResponse(
        tool_inputs=tool_inputs,
        text="\n".join(text_parts) or None,
        thinking=thinking_text,
        input_tokens=raw.usage.input_tokens,
        output_tokens=raw.usage.output_tokens,
    )
