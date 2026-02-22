"""
Tests for src/llm/anthropic_client.py — AnthropicClient.

Tests the retry logic, prompt caching flag mapping, extended thinking config,
and response parsing. All tests mock anthropic.AsyncAnthropic — no real API calls.
"""

import pytest

from src.llm.anthropic_client import AnthropicClient
from src.llm.base import CacheableText, LLMError, LLMResponse, ThinkingConfig, ToolParam


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_raw_response(mocker, tool_input: dict, thinking: str = None):
    """Build a mock Anthropic API response with a tool_use block."""
    blocks = []

    if thinking:
        thinking_block = mocker.MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = thinking
        blocks.append(thinking_block)

    tool_block = mocker.MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input
    blocks.append(tool_block)

    raw = mocker.MagicMock()
    raw.content = blocks
    raw.usage.input_tokens = 100
    raw.usage.output_tokens = 50
    return raw


def _make_tool() -> ToolParam:
    return ToolParam(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    )


# ---------------------------------------------------------------------------
# Helper to build client with mocked SDK
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_async_anthropic(mocker):
    """Patch anthropic.AsyncAnthropic and return (client, mock_sdk_instance)."""
    mock_sdk = mocker.AsyncMock()
    mocker.patch("src.llm.anthropic_client.anthropic.AsyncAnthropic", return_value=mock_sdk)
    client = AnthropicClient(api_key="test-key")
    return client, mock_sdk


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetryOnError:
    """Retry logic: 3 attempts, succeeds on the 3rd."""

    async def test_retries_twice_then_succeeds(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        success_response = _make_raw_response(mocker, {"result": "ok"})

        mock_sdk.messages.create.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            success_response,
        ]

        response = await client.generate(
            model="claude-haiku-4-5-20251001",
            system=[CacheableText(text="sys", cache=False)],
            messages=[{"role": "user", "content": "hello"}],
            tools=[_make_tool()],
            tool_choice_name="test_tool",
        )

        assert mock_sdk.messages.create.call_count == 3
        assert response.tool_inputs == [{"result": "ok"}]

    async def test_raises_after_3_failures(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.side_effect = Exception("permanent error")

        with pytest.raises(LLMError):
            await client.generate(
                model="claude-haiku-4-5-20251001",
                system=[CacheableText(text="sys", cache=False)],
                messages=[{"role": "user", "content": "hello"}],
                tools=[_make_tool()],
            )

        assert mock_sdk.messages.create.call_count == 3


class TestPromptCaching:
    """CacheableText(cache=True) → cache_control: ephemeral block."""

    async def test_cache_true_adds_cache_control(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.return_value = _make_raw_response(mocker, {"result": "ok"})

        await client.generate(
            model="claude-haiku-4-5-20251001",
            system=[CacheableText(text="big schema block", cache=True)],
            messages=[{"role": "user", "content": "q"}],
            tools=[_make_tool()],
        )

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        system_blocks = call_kwargs["system"]
        assert len(system_blocks) == 1
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}

    async def test_cache_false_omits_cache_control(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.return_value = _make_raw_response(mocker, {"result": "ok"})

        await client.generate(
            model="claude-haiku-4-5-20251001",
            system=[CacheableText(text="small block", cache=False)],
            messages=[{"role": "user", "content": "q"}],
            tools=[_make_tool()],
        )

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        system_blocks = call_kwargs["system"]
        assert "cache_control" not in system_blocks[0]


class TestExtendedThinking:
    """ThinkingConfig(enabled=True) → thinking param + temperature=1."""

    async def test_thinking_enabled_sets_params(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.return_value = _make_raw_response(
            mocker, {"result": "ok"}, thinking="my reasoning"
        )

        response = await client.generate(
            model="claude-sonnet-4-6",
            system=[CacheableText(text="sys", cache=False)],
            messages=[{"role": "user", "content": "complex q"}],
            tools=[_make_tool()],
            thinking=ThinkingConfig(enabled=True, budget_tokens=6000),
        )

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 6000}
        assert call_kwargs["temperature"] == 1.0
        assert response.thinking == "my reasoning"

    async def test_thinking_disabled_no_thinking_param(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.return_value = _make_raw_response(mocker, {"result": "ok"})

        await client.generate(
            model="claude-sonnet-4-6",
            system=[CacheableText(text="sys", cache=False)],
            messages=[{"role": "user", "content": "q"}],
            tools=[_make_tool()],
            thinking=ThinkingConfig(enabled=False),
            temperature=0.7,
        )

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs
        assert call_kwargs["temperature"] == 0.7


class TestResponseParsing:
    """LLMResponse is correctly populated from the raw API response."""

    async def test_tool_inputs_extracted(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.return_value = _make_raw_response(
            mocker, {"answer": "42"}
        )

        response = await client.generate(
            model="claude-haiku-4-5-20251001",
            system=[CacheableText(text="s", cache=False)],
            messages=[{"role": "user", "content": "q"}],
            tools=[_make_tool()],
        )

        assert response.tool_inputs == [{"answer": "42"}]
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    async def test_tool_choice_name_sets_forced_tool(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.return_value = _make_raw_response(mocker, {})

        await client.generate(
            model="claude-haiku-4-5-20251001",
            system=[CacheableText(text="s", cache=False)],
            messages=[{"role": "user", "content": "q"}],
            tools=[_make_tool()],
            tool_choice_name="test_tool",
        )

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "test_tool"}

    async def test_no_tool_choice_uses_auto(self, mocker, mock_async_anthropic):
        client, mock_sdk = mock_async_anthropic
        mock_sdk.messages.create.return_value = _make_raw_response(mocker, {})

        await client.generate(
            model="claude-haiku-4-5-20251001",
            system=[CacheableText(text="s", cache=False)],
            messages=[{"role": "user", "content": "q"}],
            tools=[_make_tool()],
            tool_choice_name=None,
        )

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}
