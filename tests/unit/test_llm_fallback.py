"""
Tests for the model fallback mechanism.

Covers:
- Base class fallback loop (str model, list model, rate-limit vs other errors)
- AnthropicClient: LLMRateLimitError raised on anthropic.RateLimitError
- GeminiClient: _is_rate_limit_error helper
- Settings: comma-separated model tier parsing
"""

import pytest

import anthropic

from src.llm.base import (
    CacheableText,
    LLMClient,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    ToolParam,
)
from src.llm.anthropic_client import AnthropicClient
from src.llm.gemini_client import _is_rate_limit_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool() -> ToolParam:
    return ToolParam(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    )


def _make_system() -> list[CacheableText]:
    return [CacheableText(text="sys", cache=False)]


def _make_messages() -> list[dict]:
    return [{"role": "user", "content": "hello"}]


def _ok_response() -> LLMResponse:
    return LLMResponse(tool_inputs=[{"result": "ok"}], input_tokens=10, output_tokens=5)


# ---------------------------------------------------------------------------
# Concrete stub for testing the base class fallback loop in isolation
# ---------------------------------------------------------------------------

class StubClient(LLMClient):
    """LLMClient stub where _generate_single can be configured to succeed or raise."""

    def __init__(self, side_effects: list):
        """side_effects: list of LLMResponse or Exception instances, consumed in order."""
        self._effects = iter(side_effects)
        self.call_args: list[str] = []  # records model names called

    async def _generate_single(self, *, model: str, **kwargs) -> LLMResponse:
        self.call_args.append(model)
        effect = next(self._effects)
        if isinstance(effect, Exception):
            raise effect
        return effect


# ---------------------------------------------------------------------------
# Tests: fallback loop in base class
# ---------------------------------------------------------------------------

class TestFallbackLoop:

    async def test_single_str_model_success(self):
        """Single string model — succeeds, no fallback attempted."""
        client = StubClient([_ok_response()])
        response = await client.generate(
            model="m1",
            system=_make_system(),
            messages=_make_messages(),
            tools=[_make_tool()],
        )
        assert response.tool_inputs == [{"result": "ok"}]
        assert client.call_args == ["m1"]

    async def test_single_element_list_success(self):
        """List with one model — works like single string."""
        client = StubClient([_ok_response()])
        response = await client.generate(
            model=["m1"],
            system=_make_system(),
            messages=_make_messages(),
            tools=[_make_tool()],
        )
        assert response.tool_inputs == [{"result": "ok"}]
        assert client.call_args == ["m1"]

    async def test_rate_limit_on_primary_falls_back_to_secondary(self):
        """Rate limit on m1 → try m2, which succeeds."""
        client = StubClient([
            LLMRateLimitError("m1 rate limited"),
            _ok_response(),
        ])
        response = await client.generate(
            model=["m1", "m2"],
            system=_make_system(),
            messages=_make_messages(),
            tools=[_make_tool()],
        )
        assert response.tool_inputs == [{"result": "ok"}]
        assert client.call_args == ["m1", "m2"]

    async def test_rate_limit_on_all_models_raises_rate_limit_error(self):
        """All models rate-limited → raises LLMRateLimitError (not generic LLMError)."""
        client = StubClient([
            LLMRateLimitError("m1 rate limited"),
            LLMRateLimitError("m2 rate limited"),
        ])
        with pytest.raises(LLMRateLimitError):
            await client.generate(
                model=["m1", "m2"],
                system=_make_system(),
                messages=_make_messages(),
                tools=[_make_tool()],
            )
        assert client.call_args == ["m1", "m2"]

    async def test_non_rate_limit_error_no_fallback(self):
        """Non-rate-limit LLMError on primary → propagates immediately, m2 never called."""
        client = StubClient([
            LLMError("schema validation failed"),
            _ok_response(),  # should never be reached
        ])
        with pytest.raises(LLMError) as exc_info:
            await client.generate(
                model=["m1", "m2"],
                system=_make_system(),
                messages=_make_messages(),
                tools=[_make_tool()],
            )
        # Only m1 was tried
        assert client.call_args == ["m1"]
        # Should NOT be a LLMRateLimitError subclass
        assert type(exc_info.value) is LLMError

    async def test_three_model_list_second_rate_limited_third_succeeds(self):
        """m1 and m2 rate-limited; m3 succeeds."""
        client = StubClient([
            LLMRateLimitError("m1 rate limited"),
            LLMRateLimitError("m2 rate limited"),
            _ok_response(),
        ])
        response = await client.generate(
            model=["m1", "m2", "m3"],
            system=_make_system(),
            messages=_make_messages(),
            tools=[_make_tool()],
        )
        assert client.call_args == ["m1", "m2", "m3"]
        assert response.tool_inputs == [{"result": "ok"}]

    async def test_fallback_logs_warning(self, caplog):
        """A warning is logged when falling back to the next model."""
        import logging
        client = StubClient([
            LLMRateLimitError("rate limited"),
            _ok_response(),
        ])
        with caplog.at_level(logging.WARNING, logger="src.llm.base"):
            await client.generate(
                model=["m1", "m2"],
                system=_make_system(),
                messages=_make_messages(),
                tools=[_make_tool()],
            )
        assert any("m1" in r.message and "m2" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: AnthropicClient raises LLMRateLimitError on anthropic.RateLimitError
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_client(mocker):
    """Return an AnthropicClient with mocked SDK."""
    mock_sdk = mocker.AsyncMock()
    mocker.patch("src.llm.anthropic_client.anthropic.AsyncAnthropic", return_value=mock_sdk)
    client = AnthropicClient(api_key="test-key")
    return client, mock_sdk


def _make_raw_response(mocker, tool_input: dict) -> object:
    tool_block = mocker.MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input
    raw = mocker.MagicMock()
    raw.content = [tool_block]
    raw.usage.input_tokens = 10
    raw.usage.output_tokens = 5
    return raw


class TestAnthropicRateLimitDetection:

    async def test_rate_limit_error_raises_llm_rate_limit_error(self, mocker, mock_anthropic_client):
        """anthropic.RateLimitError → LLMRateLimitError (not generic LLMError)."""
        client, mock_sdk = mock_anthropic_client
        mock_sdk.messages.create.side_effect = anthropic.RateLimitError(
            message="rate limit exceeded",
            response=mocker.MagicMock(status_code=429, headers={}),
            body={},
        )

        with pytest.raises(LLMRateLimitError):
            await client._generate_single(
                model="claude-sonnet-4-6",
                system=_make_system(),
                messages=_make_messages(),
                tools=[_make_tool()],
            )

    async def test_other_error_raises_generic_llm_error(self, mocker, mock_anthropic_client):
        """Non-rate-limit exception → LLMError (not LLMRateLimitError)."""
        client, mock_sdk = mock_anthropic_client
        mock_sdk.messages.create.side_effect = Exception("connection refused")

        with pytest.raises(LLMError) as exc_info:
            await client._generate_single(
                model="claude-sonnet-4-6",
                system=_make_system(),
                messages=_make_messages(),
                tools=[_make_tool()],
            )
        assert type(exc_info.value) is LLMError

    async def test_rate_limit_then_fallback_via_generate(self, mocker, mock_anthropic_client):
        """Full integration: rate limit on generate() triggers fallback to second model."""
        client, mock_sdk = mock_anthropic_client

        success_raw = _make_raw_response(mocker, {"result": "fallback"})

        call_count = {"n": 0}

        async def side_effect(**kwargs):
            call_count["n"] += 1
            if kwargs["model"] == "claude-sonnet-4-6":
                raise anthropic.RateLimitError(
                    message="rate limit exceeded",
                    response=mocker.MagicMock(status_code=429, headers={}),
                    body={},
                )
            return success_raw

        mock_sdk.messages.create.side_effect = side_effect

        response = await client.generate(
            model=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            system=_make_system(),
            messages=_make_messages(),
            tools=[_make_tool()],
        )

        assert response.tool_inputs == [{"result": "fallback"}]


# ---------------------------------------------------------------------------
# Tests: _is_rate_limit_error helper for Gemini
# ---------------------------------------------------------------------------

class TestIsRateLimitError:

    def test_resource_exhausted_string_in_message(self):
        assert _is_rate_limit_error(Exception("RESOURCE_EXHAUSTED: quota exceeded"))

    def test_429_in_message(self):
        assert _is_rate_limit_error(Exception("HTTP 429 too many requests"))

    def test_quota_in_message(self):
        assert _is_rate_limit_error(Exception("quota limit reached"))

    def test_unrelated_error_returns_false(self):
        assert not _is_rate_limit_error(Exception("invalid JSON in response"))

    def test_connection_error_returns_false(self):
        assert not _is_rate_limit_error(ConnectionError("connection refused"))


# ---------------------------------------------------------------------------
# Tests: Settings — comma-separated model tier parsing
# ---------------------------------------------------------------------------

class TestSettingsModelLists:

    def test_single_model_powerful(self, monkeypatch):
        """Single model string → list with one element."""
        monkeypatch.setenv("MODEL_POWERFUL", "claude-sonnet-4-6")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        # Re-instantiate settings to pick up env change
        from src.config.settings import Settings
        s = Settings()
        assert s.model_powerful == "claude-sonnet-4-6"
        assert s.model_powerful_list == ["claude-sonnet-4-6"]

    def test_comma_separated_model_powerful(self, monkeypatch):
        """Comma-separated → list; primary str is the first model."""
        monkeypatch.setenv("MODEL_POWERFUL", "gemini-2.5-pro,gemini-2.5-flash")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        from src.config.settings import Settings
        s = Settings()
        assert s.model_powerful == "gemini-2.5-pro"
        assert s.model_powerful_list == ["gemini-2.5-pro", "gemini-2.5-flash"]

    def test_comma_separated_with_spaces(self, monkeypatch):
        """Spaces around commas are stripped."""
        monkeypatch.setenv("MODEL_POWERFUL", " gemini-2.5-pro , gemini-2.5-flash ")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        from src.config.settings import Settings
        s = Settings()
        assert s.model_powerful_list == ["gemini-2.5-pro", "gemini-2.5-flash"]

    def test_empty_model_powerful_uses_provider_default(self, monkeypatch):
        """Empty MODEL_POWERFUL → provider default is applied, list has one element."""
        monkeypatch.delenv("MODEL_POWERFUL", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        from src.config.settings import Settings
        s = Settings()
        assert s.model_powerful == "claude-sonnet-4-6"
        assert s.model_powerful_list == ["claude-sonnet-4-6"]

    def test_three_fallbacks(self, monkeypatch):
        """Three models in list."""
        monkeypatch.setenv("MODEL_POWERFUL", "m1,m2,m3")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        from src.config.settings import Settings
        s = Settings()
        assert s.model_powerful_list == ["m1", "m2", "m3"]
        assert s.model_powerful == "m1"

    def test_model_fast_list_also_parsed(self, monkeypatch):
        """model_fast_list is also populated when MODEL_FAST has a fallback."""
        monkeypatch.setenv("MODEL_FAST", "haiku-v1,haiku-v2")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        from src.config.settings import Settings
        s = Settings()
        assert s.model_fast_list == ["haiku-v1", "haiku-v2"]
        assert s.model_fast == "haiku-v1"
