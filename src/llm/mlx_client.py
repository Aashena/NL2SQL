"""
MLX-LM implementation of LLMClient.

Connects to a locally-running mlx_lm.server OpenAI-compatible endpoint and uses
the `instructor` library (Mode.JSON) for structured JSON output.

Usage:
    # Start the server in a separate terminal:
    #   python -m mlx_lm.server \\
    #       --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \\
    #       --host 127.0.0.1 --port 8080
    #
    # Then set in .env:
    #   LLM_PROVIDER=mlx
    #   MLX_SERVER_URL=http://127.0.0.1:8080
    #   MLX_MODEL_NAME=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit

Feature gaps vs. cloud providers (intentional — local model limitations):
  - CacheableText.cache: ignored (no prompt caching on local server)
  - ThinkingConfig: ignored (Qwen3 has no Anthropic-style extended thinking)
  - LLMResponse.thinking: always None

Requires the [mlx] optional dependency group:
    pip install -e ".[mlx]"
"""

import asyncio
import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, create_model
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.base import (
    CacheableText,
    LLMClient,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    ThinkingConfig,
    ToolParam,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Schema → Pydantic model converter
# ---------------------------------------------------------------------------

def _field_type(schema: dict[str, Any]) -> Any:
    """Map a JSON Schema field schema to a Python/Pydantic-compatible type."""
    t = schema.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        inner = _field_type(schema.get("items", {}))
        return list[inner]
    if t == "object":
        return _build_model(schema, "Nested")
    # anyOf / oneOf / allOf — fall back to Any
    return Any


def _build_model(schema: dict[str, Any], name: str) -> type[BaseModel]:
    """Recursively build a Pydantic BaseModel from a JSON Schema object."""
    props: dict[str, Any] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))
    field_defs: dict[str, Any] = {}
    for fname, fschema in props.items():
        ft = _field_type(fschema)
        if fname in required:
            field_defs[fname] = (ft, ...)
        else:
            field_defs[fname] = (Optional[ft], None)
    return create_model(name, **field_defs)


def _json_schema_to_pydantic(tool: ToolParam) -> type[BaseModel]:
    """
    Convert a ToolParam's JSON Schema to a named Pydantic BaseModel.

    The model is used as the instructor response_model so that instructor can
    inject the schema into the system prompt (Mode.JSON) and validate the
    model's JSON response against it.
    """
    model_cls = _build_model(tool.input_schema, tool.name)
    model_cls.__doc__ = tool.description
    return model_cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_tool(
    tools: list[ToolParam],
    tool_choice_name: Optional[str],
) -> Optional[ToolParam]:
    """Return the active ToolParam given an optional forced name.

    Returns None if tools is empty (plain-text call).
    Raises LLMError if tool_choice_name is set but not found in tools.
    """
    if not tools:
        return None
    if tool_choice_name is None:
        return tools[0]
    for t in tools:
        if t.name == tool_choice_name:
            return t
    raise LLMError(
        f"tool_choice_name={tool_choice_name!r} not found in tools list "
        f"(available: {[t.name for t in tools]})"
    )


def _extract_tokens(result_obj: Any) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from an instructor-returned model.

    instructor stores the raw OpenAI response on the result object under
    _raw_response (instructor >=1.x). Falls back to (0, 0) if unavailable —
    token counts are only used for diagnostics, not pipeline correctness.
    """
    raw = getattr(result_obj, "_raw_response", None)
    if raw is None:
        raw = getattr(result_obj, "__instructor_raw_response__", None)
    if raw is None:
        return 0, 0
    usage = getattr(raw, "usage", None)
    if usage is None:
        return 0, 0
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _is_rate_limit(exc: Exception) -> bool:
    """
    Return True if exc looks like a rate-limit or transient server overload.

    mlx_lm.server is single-process and returns HTTP 503 when busy.
    We treat 503 the same as 429 so the base class fallback logic fires.

    ConnectError is also treated as transient: the local MLX server can drop
    briefly during model loading or after a large generation.  Classifying it
    as a rate-limit-style error makes tenacity back off (2–30 s) instead of
    hammering the server immediately.
    """
    try:
        from openai import APIStatusError, RateLimitError  # type: ignore[import]
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APIStatusError) and exc.status_code in (429, 503):
            return True
    except ImportError:
        pass
    try:
        if isinstance(exc, httpx.ConnectError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return (
        "429" in msg or "503" in msg or "rate limit" in msg
        or "too many requests" in msg
        or "all connection attempts failed" in msg
        or "connection error" in msg
    )


# ---------------------------------------------------------------------------
# Global serialisation gate for the local MLX server
# ---------------------------------------------------------------------------

# mlx_lm.server is single-process: concurrent requests cause 503 errors and
# wasted context-switching.  This semaphore ensures only ONE prompt is in
# flight to the server at any time, regardless of how many concurrent pipeline
# questions are running.
_MLX_SEMAPHORE = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# MLXClient
# ---------------------------------------------------------------------------

class MLXClient(LLMClient):
    """
    LLMClient implementation that forwards requests to a local mlx_lm.server.

    Uses instructor in Mode.JSON: the Pydantic response_model is injected into
    the system prompt by instructor, the model returns a JSON object, and
    instructor validates and parses it. This approach works with any
    OpenAI-compatible server regardless of native function-calling support.
    """

    def __init__(self, server_url: str, model_name: str) -> None:
        """
        Parameters
        ----------
        server_url:
            Base URL of the mlx_lm.server, e.g. "http://127.0.0.1:8080".
            "/v1" is appended automatically.
        model_name:
            Model identifier forwarded in every API request (the server ignores
            it but it aids logging/tracing).
        """
        try:
            import instructor  # type: ignore[import]
            from openai import AsyncOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai>=1.0 and instructor are required for the MLX provider. "
                "Install with: pip install -e '.[mlx]'"
            ) from exc

        self._model_name = model_name
        v1_url = f"{server_url.rstrip('/')}/v1"

        # One underlying HTTP client shared by both the raw client and the
        # instructor-patched client (same connection pool).
        self._raw_client = AsyncOpenAI(
            base_url=v1_url,
            api_key="local",  # mlx_lm.server ignores the API key
            # read=None: let asyncio wait_for (per-stage) be the sole deadline.
            # A fixed httpx read timeout shorter than the stage timeout causes
            # instructor to retry (3× the timeout) before asyncio can cancel cleanly.
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        )
        # instructor patches the client in-place; both references share state.
        self._client = instructor.from_openai(
            self._raw_client,
            mode=instructor.Mode.JSON,
        )

        logger.info(
            "MLXClient initialised — server: %s, model: %s", v1_url, model_name
        )

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    async def _generate_single(
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
        # Serialise all calls: the local MLX server is single-process and
        # cannot handle concurrent requests without 503 errors.
        async with _MLX_SEMAPHORE:
            return await self._generate_single_locked(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice_name=tool_choice_name,
                thinking=thinking,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    async def _generate_single_locked(
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
        # CacheableText.cache hints are ignored; text content is still used.
        system_text = "\n\n".join(block.text for block in system)

        # Qwen3 thinking-mode control via /no_think / /think prefix.
        # Qwen3 generates <think>…</think> blocks by default, which adds hundreds
        # to thousands of tokens (and minutes of latency) before the JSON answer.
        # Prepend /no_think to disable it for every call except those that
        # explicitly request extended thinking (Generator A).
        # For Generator A (thinking.enabled=True) we prepend /think instead so
        # the model reasons before answering complex SQL generation tasks.
        if thinking and thinking.enabled:
            prefix = "/think"
        else:
            prefix = "/no_think"
        system_text = f"{prefix}\n\n{system_text}" if system_text else prefix

        full_messages: list[dict[str, Any]] = []
        if system_text:
            full_messages.append({"role": "system", "content": system_text})
        full_messages.extend(messages)

        active_tool = _select_tool(tools, tool_choice_name)

        if active_tool is None:
            return await self._call_plain_text(model, full_messages, max_tokens, temperature)

        response_model = _json_schema_to_pydantic(active_tool)

        try:
            return await self._call_with_retry(
                model=model,
                messages=full_messages,
                response_model=response_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMError:
            raise
        except Exception as exc:
            if _is_rate_limit(exc):
                raise LLMRateLimitError(
                    f"MLX server rate-limited/overloaded (model={model}): {exc}"
                ) from exc
            raise LLMError(
                f"MLX call failed (model={model}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Single attempt with tenacity retry on any exception."""
        result_obj = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        tool_dict = result_obj.model_dump()
        input_tok, output_tok = _extract_tokens(result_obj)
        return LLMResponse(
            tool_inputs=[tool_dict],
            text=None,
            thinking=None,
            input_tokens=input_tok,
            output_tokens=output_tok,
            finish_reason="json",
        )

    async def _call_plain_text(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Plain text completion when no tools are provided (edge case)."""
        response = await self._raw_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            tool_inputs=[],
            text=text or None,
            thinking=None,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=response.choices[0].finish_reason,
        )
