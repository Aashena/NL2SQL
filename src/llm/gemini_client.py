"""
Google Gemini implementation of LLMClient.

Uses the google-genai SDK (install with: pip install -e ".[gemini]").
Features:
- Async calls via client.aio.models.generate_content
- Function calling (tool use) with forced tool selection
- Thinking mode via ThinkingConfig (Gemini 2.5+ models)
- Tenacity retry: 3 attempts, exponential backoff 2–30 seconds
- Context caching: CacheableText(cache=True) blocks are cached via the
  Gemini context caching API (requires ≥1,024 tokens; silently skipped if
  the model or content doesn't meet the threshold).
"""

import asyncio
import hashlib
import logging
import os
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.base import CacheableText, LLMClient, LLMError, LLMResponse, ThinkingConfig, ToolParam

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):

    def __init__(self, api_key: str) -> None:
        try:
            from google import genai  # type: ignore[import]
            self._genai = genai
            self._client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini support. "
                "Install it with: pip install -e '.[gemini]'"
            ) from exc
        # Maps SHA256(model:cacheable_text) → Gemini cache name
        self._cache_store: dict[str, str] = {}
        # Lazy-initialized semaphore limiting concurrent calls (prevents rate limit errors)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent = int(os.environ.get("GEMINI_MAX_CONCURRENT", "5"))

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
        from google.genai import types  # type: ignore[import]

        tool_defs = [_to_gemini_tool(t) for t in tools]
        contents = _to_gemini_contents(messages, types)

        # Only build tool_config when there are actual function declarations;
        # Gemini API rejects ToolConfig with mode=ANY and empty function_declarations.
        if tool_defs:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_choice_name] if tool_choice_name else None,
                )
            )
        else:
            tool_config = None

        # --- Context caching logic ---
        cacheable_blocks = [b for b in system if b.cache]
        non_cacheable_blocks = [b for b in system if not b.cache]

        cache_name: Optional[str] = None
        if cacheable_blocks:
            cacheable_text = "\n\n".join(b.text for b in cacheable_blocks)
            cache_key = hashlib.sha256(f"{model}:{cacheable_text}".encode()).hexdigest()

            if cache_key in self._cache_store:
                cache_name = self._cache_store[cache_key]
                logger.debug("Gemini context cache hit: %s", cache_name)
            else:
                cache_name = await self._try_create_cache(model, cacheable_text)
                if cache_name is not None:
                    self._cache_store[cache_key] = cache_name

        # --- Build tool/tool_config kwargs (omit entirely when no tools) ---
        tool_kwargs: dict = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if tool_defs:
            tool_kwargs["tools"] = [types.Tool(function_declarations=tool_defs)]
            tool_kwargs["tool_config"] = tool_config

        # --- Build GenerateContentConfig ---
        if cache_name is not None:
            # Cached content already contains the cacheable system text
            if non_cacheable_blocks:
                non_cacheable_text = "\n\n".join(b.text for b in non_cacheable_blocks)
                gen_config = types.GenerateContentConfig(
                    system_instruction=non_cacheable_text,
                    cached_content=cache_name,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    **tool_kwargs,
                )
            else:
                # No non-cacheable blocks — omit system_instruction entirely
                gen_config = types.GenerateContentConfig(
                    cached_content=cache_name,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    **tool_kwargs,
                )
        else:
            # Fall back to current behavior: combine all system blocks
            system_instruction = "\n\n".join(b.text for b in system)
            gen_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=max_tokens,
                temperature=temperature,
                **tool_kwargs,
            )

        if thinking and thinking.enabled:
            gen_config.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking.budget_tokens,
                include_thoughts=True,
            )

        try:
            async with self._get_semaphore():
                raw = await self._call_with_retry(model=model, contents=contents, config=gen_config)
            parsed = _parse_response(raw)

            # One-shot retry with doubled token budget when the model was truncated before
            # producing any output (finish_reason=MAX_TOKENS, no text, no tool calls).
            if parsed.text is None and not parsed.tool_inputs and "MAX_TOKENS" in str(parsed.finish_reason):
                retry_max = min(max_tokens * 2, 8192)
                logger.warning(
                    "MAX_TOKENS hit with no output (max_tokens=%d); retrying with max_tokens=%d",
                    max_tokens,
                    retry_max,
                )
                gen_config.max_output_tokens = retry_max
                async with self._get_semaphore():
                    raw = await self._call_with_retry(model=model, contents=contents, config=gen_config)
                parsed = _parse_response(raw)

            return parsed
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini API call failed after retries: {exc}") from exc

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the concurrency semaphore (must be called from async context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def _try_create_cache(self, model: str, system_text: str) -> Optional[str]:
        """Try to create a Gemini context cache. Returns cache name on success, None on failure."""
        from google.genai import types  # type: ignore[import]
        try:
            cached = await self._client.aio.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_text,
                    ttl="3600s",
                ),
            )
            logger.debug("Created Gemini context cache: %s", cached.name)
            return cached.name
        except Exception as exc:
            # Common failures: too few tokens (< 1024), model doesn't support caching
            logger.debug("Gemini context caching failed (proceeding without cache): %s", exc)
            return None

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_with_retry(self, *, model: str, contents: Any, config: Any) -> Any:
        return await self._client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gemini_tool(t: ToolParam) -> dict[str, Any]:
    """Convert a ToolParam to a Gemini function_declaration dict.

    Gemini uses "parameters" where Anthropic uses "input_schema" — same JSON Schema object.
    """
    return {
        "name": t.name,
        "description": t.description,
        "parameters": t.input_schema,
    }


def _to_gemini_contents(messages: list[dict[str, Any]], types: Any) -> list[Any]:
    """Convert standard messages list to Gemini Content objects."""
    role_map = {"user": "user", "assistant": "model"}
    result = []
    for msg in messages:
        role = role_map.get(msg["role"], "user")
        content = msg["content"]
        if isinstance(content, str):
            parts = [types.Part(text=content)]
        else:
            # Handle list-of-blocks format (e.g. tool result blocks)
            parts = [
                types.Part(text=b.get("text", ""))
                for b in content
                if b.get("type") == "text"
            ]
        result.append(types.Content(role=role, parts=parts))
    return result


def _parse_response(raw: Any) -> LLMResponse:
    from src.llm.base import LLMError  # avoid circular at module level

    tool_inputs: list[dict[str, Any]] = []
    text_parts: list[str] = []
    thinking_text: Optional[str] = None

    if not raw.candidates:
        raise LLMError("Gemini returned no candidates")

    candidate = raw.candidates[0]
    if candidate.content is None:
        finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
        raise LLMError(
            f"Gemini candidate has no content (finish_reason={finish_reason})"
        )

    finish_reason = str(getattr(candidate, "finish_reason", "UNKNOWN"))

    for part in (candidate.content.parts or []):
        if hasattr(part, "function_call") and part.function_call:
            # function_call.args is a MapComposite; convert to plain dict
            tool_inputs.append(dict(part.function_call.args))
        elif getattr(part, "thought", False):
            thinking_text = part.text
        elif part.text:
            text_parts.append(part.text)

    if not text_parts and not tool_inputs:
        safety_ratings = getattr(candidate, "safety_ratings", None)
        parts_summary = [
            (getattr(p, "thought", False), bool(getattr(p, "text", None)))
            for p in (candidate.content.parts or [])
        ]
        logger.warning(
            "Gemini response has no text or tool output: "
            "finish_reason=%s, safety_ratings=%s, parts(thought,has_text)=%s",
            finish_reason,
            safety_ratings,
            parts_summary,
        )

    return LLMResponse(
        tool_inputs=tool_inputs,
        text="\n".join(text_parts) or None,
        thinking=thinking_text,
        input_tokens=getattr(raw.usage_metadata, "prompt_token_count", 0) or 0,
        output_tokens=getattr(raw.usage_metadata, "candidates_token_count", 0) or 0,
        finish_reason=finish_reason,
    )
