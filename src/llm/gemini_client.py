"""
Google Gemini implementation of LLMClient.

Uses the google-genai SDK (install with: pip install -e ".[gemini]").
Features:
- Async calls via client.aio.models.generate_content
- Function calling (tool use) with forced tool selection
- Thinking mode via ThinkingConfig (Gemini 2.5+ models)
- Tenacity retry: 3 attempts, exponential backoff 2–30 seconds

Note: Prompt caching (CacheableText.cache hint) is not applied. Gemini's
context caching requires a separate API call and 1,024+ token minimum —
this can be added to this client later without changing any call sites.
"""

from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.base import CacheableText, LLMClient, LLMError, LLMResponse, ThinkingConfig, ToolParam


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

        system_instruction = "\n\n".join(b.text for b in system)
        tool_defs = [_to_gemini_tool(t) for t in tools]
        contents = _to_gemini_contents(messages, types)

        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=[tool_choice_name] if tool_choice_name else None,
            )
        )

        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=tool_defs)],
            tool_config=tool_config,
            max_output_tokens=max_tokens,
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        if thinking and thinking.enabled:
            gen_config.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking.budget_tokens,
                include_thoughts=True,
            )

        try:
            raw = await self._call_with_retry(model=model, contents=contents, config=gen_config)
        except Exception as exc:
            raise LLMError(f"Gemini API call failed after retries: {exc}") from exc

        return _parse_response(raw)

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
    tool_inputs: list[dict[str, Any]] = []
    text_parts: list[str] = []
    thinking_text: Optional[str] = None

    for part in raw.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call:
            # function_call.args is a MapComposite; convert to plain dict
            tool_inputs.append(dict(part.function_call.args))
        elif getattr(part, "thought", False):
            thinking_text = part.text
        elif part.text:
            text_parts.append(part.text)

    return LLMResponse(
        tool_inputs=tool_inputs,
        text="\n".join(text_parts) or None,
        thinking=thinking_text,
        input_tokens=getattr(raw.usage_metadata, "prompt_token_count", 0) or 0,
        output_tokens=getattr(raw.usage_metadata, "candidates_token_count", 0) or 0,
    )
