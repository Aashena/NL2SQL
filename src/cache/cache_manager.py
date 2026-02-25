"""
LLM Response Cache Manager

Provides disk-based caching for LLM responses keyed by SHA256 of model + messages.
Controlled by the CACHE_LLM_RESPONSES setting (default: False).

Cache format: JSON files {cache_key[:16]}.json with fields:
  key, model, prompt_hash, response, timestamp, token_count
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from src.llm.base import LLMResponse
from src.config.settings import settings
from src.monitoring.fallback_tracker import FallbackEvent, get_tracker

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Disk-based cache for LLM responses.

    Cache keys are SHA256 of (model + JSON-serialized messages).
    Cache is a directory of JSON files named by the first 16 hex chars of the key.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir if cache_dir is not None else settings.cache_dir)
        self._enabled = enabled if enabled is not None else settings.cache_llm_responses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, model: str, messages: list[dict]) -> str:
        """Compute SHA256(model + json.dumps(messages, sort_keys=True))."""
        payload = model + json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key[:16]}.json"

    @staticmethod
    def _response_to_dict(response: LLMResponse) -> dict[str, Any]:
        return {
            "tool_inputs": response.tool_inputs,
            "text": response.text,
            "thinking": response.thinking,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }

    @staticmethod
    def _dict_to_response(data: dict[str, Any]) -> LLMResponse:
        return LLMResponse(
            tool_inputs=data.get("tool_inputs", []),
            text=data.get("text"),
            thinking=data.get("thinking"),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str, ttl: Optional[int] = None) -> Optional[LLMResponse]:
        """
        Retrieve a cached LLMResponse by key.

        Parameters
        ----------
        key:
            Cache key produced by _make_key().
        ttl:
            If not None, reject entries older than ttl seconds.

        Returns
        -------
        LLMResponse if a valid cache entry exists, else None.
        """
        if not self._enabled:
            return None

        path = self._cache_path(key)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache read error for %s: %s", path, exc)
            get_tracker().record(FallbackEvent(
                component="cache_manager",
                trigger="io_error",
                action="cache_miss",
                details={"path": str(path), "error": str(exc)},
            ))
            return None

        # Validate that this entry has the right full key (collision check)
        if entry.get("key") != key:
            logger.debug("Cache key mismatch for %s — ignoring", path)
            return None

        # TTL check
        if ttl is not None:
            age = time.time() - entry.get("timestamp", 0)
            if age > ttl:
                logger.debug("Cache entry expired for key %s (age=%.1fs)", key[:16], age)
                return None

        logger.debug("Cache hit for key %s", key[:16])
        return self._dict_to_response(entry["response"])

    async def set(
        self,
        key: str,
        response: LLMResponse,
        model: str = "",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Write a LLMResponse to the cache.

        Parameters
        ----------
        key:
            Cache key produced by _make_key().
        response:
            The LLMResponse to cache.
        model:
            Model identifier (stored for debugging; not used for retrieval).
        ttl:
            Not enforced at write time; used only at read time via get().
        """
        if not self._enabled:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)

        entry = {
            "key": key,
            "model": model,
            "prompt_hash": key,
            "response": self._response_to_dict(response),
            "timestamp": time.time(),
            "token_count": response.input_tokens + response.output_tokens,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            logger.debug("Cache write for key %s", key[:16])
        except OSError as exc:
            logger.warning("Cache write error for %s: %s", path, exc)
            get_tracker().record(FallbackEvent(
                component="cache_manager",
                trigger="io_error",
                action="skip_cache_write",
                details={"path": str(path), "error": str(exc)},
            ))

    def cached(self, model: str, ttl: Optional[int] = None):
        """
        Decorator factory for async functions that call the LLM.

        The decorated function must accept `messages` as a keyword argument
        (or as the first positional argument after `self` if it is a method,
        but for standalone functions it is the first positional arg).

        On cache hit, returns the cached LLMResponse without calling the
        wrapped function. On miss, calls the function, caches the result,
        and returns it.

        Usage
        -----
        @cache_manager.cached(model=settings.model_fast)
        async def call_llm(messages, **kwargs) -> LLMResponse:
            ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract messages from kwargs or the first positional arg
                messages = kwargs.get("messages")
                if messages is None and args:
                    messages = args[0]
                if messages is None:
                    messages = []

                key = self._make_key(model, messages)
                cached_response = await self.get(key, ttl=ttl)
                if cached_response is not None:
                    return cached_response

                result = await func(*args, **kwargs)

                if isinstance(result, LLMResponse):
                    await self.set(key, result, model=model, ttl=ttl)

                return result

            return wrapper

        return decorator
