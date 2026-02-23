"""
Tests for src/cache/cache_manager.py

7 tests covering caching behaviour, key determinism, and disabled mode.
"""

from __future__ import annotations

import json
import time

import pytest

from src.cache.cache_manager import CacheManager
from src.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(**kwargs) -> LLMResponse:
    defaults = dict(
        tool_inputs=[{"result": "SELECT 1"}],
        text="some text",
        thinking=None,
        input_tokens=100,
        output_tokens=50,
    )
    defaults.update(kwargs)
    return LLMResponse(**defaults)


def _make_messages() -> list[dict]:
    return [{"role": "user", "content": "How many rows are in the table?"}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_miss_calls_api(tmp_path):
    """First call (no cache) must pass through and write to cache."""
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    messages = _make_messages()
    key = cache._make_key("model-fast", messages)

    # Nothing cached yet
    result = await cache.get(key)
    assert result is None

    # Write manually (simulates what the decorator does after an API call)
    response = _make_response()
    await cache.set(key, response, model="model-fast")

    # Now it should be cached
    cached = await cache.get(key)
    assert cached is not None


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_response(tmp_path):
    """Second call with same inputs returns the cached value."""
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    messages = _make_messages()
    key = cache._make_key("model-fast", messages)

    original = _make_response(text="first call")
    await cache.set(key, original, model="model-fast")

    # Second retrieval should return the cached version
    cached = await cache.get(key)
    assert cached is not None
    assert cached.text == "first call"


@pytest.mark.asyncio
async def test_cache_key_is_deterministic(tmp_path):
    """Same model + messages must always produce the same cache key."""
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    messages = _make_messages()

    key1 = cache._make_key("model-fast", messages)
    key2 = cache._make_key("model-fast", messages)

    assert key1 == key2
    assert len(key1) == 64  # SHA256 hex digest


@pytest.mark.asyncio
async def test_different_models_different_cache_keys(tmp_path):
    """Same messages but different models must produce different keys."""
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    messages = _make_messages()

    key_fast = cache._make_key("model-fast", messages)
    key_powerful = cache._make_key("model-powerful", messages)

    assert key_fast != key_powerful


@pytest.mark.asyncio
async def test_cache_disabled_by_config(tmp_path):
    """With enabled=False, get() always returns None and set() is a no-op."""
    cache = CacheManager(cache_dir=str(tmp_path), enabled=False)
    messages = _make_messages()
    key = cache._make_key("model-fast", messages)

    response = _make_response()
    await cache.set(key, response, model="model-fast")

    # Nothing should be written
    result = await cache.get(key)
    assert result is None

    # No files created
    files = list(tmp_path.iterdir())
    assert len(files) == 0


@pytest.mark.asyncio
async def test_cache_file_created_on_write(tmp_path):
    """After a cache miss + write, the JSON file must exist on disk."""
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    messages = _make_messages()
    key = cache._make_key("model-fast", messages)
    response = _make_response()

    await cache.set(key, response, model="model-fast")

    expected_path = tmp_path / f"{key[:16]}.json"
    assert expected_path.exists(), f"Cache file {expected_path} not found"

    # Verify the JSON structure
    with open(expected_path, encoding="utf-8") as f:
        entry = json.load(f)

    assert entry["key"] == key
    assert entry["model"] == "model-fast"
    assert "response" in entry
    assert "timestamp" in entry
    assert "token_count" in entry


@pytest.mark.asyncio
async def test_cache_roundtrip_preserves_response(tmp_path):
    """Cached response must be identical to the original LLMResponse."""
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    messages = _make_messages()
    key = cache._make_key("model-fast", messages)

    original = _make_response(
        tool_inputs=[{"sql": "SELECT COUNT(*) FROM frpm"}],
        text="explanation",
        thinking="step-by-step reasoning",
        input_tokens=200,
        output_tokens=75,
    )

    await cache.set(key, original, model="model-fast")
    retrieved = await cache.get(key)

    assert retrieved is not None
    assert retrieved.tool_inputs == original.tool_inputs
    assert retrieved.text == original.text
    assert retrieved.thinking == original.thinking
    assert retrieved.input_tokens == original.input_tokens
    assert retrieved.output_tokens == original.output_tokens
