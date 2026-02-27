"""
Unit tests for src/monitoring/fallback_tracker.py.

All tests use the autouse `isolate_tracker` fixture for isolation.
No live API calls; no external dependencies.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.monitoring.fallback_tracker import (
    ComponentTracker,
    FallbackEvent,
    FallbackTracker,
    get_tracker,
    reset_tracker,
)


@pytest.fixture(autouse=True)
def isolate_tracker():
    """Reset the singleton before and after each test."""
    reset_tracker()
    yield
    reset_tracker()


# ---------------------------------------------------------------------------
# FallbackEvent construction
# ---------------------------------------------------------------------------

class TestFallbackEvent:
    def test_defaults(self):
        ev = FallbackEvent(
            component="test_comp",
            trigger="some_trigger",
            action="some_action",
            severity="warning",
        )
        assert ev.component == "test_comp"
        assert ev.trigger == "some_trigger"
        assert ev.action == "some_action"
        assert ev.severity == "warning"
        assert ev.details == {}
        assert isinstance(ev.timestamp, datetime)
        assert ev.timestamp.tzinfo is not None  # must be timezone-aware

    def test_custom_details(self):
        ev = FallbackEvent(
            component="c",
            trigger="t",
            action="a",
            severity="error",
            details={"model": "gemini-2.5-flash", "count": 3},
        )
        assert ev.details["model"] == "gemini-2.5-flash"
        assert ev.details["count"] == 3

    def test_timestamp_is_utc(self):
        ev = FallbackEvent("c", "t", "a", "warning")
        assert ev.timestamp.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# FallbackTracker: record / get_events round-trip
# ---------------------------------------------------------------------------

class TestFallbackTrackerRecord:
    def test_record_and_retrieve(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        ev = FallbackEvent(
            component="schema_linker",
            trigger="llm_error",
            action="faiss_top15_as_s1",
            severity="error",
        )
        tracker.record(ev)
        events = tracker.get_events()
        assert len(events) == 1
        assert events[0].component == "schema_linker"
        assert events[0].trigger == "llm_error"
        assert events[0].action == "faiss_top15_as_s1"

    def test_get_events_returns_copy(self):
        """Mutating the returned list must not affect the tracker's internal state."""
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("c", "t", "a", "warning"))
        events = tracker.get_events()
        events.clear()
        assert len(tracker.get_events()) == 1

    def test_multiple_events_ordered(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        for i in range(5):
            tracker.record(FallbackEvent("c", f"trigger_{i}", "a", "warning"))
        events = tracker.get_events()
        assert len(events) == 5
        assert [e.trigger for e in events] == [f"trigger_{i}" for i in range(5)]

    def test_disabled_tracker_records_nothing(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=False, log_file=None)
        tracker.record(FallbackEvent("c", "t", "a", "error"))
        assert tracker.get_events() == []

    def test_record_emits_warning_log(self, caplog):
        import logging
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        with caplog.at_level(logging.WARNING, logger="src.monitoring.fallback_tracker"):
            tracker.record(FallbackEvent("my_comp", "rate_limit", "model_fallback", "warning"))
        assert "my_comp" in caplog.text
        assert "rate_limit" in caplog.text

    def test_record_emits_error_log(self, caplog):
        import logging
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        with caplog.at_level(logging.ERROR, logger="src.monitoring.fallback_tracker"):
            tracker.record(FallbackEvent("gemini", "malformed_function_call", "raise_error", "error"))
        assert "[fallback]" in caplog.text


# ---------------------------------------------------------------------------
# FallbackTracker: get_summary aggregation
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_empty_summary(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        assert tracker.get_summary() == {}

    def test_single_event_summary(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("gemini_client", "max_tokens", "token_escalation", "warning"))
        summary = tracker.get_summary()
        assert summary == {"gemini_client": {"max_tokens": 1}}

    def test_aggregation_by_component_and_trigger(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("schema_linker", "llm_error", "faiss_top15_as_s1", "error"))
        tracker.record(FallbackEvent("schema_linker", "llm_error", "s1_as_s2", "error"))
        tracker.record(FallbackEvent("schema_linker", "validation_error", "empty_result", "warning"))
        tracker.record(FallbackEvent("query_fixer", "llm_error", "fix_loop_break", "warning"))
        summary = tracker.get_summary()
        assert summary["schema_linker"]["llm_error"] == 2
        assert summary["schema_linker"]["validation_error"] == 1
        assert summary["query_fixer"]["llm_error"] == 1

    def test_different_actions_same_trigger_counted_together(self):
        """get_summary aggregates by (component, trigger), not by action."""
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("comp", "llm_error", "action_a", "error"))
        tracker.record(FallbackEvent("comp", "llm_error", "action_b", "error"))
        summary = tracker.get_summary()
        assert summary["comp"]["llm_error"] == 2

    def test_multiple_components(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("llm_base", "rate_limit", "model_fallback", "warning"))
        tracker.record(FallbackEvent("gemini_client", "max_tokens", "token_escalation", "warning"))
        tracker.record(FallbackEvent("cache_manager", "io_error", "cache_miss", "warning"))
        summary = tracker.get_summary()
        assert set(summary.keys()) == {"llm_base", "gemini_client", "cache_manager"}


# ---------------------------------------------------------------------------
# FallbackTracker: reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_events(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("c", "t", "a", "warning"))
        tracker.record(FallbackEvent("c", "t2", "a2", "error"))
        tracker.reset()
        assert tracker.get_events() == []

    def test_reset_clears_summary(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("c", "t", "a", "warning"))
        tracker.reset()
        assert tracker.get_summary() == {}

    def test_can_record_after_reset(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        tracker.record(FallbackEvent("c", "t", "a", "warning"))
        tracker.reset()
        tracker.record(FallbackEvent("c2", "t2", "a2", "error"))
        events = tracker.get_events()
        assert len(events) == 1
        assert events[0].component == "c2"


# ---------------------------------------------------------------------------
# FallbackTracker: JSONL file writing
# ---------------------------------------------------------------------------

class TestJsonlWriting:
    def test_jsonl_written_on_record(self, tmp_path):
        log_file = tmp_path / "fallbacks.jsonl"
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=str(log_file))
        tracker.record(FallbackEvent(
            component="cache_manager",
            trigger="io_error",
            action="cache_miss",
            severity="warning",
            details={"path": "/tmp/test.json"},
        ))
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["component"] == "cache_manager"
        assert row["trigger"] == "io_error"
        assert row["action"] == "cache_miss"
        assert row["severity"] == "warning"
        assert row["details"]["path"] == "/tmp/test.json"
        # Timestamp must be an ISO string parseable back to a datetime
        parsed_ts = datetime.fromisoformat(row["timestamp"])
        assert parsed_ts.tzinfo is not None

    def test_multiple_events_append_to_jsonl(self, tmp_path):
        log_file = tmp_path / "fallbacks.jsonl"
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=str(log_file))
        for i in range(3):
            tracker.record(FallbackEvent("c", f"t{i}", "a", "warning"))
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_jsonl_write_failure_is_silent(self, tmp_path):
        """An OSError on write must not propagate — best-effort only."""
        log_file = tmp_path / "fallbacks.jsonl"
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=str(log_file))
        with patch("builtins.open", side_effect=OSError("disk full")):
            # Must not raise
            tracker.record(FallbackEvent("c", "t", "a", "warning"))
        # Event is still collected in memory
        assert len(tracker.get_events()) == 1

    def test_no_file_write_when_log_file_is_none(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        with patch("builtins.open") as mock_file:
            tracker.record(FallbackEvent("c", "t", "a", "warning"))
            mock_file.assert_not_called()

    def test_jsonl_creates_parent_dirs(self, tmp_path):
        log_file = tmp_path / "nested" / "dir" / "fallbacks.jsonl"
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=str(log_file))
        tracker.record(FallbackEvent("c", "t", "a", "warning"))
        assert log_file.exists()


# ---------------------------------------------------------------------------
# ComponentTracker convenience wrapper
# ---------------------------------------------------------------------------

class TestComponentTracker:
    def test_component_pre_filled(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        ct = tracker.module_tracker("schema_linker")
        ct.record(trigger="llm_error", action="faiss_top15_as_s1")
        events = tracker.get_events()
        assert len(events) == 1
        assert events[0].component == "schema_linker"
        assert events[0].trigger == "llm_error"
        assert events[0].action == "faiss_top15_as_s1"
        assert events[0].severity == "warning"  # default

    def test_severity_override(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        ct = tracker.module_tracker("gemini_client")
        ct.record(trigger="malformed_function_call", action="raise_error", severity="error")
        assert tracker.get_events()[0].severity == "error"

    def test_details_passed_through(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        ct = tracker.module_tracker("query_fixer")
        ct.record(
            trigger="llm_error",
            action="fix_loop_break",
            details={"candidate_id": "standard_B1_s1", "iteration": 2},
        )
        ev = tracker.get_events()[0]
        assert ev.details["candidate_id"] == "standard_B1_s1"
        assert ev.details["iteration"] == 2

    def test_multiple_component_trackers_share_underlying_tracker(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        ct1 = tracker.module_tracker("comp_a")
        ct2 = tracker.module_tracker("comp_b")
        ct1.record(trigger="t1", action="a1")
        ct2.record(trigger="t2", action="a2")
        events = tracker.get_events()
        assert len(events) == 2
        assert {e.component for e in events} == {"comp_a", "comp_b"}

    def test_default_details_is_empty_dict(self):
        tracker = FallbackTracker()
        tracker.configure(enabled=True, log_file=None)
        ct = tracker.module_tracker("x")
        ct.record(trigger="t", action="a")
        assert tracker.get_events()[0].details == {}


# ---------------------------------------------------------------------------
# Singleton: get_tracker() and reset_tracker()
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_tracker_returns_same_instance(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_reset_tracker_gives_fresh_instance(self):
        t1 = get_tracker()
        t1.record(FallbackEvent("c", "t", "a", "warning"))
        reset_tracker()
        t2 = get_tracker()
        assert t2 is not t1
        assert t2.get_events() == []

    def test_singleton_accepts_events(self):
        tracker = get_tracker()
        tracker.record(FallbackEvent("singleton_test", "trigger", "action", "warning"))
        assert len(get_tracker().get_events()) == 1

    def test_reset_tracker_clears_events_of_old_instance(self):
        tracker = get_tracker()
        tracker.record(FallbackEvent("c", "t", "a", "warning"))
        assert len(tracker.get_events()) == 1
        reset_tracker()
        # Old reference still has events cleared (reset() was called)
        # New singleton is fresh
        new_tracker = get_tracker()
        assert len(new_tracker.get_events()) == 0
