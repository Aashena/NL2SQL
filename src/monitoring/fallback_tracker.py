"""
Fallback event tracking for the NL2SQL pipeline.

Records structured FallbackEvent objects whenever the pipeline uses a
degraded code path (model fallback, schema fallback, empty-result fallback,
token escalation, etc.).

Usage:
    from src.monitoring.fallback_tracker import get_tracker, FallbackEvent

    get_tracker().record(FallbackEvent(
        component="schema_linker",
        trigger="llm_error",
        action="faiss_top15_as_s1",
        details={"error": str(exc), "stage": "s1"},
        severity="error",
    ))

    # Or via ComponentTracker convenience wrapper:
    ct = get_tracker().module_tracker("schema_linker")
    ct.record(trigger="llm_error", action="faiss_top15_as_s1", details={"error": str(exc)})
"""

import json
import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FallbackEvent:
    """A single fallback event in the NL2SQL pipeline."""

    component: str
    """Which component triggered the fallback (e.g. 'gemini_client', 'schema_linker', 'query_fixer')."""

    trigger: str
    """What caused the fallback (e.g. 'rate_limit', 'malformed_function_call', 'max_tokens',
    'llm_error', 'validation_error', 'empty_output', 'io_error', 'empty_candidates')."""

    action: str
    """What the code did in response (e.g. 'model_fallback', 'faiss_fallback',
    'token_escalation', 'text_fallback', 's1_as_s2', 'empty_result',
    'fix_loop_break', 'default_winner_a', 'cache_miss')."""

    severity: str = "warning"
    """Log severity: 'warning' | 'error' | 'critical'. Defaults to 'warning'."""

    details: dict[str, Any] = field(default_factory=dict)
    """Extra diagnostic context. Must contain only JSON-serializable values
    (str, int, float, bool, list, dict). Use str(exc) for exceptions."""

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    """UTC timestamp of when the fallback occurred."""


class ComponentTracker:
    """
    Convenience wrapper around FallbackTracker with `component` pre-filled.

    Obtain via: tracker.module_tracker("my_component")
    """

    def __init__(self, tracker: "FallbackTracker", component: str) -> None:
        self._tracker = tracker
        self._component = component

    def record(
        self,
        trigger: str,
        action: str,
        details: Optional[dict[str, Any]] = None,
        severity: str = "warning",
    ) -> None:
        """Record a fallback event for this component."""
        self._tracker.record(
            FallbackEvent(
                component=self._component,
                trigger=trigger,
                action=action,
                details=details or {},
                severity=severity,
            )
        )


class FallbackTracker:
    """
    In-memory + optional JSONL-on-disk tracker for pipeline fallback events.

    Thread-safety: designed for a single-threaded asyncio event loop — no lock needed.
    JSONL writes are best-effort: any exception is caught and logged at DEBUG level,
    so the pipeline never fails due to tracker I/O.
    """

    def __init__(self) -> None:
        self._events: list[FallbackEvent] = []
        self._enabled: bool = True
        self._log_file: Optional[pathlib.Path] = None

    def configure(self, enabled: bool, log_file: Optional[str]) -> None:
        """
        Configure the tracker. Call once at startup with values from Settings.

        Separated from __init__ to allow the singleton to be created without
        importing Settings at module load time (avoids circular import risk).
        """
        self._enabled = enabled
        self._log_file = pathlib.Path(log_file) if log_file else None

    def record(self, event: FallbackEvent) -> None:
        """Append event to in-memory list, write to JSONL, and emit a structured log entry."""
        if not self._enabled:
            return
        self._events.append(event)
        self._maybe_write_jsonl(event)
        if event.severity == "critical":
            logger.critical(
                "[fallback] component=%s trigger=%s action=%s details=%r",
                event.component,
                event.trigger,
                event.action,
                event.details,
            )
        elif event.severity == "error":
            logger.error(
                "[fallback] component=%s trigger=%s action=%s details=%r",
                event.component,
                event.trigger,
                event.action,
                event.details,
            )
        else:
            logger.warning(
                "[fallback] component=%s trigger=%s action=%s details=%r",
                event.component,
                event.trigger,
                event.action,
                event.details,
            )

    def get_events(self) -> list[FallbackEvent]:
        """Return a shallow copy of all recorded events."""
        return list(self._events)

    def get_summary(self) -> dict[str, dict[str, int]]:
        """
        Aggregate event counts by (component, trigger).

        Returns:
            {
                "schema_linker": {"llm_error": 3, "validation_error": 1},
                "gemini_client": {"max_tokens": 2},
                ...
            }
        """
        summary: dict[str, dict[str, int]] = {}
        for ev in self._events:
            comp_dict = summary.setdefault(ev.component, {})
            comp_dict[ev.trigger] = comp_dict.get(ev.trigger, 0) + 1
        return summary

    def reset(self) -> None:
        """Clear all events. Use in tests for isolation between test cases."""
        self._events.clear()

    def module_tracker(self, component: str) -> ComponentTracker:
        """Return a ComponentTracker pre-bound to the given component name."""
        return ComponentTracker(self, component)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_write_jsonl(self, event: FallbackEvent) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "timestamp": event.timestamp.isoformat(),
                "component": event.component,
                "trigger": event.trigger,
                "action": event.action,
                "severity": event.severity,
                "details": event.details,
            }
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # best-effort: never crash the pipeline
            logger.debug("FallbackTracker JSONL write failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: Optional[FallbackTracker] = None


def get_tracker() -> FallbackTracker:
    """
    Return the module-level FallbackTracker singleton.

    Creates and configures it on first call using the current settings.
    Safe to call at import time of any module — Settings import happens
    only on first call (deferred), not at this module's load time.
    """
    global _tracker
    if _tracker is None:
        _tracker = FallbackTracker()
        try:
            from src.config.settings import settings  # noqa: PLC0415
            _tracker.configure(
                enabled=settings.fallback_tracking_enabled,
                log_file=settings.fallback_log_file,
            )
        except Exception:
            # If settings can't be loaded (e.g. in isolated unit tests), use safe defaults.
            _tracker.configure(enabled=True, log_file=None)
    return _tracker


def reset_tracker() -> None:
    """
    Reset the singleton. Primarily for test isolation.

    After calling this, the next get_tracker() re-creates and re-configures
    the tracker from settings.
    """
    global _tracker
    if _tracker is not None:
        _tracker.reset()
    _tracker = None
