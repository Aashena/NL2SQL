"""
SQLite execution utilities with timeout and safety guards.

Key safeguards:
  - Configurable wall-clock timeout via a background thread
  - Cartesian product protection: reject results > 10,000 rows if no LIMIT clause
  - Always closes the DB connection, even on error
"""

import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    success: bool
    rows: list
    error: Optional[str]
    execution_time: float
    is_empty: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)

_CARTESIAN_ROW_LIMIT = 10_000


def _has_limit_clause(sql: str) -> bool:
    return bool(_LIMIT_RE.search(sql))


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

def execute_sql(
    db_path: str,
    sql: str,
    timeout: float = 30.0,
) -> ExecutionResult:
    """
    Execute *sql* against the SQLite database at *db_path*.

    Parameters
    ----------
    db_path:
        Filesystem path to the ``.sqlite`` / ``.db`` file.
    sql:
        SQL statement to execute.
    timeout:
        Wall-clock time limit in seconds.  A background thread interrupts the
        SQLite connection if the query takes longer.

    Returns
    -------
    ExecutionResult
        Always returned (never raises); check ``success`` for status.
    """
    start = time.perf_counter()
    conn: Optional[sqlite3.Connection] = None

    # Shared state between the main thread and the timeout thread
    result_container: dict = {"done": False}

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)

        # -------------------------------------------------------------------
        # Timeout via a daemon thread that calls conn.interrupt()
        # -------------------------------------------------------------------
        def _interrupt_after(seconds: float):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if result_container["done"]:
                    return
                time.sleep(0.001)
            if not result_container["done"]:
                try:
                    conn.interrupt()
                except Exception:
                    pass

        timer = threading.Thread(target=_interrupt_after, args=(timeout,), daemon=True)
        timer.start()

        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            # sqlite3 raises OperationalError for both syntax errors and
            # interrupted queries (timeout).
            result_container["done"] = True
            elapsed = time.perf_counter() - start
            err_msg = str(exc)
            # Distinguish timeout from other operational errors
            if "interrupted" in err_msg.lower():
                err_msg = f"Query timed out after {timeout}s"
            return ExecutionResult(
                success=False,
                rows=[],
                error=err_msg,
                execution_time=elapsed,
                is_empty=False,
            )
        except sqlite3.DatabaseError as exc:
            result_container["done"] = True
            elapsed = time.perf_counter() - start
            return ExecutionResult(
                success=False,
                rows=[],
                error=str(exc),
                execution_time=elapsed,
                is_empty=False,
            )

        result_container["done"] = True
        elapsed = time.perf_counter() - start

        # -------------------------------------------------------------------
        # Cartesian product guard
        # -------------------------------------------------------------------
        if len(rows) > _CARTESIAN_ROW_LIMIT and not _has_limit_clause(sql):
            return ExecutionResult(
                success=False,
                rows=[],
                error=(
                    f"Query returned {len(rows):,} rows without a LIMIT clause — "
                    "possible cartesian product. Rejected for safety."
                ),
                execution_time=elapsed,
                is_empty=False,
            )

        return ExecutionResult(
            success=True,
            rows=rows,
            error=None,
            execution_time=elapsed,
            is_empty=len(rows) == 0,
        )

    except Exception as exc:
        result_container["done"] = True
        elapsed = time.perf_counter() - start
        return ExecutionResult(
            success=False,
            rows=[],
            error=str(exc),
            execution_time=elapsed,
            is_empty=False,
        )
    finally:
        result_container["done"] = True
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
