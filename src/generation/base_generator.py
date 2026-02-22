"""
Shared generation utilities for all SQL generators.

Provides:
- SQLCandidate dataclass
- clean_sql() — strip markdown fences, semicolons, normalize whitespace
- validate_sql_syntax() — regex-based structural check (no DB execution)
- build_base_prompt() — shared user prompt string
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SQLCandidate:
    """A single SQL candidate produced by one generator."""
    sql: str
    generator_id: str
    schema_used: str           # "s1" or "s2"
    schema_format: str         # "ddl" or "markdown"
    reasoning_trace: Optional[str] = None
    error_flag: bool = False


def clean_sql(raw: str) -> str:
    """Extract SQL from markdown code fences; strip semicolons; normalize whitespace.

    Handles:
    - ```sql ... ``` fences
    - ``` ... ``` fences (no language tag)
    - Trailing semicolons
    - Runs of whitespace collapsed to a single space
    """
    if not raw:
        return ""

    text = raw.strip()

    # Strip markdown code fences — e.g. ```sql\n...\n``` or ```\n...\n```
    # Pattern: optional ```<lang>\n ... \n``` with possible leading/trailing whitespace
    fence_match = re.search(
        r"```(?:sql|SQL)?\s*\n?(.*?)```",
        text,
        flags=re.DOTALL,
    )
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # Also handle inline fences without newlines: `SELECT ...`
        inline_match = re.match(r"^`([^`]+)`$", text, flags=re.DOTALL)
        if inline_match:
            text = inline_match.group(1).strip()

    # Remove trailing semicolons (possibly surrounded by whitespace)
    text = text.rstrip("; \t\n\r")

    # Normalize whitespace: collapse runs of whitespace (spaces, tabs, newlines)
    # to a single space, then strip leading/trailing whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_sql_syntax(sql: str) -> bool:
    """Lightweight regex-based SQL syntax check. Not execution-based.

    Returns True if the string looks like a valid SQL query:
    - Non-empty
    - Contains SELECT (case-insensitive)
    - Contains FROM (case-insensitive)
    """
    if not sql or not sql.strip():
        return False

    upper = sql.upper()
    if "SELECT" not in upper:
        return False
    if "FROM" not in upper:
        return False

    return True


def build_base_prompt(question: str, evidence: str, cell_matches: list) -> str:
    """Build the shared user prompt portion used by all generators.

    Format:
        Question: {question}
        Evidence: {evidence}

        Relevant cell values from the database:
        - {table}.{column} = '{value}'   (for each cell match)

        Write a SQL query that answers the question.
    """
    lines: list[str] = []
    lines.append(f"Question: {question}")
    lines.append(f"Evidence: {evidence if evidence else 'None'}")
    lines.append("")

    if cell_matches:
        lines.append("Relevant cell values from the database:")
        for match in cell_matches:
            lines.append(
                f"- {match.table}.{match.column} = '{match.matched_value}'"
            )
        lines.append("")

    lines.append("Write a SQL query that answers the question.")

    return "\n".join(lines)
