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
    - ```sql ... ``` fences (any language tag: sql, SQL, sqlite, etc.)
    - ``` ... ``` fences (no language tag)
    - Orphan opening fence with no closing ``` (model truncation)
    - Inline single-backtick: `SELECT ...`
    - Trailing semicolons
    - Runs of whitespace collapsed to a single space
    """
    if not raw:
        return ""

    text = raw.strip()

    # Strategy 1 (full fence): match any opening ```<lang> ... ``` pair.
    # Accepts any alphanumeric language tag (sql, SQL, sqlite, SQL WITH, etc.)
    # or no tag at all.  Uses DOTALL so newlines inside the fence are captured.
    fence_match = re.search(
        r"```[a-zA-Z0-9]*\s*\r?\n?(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # Strategy 2 (orphan opening fence): the model stopped generating before
        # emitting the closing ```.  Strip the opening fence and any stray trailing
        # ``` that might appear at the very end of the string.
        orphan_match = re.match(r"^```[a-zA-Z0-9]*\s*\r?\n?", text, flags=re.IGNORECASE)
        if orphan_match:
            text = text[orphan_match.end():]
            # Also strip a trailing ``` if present (safety measure)
            text = re.sub(r"```\s*$", "", text).strip()
        else:
            # Strategy 3 (inline single-backtick): `SELECT ...`
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

        ## Evidence (use for thresholds and domain values)
        {evidence}

        Relevant cell values from the database:
        - {table}.{column} = '{value}'   (for each cell match)

        Write a SQL query that answers the question.

    Evidence is placed in a prominent labeled section so the model treats it as
    an authoritative constraint (e.g. threshold values, domain definitions) rather
    than supplementary text.
    """
    lines: list[str] = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("## Evidence (use for thresholds and domain values)")
    lines.append(evidence if evidence else "None")
    lines.append("")

    if cell_matches:
        lines.append("Relevant cell values from the database:")
        for match in cell_matches:
            lines.append(
                f"- {match.table}.{match.column} = '{match.matched_value}'"
            )
        lines.append("")

    lines.append("Write a SQL query that answers the question.")
    lines.append("")
    lines.append("## SQL Writing Rules")
    lines.append(
        "1. SELECT completeness: Return ALL columns the question explicitly asks for. "
        "If the question asks \"what is the X and Y\", your SELECT must include both X and Y."
    )
    lines.append(
        "2. Value mappings: If the evidence defines a mapping (e.g., label = '+' means 'YES', "
        "'M' means 'Male'), apply that transformation using CASE or IIF in your SELECT — "
        "do not return the raw stored value."
    )
    lines.append(
        "3. Ratio direction: For questions asking \"how many times is A compared to B\", compute "
        "A/B (not B/A). Use DISTINCT only when the question implies uniqueness (\"unique\", "
        "\"distinct\", \"different\")."
    )
    lines.append(
        "4. Evidence scope: The evidence provides schema context (column meanings, value "
        "definitions). Use it to understand column semantics and filtering conditions. "
        "Trust the question wording to determine the output format, number of rows, "
        "and whether to aggregate."
    )
    lines.append(
        "5. Binary output format: When the question asks whether something is true/false, "
        "yes/no, or carcinogenic/not carcinogenic, return 'YES' or 'NO' (not full words like "
        "'carcinogenic' or 'not carcinogenic'). Use IIF(condition, 'YES', 'NO') or "
        "CASE WHEN condition THEN 'YES' ELSE 'NO' END."
    )

    return "\n".join(lines)
