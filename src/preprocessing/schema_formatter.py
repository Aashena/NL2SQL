"""
Op 0c: Schema Formatter

Converts DatabaseProfile + DatabaseSummary into two complementary text formats:

  DDL (SQL)   — Used by reasoning generators (Generator A, B2). Mimics SQL
                code pretraining format.  Includes CREATE TABLE blocks, column
                comments with long summaries, FK comments, and an example row.

  Markdown    — Used by standard/ICL generators (B1, C).  Produces a
                human-readable pipe table per table, with short summaries and
                sample values.

Results are cached to disk as plain text files.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.preprocessing.profiler import ColumnProfile, DatabaseProfile
from src.preprocessing.summarizer import FieldSummary, DatabaseSummary


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FormattedSchemas:
    db_id: str
    ddl: str       # Full DDL schema for all tables
    markdown: str  # Full Markdown schema for all tables


# ---------------------------------------------------------------------------
# Helper: column name quoting
# ---------------------------------------------------------------------------

# A "safe" identifier contains only letters, digits, and underscores.
_SAFE_IDENTIFIER_RE = re.compile(r'^[A-Za-z0-9_]+$')


def needs_quoting(name: str) -> bool:
    """
    Return True if a column name must be double-quoted in a DDL statement.

    Any name that contains characters outside [A-Za-z0-9_] — e.g. spaces,
    parentheses, hyphens, slashes, dots — needs quoting.
    """
    return not bool(_SAFE_IDENTIFIER_RE.match(name))


def _quote_col(name: str) -> str:
    """Return a DDL-safe representation of a column name."""
    if needs_quoting(name):
        # Escape any embedded double-quotes by doubling them (SQL standard)
        escaped = name.replace('"', '""')
        return f'"{escaped}"'
    return name


# ---------------------------------------------------------------------------
# Helper: summary lookup
# ---------------------------------------------------------------------------

def _build_summary_lookup(summary: DatabaseSummary) -> dict[tuple[str, str], FieldSummary]:
    """Build a (table_name, column_name) → FieldSummary dict for fast lookup."""
    return {
        (fs.table_name, fs.column_name): fs
        for fs in summary.field_summaries
    }


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------

_LONG_SUMMARY_MAX = 120
_SAMPLE_VALUE_MAX_DDL = 50  # truncation for sample row values in DDL comments


def _truncate(text: str, max_len: int, suffix: str = "...") -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + suffix


def _format_ddl_table(
    table_name: str,
    columns: list[ColumnProfile],
    summary_lookup: dict[tuple[str, str], FieldSummary],
    foreign_keys: list[tuple],
) -> str:
    """Generate the DDL block for a single table."""
    lines: list[str] = []

    # --- Table-level comment ---
    lines.append(f"-- Table: {table_name}")

    # Optional table description — use first column's long_summary as a proxy
    # (just skip if no summary available)

    lines.append(f"CREATE TABLE {_quote_col(table_name) if needs_quoting(table_name) else table_name} (")

    col_lines: list[str] = []
    for col in columns:
        fs = summary_lookup.get((table_name, col.column_name))
        long_summary = fs.long_summary if fs else ""
        truncated_summary = _truncate(long_summary, _LONG_SUMMARY_MAX)

        quoted_name = _quote_col(col.column_name)
        pk_clause = " PRIMARY KEY" if col.is_primary_key else ""

        if truncated_summary:
            col_line = f"  {quoted_name} {col.data_type}{pk_clause},  -- {truncated_summary}"
        else:
            col_line = f"  {quoted_name} {col.data_type}{pk_clause},"

        col_lines.append(col_line)

    # Remove trailing comma from last column line (before comment), add closing paren
    if col_lines:
        # Strip the trailing comma only from the actual column definition part,
        # not from the comment. We need to find the comma before the "--" if any.
        last = col_lines[-1]
        # Pattern: "  colname TYPE[PK]," or "  colname TYPE[PK],  -- comment"
        # We want to remove the comma right after the type/PK clause.
        # Replace the first occurrence of ",  --" with "  --" or strip trailing ","
        if ",  --" in last:
            last = last.replace(",  --", "  --", 1)
        elif last.rstrip().endswith(","):
            last = last.rstrip()[:-1]
        col_lines[-1] = last

    lines.extend(col_lines)
    lines.append(");")

    # --- Foreign key comment ---
    table_fks = [
        fk for fk in foreign_keys
        if fk[0] == table_name  # from_table
    ]
    if table_fks:
        fk_parts = [f"{fk[1]} REFERENCES {fk[2]}({fk[3]})" for fk in table_fks]
        lines.append(f"-- Foreign keys: {', '.join(fk_parts)}")

    # --- Example row comment ---
    example_values: list[str] = []
    for col in columns:
        if col.sample_values:
            val = col.sample_values[0][0]  # most frequent value
            val_str = str(val)
            val_str = _truncate(val_str, _SAMPLE_VALUE_MAX_DDL)
            # Quote string-like values
            if col.data_type in ("TEXT", "BLOB"):
                val_str = f"'{val_str}'"
            example_values.append(val_str)
        else:
            example_values.append("NULL")

    lines.append(f"-- Example row: ({', '.join(example_values)})")

    return "\n".join(lines)


def _generate_ddl(profile: DatabaseProfile, summary: DatabaseSummary) -> str:
    """Generate full DDL string for all tables in the database."""
    summary_lookup = _build_summary_lookup(summary)

    # Group columns by table, preserving table order from profile.tables
    table_columns: dict[str, list[ColumnProfile]] = {}
    for table in profile.tables:
        table_columns[table] = []
    for col in profile.columns:
        if col.table_name in table_columns:
            table_columns[col.table_name].append(col)

    blocks: list[str] = []
    for table_name in profile.tables:
        cols = table_columns.get(table_name, [])
        block = _format_ddl_table(
            table_name, cols, summary_lookup, profile.foreign_keys
        )
        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

_MARKDOWN_SAMPLE_MAX = 30   # chars per sample value before truncation
_MARKDOWN_SAMPLES_N = 3     # number of sample values to show


def _format_markdown_table(
    table_name: str,
    columns: list[ColumnProfile],
    summary_lookup: dict[tuple[str, str], FieldSummary],
    fk_cols: set[str],
) -> str:
    """Generate the Markdown block for a single table."""
    lines: list[str] = []

    # --- Header ---
    lines.append(f"## Table: {table_name}")

    # --- Italic description line ---
    # Use short_summary of the first column that has a meaningful (non-empty) summary
    description: Optional[str] = None
    for col in columns:
        fs = summary_lookup.get((table_name, col.column_name))
        if fs and fs.short_summary.strip():
            description = fs.short_summary.strip()
            break
    if not description:
        description = f"Data from {table_name} table."
    lines.append(f"*{description}*")
    lines.append("")

    # --- Pipe table ---
    lines.append("| Column | Type | Description | Sample Values |")
    lines.append("|--------|------|-------------|---------------|")

    for col in columns:
        fs = summary_lookup.get((table_name, col.column_name))

        # Type cell
        type_str = col.data_type
        if col.is_primary_key:
            type_str += " (PK)"
        if col.foreign_key_ref is not None:
            type_str += " (FK)"

        # Description cell
        description_cell = (
            fs.short_summary if (fs and fs.short_summary.strip())
            else col.column_name
        )

        # Sample values cell
        samples = col.sample_values[:_MARKDOWN_SAMPLES_N]
        sample_strs: list[str] = []
        for val, _freq in samples:
            val_str = str(val)
            val_str = _truncate(val_str, _MARKDOWN_SAMPLE_MAX)
            sample_strs.append(val_str)
        sample_cell = ", ".join(sample_strs)

        # Escape pipe characters within cell content to avoid breaking the table
        col_name_cell = col.column_name.replace("|", "\\|")
        description_cell = description_cell.replace("|", "\\|")
        sample_cell = sample_cell.replace("|", "\\|")

        lines.append(
            f"| {col_name_cell} | {type_str} | {description_cell} | {sample_cell} |"
        )

    return "\n".join(lines)


def _generate_markdown(profile: DatabaseProfile, summary: DatabaseSummary) -> str:
    """Generate full Markdown string for all tables in the database."""
    summary_lookup = _build_summary_lookup(summary)

    # Collect FK columns per table
    fk_cols_by_table: dict[str, set[str]] = {}
    for from_table, from_col, _to_table, _to_col in profile.foreign_keys:
        fk_cols_by_table.setdefault(from_table, set()).add(from_col)

    # Group columns by table, preserving table order
    table_columns: dict[str, list[ColumnProfile]] = {}
    for table in profile.tables:
        table_columns[table] = []
    for col in profile.columns:
        if col.table_name in table_columns:
            table_columns[col.table_name].append(col)

    blocks: list[str] = []
    for table_name in profile.tables:
        cols = table_columns.get(table_name, [])
        fk_cols = fk_cols_by_table.get(table_name, set())
        block = _format_markdown_table(table_name, cols, summary_lookup, fk_cols)
        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_schemas(
    profile: DatabaseProfile,
    summary: DatabaseSummary,
) -> FormattedSchemas:
    """
    Format a DatabaseProfile + DatabaseSummary into DDL and Markdown schemas.

    Parameters
    ----------
    profile:
        A DatabaseProfile produced by profile_database().
    summary:
        A DatabaseSummary produced by summarize_database().

    Returns
    -------
    FormattedSchemas with .ddl and .markdown populated.
    """
    ddl = _generate_ddl(profile, summary)
    markdown = _generate_markdown(profile, summary)
    return FormattedSchemas(db_id=profile.db_id, ddl=ddl, markdown=markdown)


def format_and_save_schemas(
    profile: DatabaseProfile,
    summary: DatabaseSummary,
    output_dir: str,
) -> FormattedSchemas:
    """
    Format schemas and save them to disk.

    Saves:
        {output_dir}/{db_id}_ddl.sql
        {output_dir}/{db_id}_markdown.md

    Parameters
    ----------
    profile:
        A DatabaseProfile produced by profile_database().
    summary:
        A DatabaseSummary produced by summarize_database().
    output_dir:
        Directory path where files will be written.  Created if it does not
        already exist.

    Returns
    -------
    FormattedSchemas
    """
    schemas = format_schemas(profile, summary)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ddl_path = out / f"{profile.db_id}_ddl.sql"
    md_path = out / f"{profile.db_id}_markdown.md"

    ddl_path.write_text(schemas.ddl, encoding="utf-8")
    md_path.write_text(schemas.markdown, encoding="utf-8")

    return schemas
