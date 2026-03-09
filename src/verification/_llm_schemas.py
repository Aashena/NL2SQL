"""
LLM tool schemas and system prompts for verification plan generation and
column-alignment inference.

Two tool definitions and their paired system prompts are defined here:

  _PLAN_TOOL / _PLAN_SYSTEM
      Used by _generate_main_plan() to ask the LLM to produce a list of
      semantic verification tests (grain, null, duplicate, ordering, scale,
      boundary, symmetry) for a given NL question.

  _COLUMN_ALIGNMENT_TOOL / _COLUMN_ALIGNMENT_SYSTEM
      Used by _generate_column_alignment_spec() to ask a fast model to
      reason about how many SELECT columns a correct answer must produce.

Both system prompts are marked cacheable to reduce API costs when the same
database schema is used across many questions.
"""
from __future__ import annotations

from src.llm import CacheableText
from src.llm.base import ToolParam

# ---------------------------------------------------------------------------
# Tool schema for plan generation
# ---------------------------------------------------------------------------

_PLAN_TOOL = ToolParam(
    name="verification_plan",
    description=(
        "Generate a list of semantic verification tests for the given SQL question. "
        "Always include grain. Add null, duplicate, ordering, scale, boundary, or symmetry "
        "only when explicitly applicable."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "tests": {
                "type": "array",
                "description": "Verification tests to run on every SQL candidate for this question.",
                "items": {
                    "oneOf": [
                        # ── grain ──────────────────────────────────────────────────────────
                        {
                            "type": "object",
                            "properties": {
                                "test_type": {"type": "string", "const": "grain"},
                                "verification_sql_upper": {
                                    "type": "string",
                                    "description": (
                                        "REQUIRED — a valid SQLite COUNT query against the raw database "
                                        "tables returning an integer upper bound on the number of correct "
                                        "result rows (e.g. SELECT COUNT(DISTINCT id) FROM students). "
                                        "Never omit this; if uncertain, use a conservative COUNT over the "
                                        "main entity table."
                                    ),
                                },
                                "row_count_min": {
                                    "type": "integer",
                                    "description": (
                                        "Minimum expected result rows. Default is 1. "
                                        "Only set higher when you are certain at least N rows must exist."
                                    ),
                                },
                                "row_count_max": {
                                    "type": "integer",
                                    "description": (
                                        "Static integer upper bound when no bounding SQL is available. "
                                        "Prefer verification_sql_upper when possible."
                                    ),
                                },
                                "fix_hint": {
                                    "type": "string",
                                    "description": "Specific, actionable hint for fixing the SQL if this test fails.",
                                },
                            },
                            "required": ["test_type", "verification_sql_upper", "fix_hint"],
                            "additionalProperties": False,
                        },
                        # ── null ───────────────────────────────────────────────────────────
                        {
                            "type": "object",
                            "properties": {
                                "test_type": {"type": "string", "const": "null"},
                                "check_columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Column names to check for unexpected NULLs.",
                                },
                                "fix_hint": {
                                    "type": "string",
                                    "description": "Specific, actionable hint for fixing the SQL if this test fails.",
                                },
                            },
                            "required": ["test_type", "check_columns", "fix_hint"],
                            "additionalProperties": False,
                        },
                        # ── duplicate ──────────────────────────────────────────────────────
                        {
                            "type": "object",
                            "properties": {
                                "test_type": {"type": "string", "const": "duplicate"},
                                "verification_sql": {
                                    "type": "string",
                                    "description": (
                                        "A direct SQLite COUNT(DISTINCT pk) query against the raw database "
                                        "returning the expected distinct entity count "
                                        "(e.g. SELECT COUNT(DISTINCT student_id) FROM students). "
                                        "Used to detect JOIN-induced row multiplication."
                                    ),
                                },
                                "fix_hint": {
                                    "type": "string",
                                    "description": "Specific, actionable hint for fixing the SQL if this test fails.",
                                },
                            },
                            "required": ["test_type", "verification_sql", "fix_hint"],
                            "additionalProperties": False,
                        },
                        # ── ordering ───────────────────────────────────────────────────────
                        {
                            "type": "object",
                            "properties": {
                                "test_type": {"type": "string", "const": "ordering"},
                                "required_sql_keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "SQL keywords that must appear in the candidate SQL (e.g. ['ORDER BY', 'LIMIT']).",
                                },
                                "order_by_column": {
                                    "type": "string",
                                    "description": (
                                        "The column or expression to sort by "
                                        "(e.g. 'score', 'total_revenue')."
                                    ),
                                },
                                "order_by_direction": {
                                    "type": "string",
                                    "enum": ["ASC", "DESC"],
                                    "description": (
                                        "'DESC' for highest/best/largest/most/top; "
                                        "'ASC' for lowest/worst/smallest/least/bottom."
                                    ),
                                },
                                "order_limit": {
                                    "type": "integer",
                                    "description": (
                                        "The integer N from 'top N' / 'N highest' etc. "
                                        "Extract from the question text (e.g. 'top 5 students' → 5). "
                                        "Omit if no explicit N."
                                    ),
                                },
                                "fix_hint": {
                                    "type": "string",
                                    "description": "Specific, actionable hint for fixing the SQL if this test fails.",
                                },
                            },
                            "required": [
                                "test_type", "required_sql_keywords",
                                "order_by_column", "order_by_direction", "fix_hint",
                            ],
                            "additionalProperties": False,
                        },
                        # ── scale ──────────────────────────────────────────────────────────
                        {
                            "type": "object",
                            "properties": {
                                "test_type": {"type": "string", "const": "scale"},
                                "numeric_min": {
                                    "type": "number",
                                    "description": "Minimum expected numeric value in the result.",
                                },
                                "numeric_max": {
                                    "type": "number",
                                    "description": "Maximum expected numeric value in the result.",
                                },
                                "fix_hint": {
                                    "type": "string",
                                    "description": "Specific, actionable hint for fixing the SQL if this test fails.",
                                },
                            },
                            "required": ["test_type", "numeric_min", "numeric_max", "fix_hint"],
                            "additionalProperties": False,
                        },
                        # ── boundary ───────────────────────────────────────────────────────
                        {
                            "type": "object",
                            "properties": {
                                "test_type": {"type": "string", "const": "boundary"},
                                "description": {"type": "string"},
                                "expected_outcome": {"type": "string"},
                                "fix_hint": {
                                    "type": "string",
                                    "description": "Specific, actionable hint for fixing the SQL if this test fails.",
                                },
                            },
                            "required": ["test_type", "description", "expected_outcome", "fix_hint"],
                            "additionalProperties": False,
                        },
                        # ── symmetry ───────────────────────────────────────────────────────
                        {
                            "type": "object",
                            "properties": {
                                "test_type": {"type": "string", "const": "symmetry"},
                                "verification_sql": {
                                    "type": "string",
                                    "description": (
                                        "A direct SQLite query returning the expected sub-group total "
                                        "(e.g. SELECT SUM(amount) FROM payments WHERE status='active'). "
                                        "Used to verify that the candidate SQL total equals the sum of parts."
                                    ),
                                },
                                "fix_hint": {
                                    "type": "string",
                                    "description": "Specific, actionable hint for fixing the SQL if this test fails.",
                                },
                            },
                            "required": ["test_type", "verification_sql", "fix_hint"],
                            "additionalProperties": False,
                        },
                    ]
                },
            }
        },
        "required": ["tests"],
    },
)

_COLUMN_ALIGNMENT_TOOL = ToolParam(
    name="column_alignment_spec",
    description=(
        "Specify the exact number of SELECT columns a correct SQL answer must produce "
        "for the given natural-language question."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Step-by-step analysis: identify what the question is asking to output, "
                    "then count only the columns that belong in the final SELECT list."
                ),
            },
            "expected_column_count": {
                "type": "integer",
                "description": "Exact count of SELECT columns a correct answer must produce (must be >= 1).",
            },
            "column_descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Ordered list of short semantic labels for each SELECT column "
                    "(one entry per column, matching expected_column_count). "
                    "Examples: 'student name', 'average GPA', 'department count'."
                ),
            },
        },
        "required": ["reasoning", "expected_column_count", "column_descriptions"],
    },
)

_PLAN_SYSTEM = CacheableText(
    text=(
        "You are an expert SQL quality analyst. Given a natural-language question "
        "and a database schema, generate a concise set of semantic verification "
        "tests that any correct SQL answer to this question should pass.\n\n"
        "Test types and when to use each:\n\n"
        "grain: ALWAYS include — this test is MANDATORY for every question.\n"
        "  verification_sql_upper is REQUIRED for every grain test. You MUST provide a valid "
        "SQLite COUNT query against the raw database tables that returns an integer upper bound "
        "on the number of correct result rows "
        "(e.g. SELECT COUNT(DISTINCT student_id) FROM students). "
        "Never omit verification_sql_upper. If you are uncertain which entity to count, "
        "write a conservative COUNT over the primary entity table. "
        "Setting it to an empty string is NOT acceptable.\n"
        "  - Set row_count_min only if you are certain at least N>1 rows must exist; "
        "otherwise leave it unset (default is 1 — empty results are always wrong).\n\n"
        "null: If key output columns should not be NULL (e.g. names, IDs), list "
        "those column names in check_columns.\n\n"
        "duplicate: If the query involves JOINs across multiple tables, generate a "
        "verification_sql counting DISTINCT primary key entities to detect row "
        "multiplication.\n\n"
        "ordering: If the question uses 'top N', 'highest', 'lowest', 'ranked', "
        "'best', or 'worst', set required_sql_keywords to ['ORDER BY', 'LIMIT']. "
        "Also set:\n"
        "  - order_by_column: the ranking metric column from the question+schema "
        "(e.g. 'gpa', 'total_sales'). Pick the most likely numeric ranking column.\n"
        "  - order_by_direction: 'DESC' for highest/best/largest/most/top, "
        "'ASC' for lowest/worst/smallest/least/bottom.\n"
        "  - order_limit: integer N from the question ('top 5' → 5, 'three highest' → 3). "
        "Null if no explicit N.\n"
        "  - fix_hint: a concrete SQL clause, e.g. "
        "'Add ORDER BY score DESC LIMIT 5 before the semicolon.'\n\n"
        "scale: If numeric values have known bounds (percentages: 0-100, ages: "
        "0-120, ratings: 0-5), set numeric_min and numeric_max.\n\n"
        "boundary: Only include if the question mentions specific dates, years, or "
        "time periods. Provide description (the time constraint from the question) and "
        "expected_outcome (what correct SQL date handling looks like).\n\n"
        "symmetry: Only include if the question asks for a total that should equal "
        "the sum of sub-groups (e.g. 'total = active + inactive'). Generate a "
        "verification_sql for the sub-total check.\n\n"
        "RULES:\n"
        "- ALWAYS include grain — it applies to every NL2SQL question.\n"
        "- Additionally include null, duplicate, ordering, scale, boundary, or symmetry "
        "ONLY when explicitly applicable.\n"
        "- 1-5 tests total. grain is mandatory.\n"
        "- verification_sql must be valid SQLite querying the raw database tables.\n"
        "- fix_hint must be concrete and mention specific SQL clauses to add/change.\n"
    ),
    cache=True,
)

_COLUMN_ALIGNMENT_SYSTEM = CacheableText(
    text=(
        "You are an expert at reading natural-language database questions and determining "
        "exactly how many columns a correct SQL answer must produce in its SELECT clause.\n\n"
        "RULES:\n"
        "- Count ONLY the columns the question explicitly asks to OUTPUT (the final SELECT list).\n"
        "- Do NOT count helper columns used for ORDER BY, WHERE, JOINs, or subqueries.\n"
        "- Pattern guide:\n"
        "  'How many …?' / 'What is the total/average/min/max …?' → 1 (scalar aggregate)\n"
        "  'What is the X of Y?' → 1 (only X is output)\n"
        "  'What is the name of the X with the highest Y?' → 1 (name only; Y is for ordering)\n"
        "  'Is there any …?' / 'Does … exist?' → 1 (boolean/count)\n"
        "  'List the X and Y of each Z' → 2\n"
        "  'Return X, Y, and Z for each W' → 3\n"
        "  'For each A, return the A name and the number of Bs' → 2\n"
        "- When the question says only 'name' or only 'id', count is 1 even if JOINs are needed.\n"
        "- Tie-breaking columns (e.g. score used only to pick the winner) do NOT count.\n\n"
        "EXAMPLES:\n\n"
        "Q: 'How many students are enrolled in the club?'\n"
        "Reasoning: The question asks for a count → single aggregate column.\n"
        "expected_column_count: 1\n"
        'column_descriptions: ["enrollment count"]\n\n'
        "Q: 'What is the average GPA of students in the CS department?'\n"
        "Reasoning: AVG aggregate → one output value.\n"
        "expected_column_count: 1\n"
        'column_descriptions: ["average GPA"]\n\n'
        "Q: 'What is the name of the school with the highest average SAT score?'\n"
        "Reasoning: Output is 'name'; SAT score is used only for ranking, not output.\n"
        "expected_column_count: 1\n"
        'column_descriptions: ["school name"]\n\n'
        "Q: 'List the name and GPA of each student who made the honor roll.'\n"
        "Reasoning: Two things explicitly requested: name AND GPA.\n"
        "expected_column_count: 2\n"
        'column_descriptions: ["student name", "GPA"]\n\n'
        "Q: 'Return the employee ID, department name, and salary for all managers.'\n"
        "Reasoning: Three columns explicitly listed: employee_id, department_name, salary.\n"
        "expected_column_count: 3\n"
        'column_descriptions: ["employee ID", "department name", "salary"]\n\n'
        "Q: 'What percentage of games were won by the home team?'\n"
        "Reasoning: Single computed ratio → one output column.\n"
        "expected_column_count: 1\n"
        'column_descriptions: ["win percentage"]\n\n'
        "Q: 'For each department, list the department name and the number of employees.'\n"
        "Reasoning: Per-group output with two values: name + count.\n"
        "expected_column_count: 2\n"
        'column_descriptions: ["department name", "employee count"]\n\n'
        "Q: 'Which player scored the most goals? Return the player name.'\n"
        "Reasoning: Explicitly asks for name only; goals used for ordering.\n"
        "expected_column_count: 1\n"
        'column_descriptions: ["player name"]\n\n'
        "Q: 'What is the name and age of the oldest active user?'\n"
        "Reasoning: Two attributes requested: name AND age.\n"
        "expected_column_count: 2\n"
        'column_descriptions: ["user name", "age"]\n\n'
        "Q: 'Is there any student with a GPA above 3.9?'\n"
        "Reasoning: Yes/no question → single boolean or count output.\n"
        "expected_column_count: 1\n"
        'column_descriptions: ["existence flag"]\n\n'
        "Now analyze the given question and evidence, then call column_alignment_spec "
        "with your step-by-step reasoning, the final expected_column_count, and a "
        "column_descriptions list with one short semantic label per column."
    ),
    cache=True,
)
