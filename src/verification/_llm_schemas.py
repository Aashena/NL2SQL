"""
LLM tool schemas and system prompts for verification plan generation and
column-alignment inference.

Three tool definitions and their paired system prompts are defined here:

  _GRAIN_TOOL / _GRAIN_SYSTEM
      Used by _generate_grain_spec() to ask the LLM to produce the mandatory
      grain verification test (row-count upper bound) for a given NL question.
      verification_sql_upper is schema-level required.

  _OPTIONAL_TESTS_TOOL / _OPTIONAL_TESTS_SYSTEM
      Used by _generate_optional_tests() to ask the LLM for the applicable
      optional tests (null, duplicate, ordering, scale, boundary, symmetry).
      Grain is excluded from this call entirely.

  _COLUMN_ALIGNMENT_TOOL / _COLUMN_ALIGNMENT_SYSTEM
      Used by _generate_column_alignment_spec() to ask a fast model to
      reason about how many SELECT columns a correct answer must produce.

All system prompts are marked cacheable to reduce API costs when the same
database schema is used across many questions.

Backward-compat aliases at the bottom of this file map the old names
(_PLAN_TOOL, _PLAN_SYSTEM) to the new optional-tests equivalents so that
existing test patches continue to work without modification.
"""
from __future__ import annotations

from src.llm import CacheableText
from src.llm.base import ToolParam

# ---------------------------------------------------------------------------
# Grain tool — dedicated single-test schema
# ---------------------------------------------------------------------------

_GRAIN_TOOL = ToolParam(
    name="grain_verification",
    description=(
        "Generate the grain verification test for the given SQL question. "
        "The grain test checks that the result row count does not exceed the number "
        "of real entities the question is asking about."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "verification_sql_upper": {
                "type": "string",
                "description": (
                    "A valid SQLite SELECT COUNT(...) query against the raw database tables "
                    "returning an integer upper bound on the number of correct result rows. "
                    "Must be UNCONDITIONAL — do NOT add WHERE, HAVING, or JOIN conditions from the question. "
                    "EXAMPLE: SELECT COUNT(DISTINCT student_id) FROM students\n"
                    "For scalar aggregate questions use: SELECT 1\n"
                    "Omit or leave empty ONLY when upper_bound_confidence is 'none'."
                ),
            },
            "upper_bound_confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "none"],
                "description": (
                    "Your confidence that verification_sql_upper correctly bounds the result rows. "
                    "'high'   — you know exactly which entity to count. "
                    "'medium' — you have a reasonable COUNT query over the main entity. "
                    "'low'    — you are guessing; still provide a conservative COUNT query. "
                    "'none'   — use ONLY when you truly cannot identify ANY table to count; "
                    "grain test will be SKIPPED and verification_sql_upper may be omitted. "
                    "Decision rule: ask 'Can I name ANY table to count rows from, even loosely?' "
                    "If YES → use 'low', never 'none'. 'none' should be extremely rare."
                ),
            },
            "row_count_min": {
                "type": "integer",
                "description": (
                    "Minimum expected result rows. Default is 1. "
                    "Set to 0 when the question may legitimately return empty results "
                    "(e.g. 'List students with GPA above 4.5' — there may be none). "
                    "Set > 1 when the question explicitly demands at least N rows "
                    "(e.g. 'List the top 3 students by GPA' → row_count_min = 3)."
                ),
            },
        },
        "required": ["verification_sql_upper", "upper_bound_confidence"],
    },
)

# NOTE: cache=True is set on all three system prompts below.
# For the Anthropic client this enables prompt caching via cache_control:ephemeral.
# For the Gemini client, Gemini's API rejects requests that combine cached_content
# with tools/tool_config in the same call, so GeminiClient sets can_use_cache=False
# whenever tool_defs is non-empty.  All three verification calls use tools, so
# caching is currently a no-op for Gemini.  The markers are kept so that caching
# works automatically if Gemini ever lifts this restriction.

_GRAIN_SYSTEM = CacheableText(
    text=(
        "You are an expert SQL quality analyst. Given a natural-language question "
        "and a database schema, generate the grain verification test that any correct "
        "SQL answer to this question should pass.\n\n"
        "The grain test checks that the result row count does not exceed the number of "
        "real entities the question is asking about.\n\n"
        "━━━ HOW TO WRITE verification_sql_upper ━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Decision rule — ask: 'Can I name ANY table to count rows from, even loosely?'\n\n"
        "  → YES (almost always the case):\n"
        "    • Write a real SQLite SELECT COUNT(...) query using EXACT table names from the schema.\n"
        "    • Set upper_bound_confidence to 'high', 'medium', or 'low'.\n"
        "    • Even a rough SELECT COUNT(*) FROM <most_likely_table> is fine with confidence 'low'.\n\n"
        "  → NO (extremely rare — cannot identify any table at all):\n"
        "    • Set upper_bound_confidence to 'none' and omit verification_sql_upper.\n"
        "    • The grain test will be SKIPPED. This should almost never happen.\n\n"
        "upper_bound_confidence guide:\n"
        "  'high'   — you know exactly which entity to count.\n"
        "  'medium' — you have a reasonable COUNT over the main entity table.\n"
        "  'low'    — you are guessing; provide a conservative COUNT query anyway.\n"
        "  'none'   — truly cannot identify any table; verification_sql_upper may be omitted.\n\n"
        "HIGH CONFIDENCE EXAMPLE (question asks about patients):\n"
        "  verification_sql_upper: SELECT COUNT(DISTINCT patient_id) FROM patients\n"
        "  upper_bound_confidence: high\n\n"
        "LOW CONFIDENCE EXAMPLE (main table is uncertain):\n"
        "  verification_sql_upper: SELECT COUNT(*) FROM orders\n"
        "  upper_bound_confidence: low\n\n"
        "Choosing between COUNT(DISTINCT pk) and COUNT(*):\n"
        "  • Use COUNT(DISTINCT <primary_key>) when you are counting distinct named\n"
        "    entities (patients, students, products, employees). This is ALWAYS safer\n"
        "    than COUNT(*) when the table may have duplicate rows or when a JOIN is involved.\n"
        "  • Use COUNT(*) only when the table rows ARE the entities being counted\n"
        "    (e.g. a bridge/fact table with no natural primary key to DISTINCT on,\n"
        "    or as a quick low-confidence bound).\n"
        "  Rule of thumb: if you can name the primary key column, use\n"
        "  COUNT(DISTINCT <pk>). Prefer COUNT(*) only as a last resort.\n\n"
        "━━━ ALWAYS WRITE UNCONDITIONAL COUNT QUERIES ━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "verification_sql_upper is an UPPER BOUND, NOT an exact match.\n"
        "NEVER add WHERE, HAVING, or JOIN conditions from the question to it.\n\n"
        "WRONG (adds question's filter → too tight, may return 0):\n"
        "  Question: 'List patients with blood type A admitted after 2020.'\n"
        "  verification_sql_upper: SELECT COUNT(DISTINCT id) FROM patients\n"
        "                          WHERE blood_type = 'A' AND year > 2020  ← WRONG\n\n"
        "RIGHT (unconditional count of the entity type — always a safe ceiling):\n"
        "  verification_sql_upper: SELECT COUNT(DISTINCT id) FROM patients  ← RIGHT\n\n"
        "The fixer already sees the exact row diff (actual vs. bound). The bound just\n"
        "needs to be a safe ceiling — an unconditional entity count is always safe.\n\n"
        "━━━ SCALAR VS. GROUPED AGGREGATE QUESTIONS ━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "SCALAR aggregate — the question asks for a single aggregate value with NO grouping.\n"
        "The correct SQL returns exactly ONE row.\n"
        "Signals: 'how many total', 'what is the overall average/sum/min/max', 'the total of',\n"
        "  'in total', 'overall count', 'across all'.\n"
        "  → verification_sql_upper = 'SELECT 1'\n"
        "  → row_count_min = 1  (exactly one row expected)\n\n"
        "GROUPED aggregate — the question groups results by an entity, producing one row per group.\n"
        "The correct SQL returns N rows (one per group entity).\n"
        "Signals: 'per X', 'by X', 'for each X', 'broken down by', 'for every X'.\n"
        "  → Treat as a normal entity-list question.\n"
        "  → COUNT the grouping entity: SELECT COUNT(DISTINCT <group_pk>) FROM <group_table>\n\n"
        "SCALAR EXAMPLE:\n"
        "  Question: 'What is the average salary of all employees?'\n"
        "  No grouping → one aggregate row.\n"
        "  verification_sql_upper: SELECT 1\n"
        "  row_count_min: 1\n\n"
        "GROUPED EXAMPLE:\n"
        "  Question: 'What is the average salary per department?'\n"
        "  'per department' → one row per department.\n"
        "  verification_sql_upper: SELECT COUNT(DISTINCT department_id) FROM departments\n"
        "  (Do NOT use 'SELECT 1' here — the result has multiple rows.)\n\n"
        "━━━ MULTI-TABLE / JOIN QUESTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "When the question references one entity type but filtering or ranking\n"
        "requires joining to a related table, always COUNT the ASKED-ABOUT entity,\n"
        "not the joined table.\n\n"
        "WRONG (counts visits, not patients):\n"
        "  Question: 'List all patients who visited more than 3 doctors.'\n"
        "  verification_sql_upper: SELECT COUNT(*) FROM visits  ← WRONG\n\n"
        "RIGHT (counts the entity the question asks about):\n"
        "  verification_sql_upper: SELECT COUNT(DISTINCT patient_id) FROM patients\n\n"
        "General rule:\n"
        "  • Identify the SUBJECT entity of the question (what the result rows\n"
        "    represent — e.g. patients, orders, employees).\n"
        "  • COUNT(DISTINCT <subject_pk>) FROM <subject_table> — NO conditions.\n"
        "  • Never count rows from a fact/bridge table as a proxy for distinct\n"
        "    entities in the subject table.\n\n"
        "━━━ COMPLEX / MULTI-STEP QUESTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "For questions with complex filtering across multiple joins, still write the\n"
        "simplest unconditional COUNT of the main entity. The bound will be loose\n"
        "(set confidence 'low') but that is always safe.\n\n"
        "WRONG (adds conditions → too tight, may return 0):\n"
        "  Question: 'Among customers who placed at least 2 orders, how many\n"
        "             also purchased product X?'\n"
        "  verification_sql_upper: SELECT COUNT(DISTINCT c.id) FROM customers c\n"
        "                          JOIN orders o ON c.id = o.customer_id\n"
        "                          WHERE o.product_id = 'X'  ← WRONG\n\n"
        "RIGHT (simple unconditional upper bound):\n"
        "  verification_sql_upper: SELECT COUNT(DISTINCT id) FROM customers  ← RIGHT\n"
        "  upper_bound_confidence: low\n"
        "  (The grain test just ensures the result does not exceed the total customer count.)\n\n"
        "━━━ SETTING row_count_min ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "row_count_min controls how many rows the result must have at minimum.\n\n"
        "  Default (omit the field): effective minimum is 1.\n\n"
        "  Set row_count_min = 0 when the question may legitimately return NO rows:\n"
        "    • Question has a filter condition that may match nothing:\n"
        "      'List students with GPA above 4.5' — there may be none.\n"
        "      'Find orders placed after 2099-01-01' — likely none.\n\n"
        "  Set row_count_min = N (N > 1) when the question explicitly demands at least N rows:\n"
        "    • Explicit count: 'List the top 3 students by GPA' → row_count_min = 3\n"
        "      verification_sql_upper: SELECT COUNT(DISTINCT student_id) FROM students\n"
        "      (grain test enforces: 3 ≤ actual_count ≤ total_students)\n"
        "    • 'Show the 5 most recent orders' → row_count_min = 5\n"
        "    • 'Return the bottom 2 performers' → row_count_min = 2\n\n"
        "  Keep row_count_min = 1 (default) for all other cases.\n\n"
        "━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "- Use ONLY table and column names that appear in the Database Schema provided.\n"
        "- Do NOT invent or guess table names — use the exact names from the schema.\n"
        "- verification_sql_upper must be a valid SQLite SELECT query returning a single integer.\n"
        "- NEVER add WHERE, HAVING, or JOIN conditions from the question to verification_sql_upper.\n"
    ),
    cache=True,
)

# ---------------------------------------------------------------------------
# Optional tests tool — null / duplicate / ordering / scale / boundary / symmetry
# ---------------------------------------------------------------------------

_OPTIONAL_TESTS_TOOL = ToolParam(
    name="verification_plan",
    description=(
        "For each applicable optional test type (null, duplicate, ordering, scale, boundary, "
        "symmetry), include the corresponding key with its required sub-fields. "
        "Omit keys for tests that do not apply. Do NOT include a grain key — that is handled separately."
    ),
    input_schema={
        "type": "object",
        "properties": {
            # ── null ───────────────────────────────────────────────────────────
            "null": {
                "type": "object",
                "properties": {
                    "check_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "REQUIRED — list of output column names that must be non-NULL. "
                            "Omit this entire test key if you cannot identify at least one specific column."
                        ),
                    },
                    "fix_hint": {
                        "type": "string",
                        "description": "REQUIRED — concrete hint naming which columns to check and why they may be NULL.",
                    },
                },
                # no nested required — Gemini handles flat schemas more reliably;
                # field descriptions with REQUIRED guide the model instead.
            },
            # ── duplicate ──────────────────────────────────────────────────────
            "duplicate": {
                "type": "object",
                "properties": {
                    "verification_sql": {
                        "type": "string",
                        "description": (
                            "REQUIRED — a valid SQLite SELECT COUNT(DISTINCT ...) query that returns "
                            "the expected number of unique entities. "
                            "Omit this entire test key if you cannot write this query."
                        ),
                    },
                    "fix_hint": {"type": "string"},
                },
            },
            # ── ordering ───────────────────────────────────────────────────────
            "ordering": {
                "type": "object",
                "properties": {
                    "required_sql_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ALWAYS include this field with at minimum [\"ORDER BY\"]. "
                            "Include \"LIMIT\" whenever the question implies a fixed number of rows: "
                            "(a) explicit counts: 'top 5', '3 highest', 'bottom 2' → include \"LIMIT\"; "
                            "(b) superlatives returning a single entity: 'which hospital has the "
                            "lowest...', 'who has the most...' → also include \"LIMIT\" (answer is 1). "
                            "Only omit \"LIMIT\" when the question asks for ALL matching entities "
                            "without a count or superlative (e.g., 'list all students with GPA > 3.5')."
                        ),
                    },
                    "order_by_column": {
                        "type": "string",
                        "description": (
                            "REQUIRED — the column or expression to sort by. "
                            "Omit this entire test key if you cannot identify the sort column."
                        ),
                    },
                    "order_by_direction": {"type": "string", "enum": ["ASC", "DESC"]},
                    "order_limit": {
                        "type": "integer",
                        "description": (
                            "The expected LIMIT value. "
                            "Set to N for explicit counts like 'top N' / 'N highest'. "
                            "Set to 1 for superlatives like 'which X has the highest/lowest Y' "
                            "(the question implicitly asks for exactly one answer). "
                            "Always set order_limit together with 'LIMIT' in required_sql_keywords."
                        ),
                    },
                    "fix_hint": {"type": "string"},
                },
            },
            # ── scale ──────────────────────────────────────────────────────────
            "scale": {
                "type": "object",
                "properties": {
                    "numeric_min": {
                        "type": "number",
                        "description": (
                            "REQUIRED — tight lower bound. Only include this test when you can "
                            "specify a meaningful tight range (e.g., 0 for percentages, 1 for ratings). "
                            "Omit this test if the range would be very wide or unknown."
                        ),
                    },
                    "numeric_max": {
                        "type": "number",
                        "description": (
                            "REQUIRED — tight upper bound (e.g., 100 for percentages, 5 for 1–5 ratings). "
                            "Omit this test if you cannot specify a tight, meaningful maximum."
                        ),
                    },
                    "fix_hint": {"type": "string"},
                },
            },
            # ── boundary ───────────────────────────────────────────────────────
            "boundary": {
                "type": "object",
                "properties": {
                    "boundary_description": {
                        "type": "string",
                        "description": (
                            "REQUIRED — describe the specific time window from the question "
                            "(e.g., 'Q1 2022 (January–March 2022)'). "
                            "Omit this entire test key if you cannot specify the time window clearly."
                        ),
                    },
                    "expected_outcome": {
                        "type": "string",
                        "description": (
                            "REQUIRED — describe what a correct SQL date filter should look like "
                            "(e.g., 'WHERE clause filters date BETWEEN 2022-01-01 AND 2022-03-31'). "
                            "Be specific about the exact period boundaries."
                        ),
                    },
                    "fix_hint": {"type": "string"},
                },
            },
            # ── symmetry ───────────────────────────────────────────────────────
            "symmetry": {
                "type": "object",
                "properties": {
                    "verification_sql": {
                        "type": "string",
                        "description": (
                            "REQUIRED — a valid SQLite query computing the grand total that "
                            "sub-group sums must equal (e.g., SELECT SUM(amount) FROM loans). "
                            "Omit this entire test key if you cannot write this query."
                        ),
                    },
                    "fix_hint": {"type": "string"},
                },
            },
        },
        # No top-level required — all test types are optional; omit inapplicable ones
    },
)

_OPTIONAL_TESTS_SYSTEM = CacheableText(
    text=(
        "You are an expert SQL quality analyst. Given a natural-language question "
        "and a database schema, generate the applicable optional semantic verification tests "
        "that any correct SQL answer to this question should pass.\n\n"
        "Call verification_plan with one key per applicable test type "
        "(null, duplicate, ordering, scale, boundary, symmetry). "
        "Omit keys for tests that do not apply. Do NOT include a grain key.\n\n"
        "You will include each optional test (null, duplicate, ordering, scale, boundary, symmetry) "
        "whenever the question or schema shows any of the listed signals — a single matching "
        "signal is sufficient. Do NOT include a grain test — that is handled separately.\n\n"
        "━━━ RESPONSE FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Call verification_plan with one top-level key per applicable test type.\n"
        "The KEY is the test type name; the VALUE is an object with the test sub-fields.\n"
        "Omit keys for test types that do not apply.\n\n"
        "Single-test example (only 'null' applies):\n"
        "  Include key \"null\" with:\n"
        "    check_columns: [\"name\", \"email\"]\n"
        "    fix_hint: \"Ensure SELECT returns non-NULL name and email.\"\n\n"
        "Multi-test example (both 'null' and 'ordering' apply):\n"
        "  Include key \"null\" with:\n"
        "    check_columns: [\"name\", \"email\"]\n"
        "    fix_hint: \"Ensure non-NULL name and email.\"\n"
        "  Include key \"ordering\" with:\n"
        "    required_sql_keywords: [\"ORDER BY\", \"LIMIT\"]\n"
        "    order_by_column: \"total_sales\"\n"
        "    order_by_direction: \"DESC\"\n"
        "    order_limit: 3\n"
        "    fix_hint: \"Add ORDER BY total_sales DESC LIMIT 3.\"\n\n"
        "No-test example (no optional tests apply for this question):\n"
        "  Call verification_plan with an empty object — no keys at all.\n\n"
        "━━━ OPTIONAL TESTS — include whenever ANY listed signal is present ━━\n\n"
        "null: Include when the question asks for attributes that may be sparsely\n"
        "  populated (contact info, descriptions, bios, website URLs, secondary fields)\n"
        "  or when the schema has nullable columns for the output fields.\n"
        "  Signals:\n"
        "    • Question requests names, emails, addresses, phone numbers, or descriptions\n"
        "      of entities that may not always have that field filled.\n"
        "    • Schema shows a nullable column for one of the requested output attributes.\n"
        "    • Query likely uses a LEFT JOIN whose right-side columns appear in SELECT.\n"
        "  Example:\n"
        "    Question: 'List the full name and email address of every librarian.'\n"
        "    Schema: librarians(id, name, email) — email is nullable\n"
        "    → include 'null' key:\n"
        "      null={check_columns: ['name', 'email'],\n"
        "            fix_hint: 'Ensure SELECT returns non-NULL name and email. Check JOIN and WHERE conditions.'}\n\n"
        "duplicate: Include when the question lists entities (not aggregates) AND the\n"
        "  query joins to a table where one entity can have many rows.\n"
        "  Signals:\n"
        "    • Question asks to 'list', 'show', 'find', or 'return' entities\n"
        "      (customers, patients, products, employees) while joining or filtering\n"
        "      on a related table (orders, visits, purchases, enrollments).\n"
        "    • The joining table is in a one-to-many relationship with the listed entity.\n"
        "    • The question does NOT ask for a COUNT, SUM, AVG, or other aggregate\n"
        "      (aggregates collapse rows naturally; lists do not).\n"
        "  Example:\n"
        "    Question: 'List the names of all customers who have placed at least one order.'\n"
        "    Schema: customers(id, name), orders(id, customer_id, amount)\n"
        "    One customer → many orders → JOIN can produce duplicate customer rows.\n"
        "    → include 'duplicate' key:\n"
        "      duplicate={verification_sql: 'SELECT COUNT(DISTINCT id) FROM customers',\n"
        "                 fix_hint: 'Add DISTINCT to SELECT or use EXISTS/IN subquery to avoid duplicate customer rows.'}\n\n"
        "ordering: Include when the question uses any ranking or superlative language.\n"
        "  Signals — include ordering if ANY of these appear:\n"
        "    • 'top N', 'bottom N', 'first N', 'last N' (N is any integer or word)\n"
        "    • 'highest', 'lowest', 'largest', 'smallest', 'greatest', 'least'\n"
        "    • 'best', 'worst', 'most', 'fewest', 'maximum', 'minimum'\n"
        "    • 'ranked', 'rank', 'leading', 'trailing'\n"
        "    • 'who has the most/least …', 'which … has the highest/lowest …'\n"
        "  Set order_by_direction to 'DESC' for highest/best/most/top, 'ASC' for lowest/worst/fewest/bottom.\n"
        "  Set order_limit and include 'LIMIT' in required_sql_keywords:\n"
        "    • Explicit count: 'top 3' → order_limit: 3\n"
        "    • Superlative (single entity): 'which hospital has the lowest...' → order_limit: 1\n"
        "    • No count, list all: 'list students ranked by GPA' → omit order_limit, no LIMIT needed\n"
        "  Examples:\n"
        "    Question: 'What are the top 3 best-selling products this year?'\n"
        "    → include 'ordering' key:\n"
        "      required_sql_keywords: [\"ORDER BY\", \"LIMIT\"]\n"
        "      order_by_column: \"total_sales\"\n"
        "      order_by_direction: \"DESC\"\n"
        "      order_limit: 3\n"
        "      fix_hint: \"Add ORDER BY total_sales DESC LIMIT 3 before the semicolon.\"\n\n"
        "    Question: 'Which hospital has the lowest average wait time?'\n"
        "    Superlative → exactly one hospital expected → require LIMIT 1.\n"
        "    → include 'ordering' key:\n"
        "      required_sql_keywords: [\"ORDER BY\", \"LIMIT\"]\n"
        "      order_by_column: \"avg_wait_time\"\n"
        "      order_by_direction: \"ASC\"\n"
        "      order_limit: 1\n"
        "      fix_hint: \"Add ORDER BY avg_wait_time ASC LIMIT 1 — the question asks for exactly one hospital.\"\n\n"
        "scale: Include ONLY when you can specify a TIGHT, MEANINGFUL numeric range for the result.\n"
        "  Signals:\n"
        "    • Result is a percentage or ratio with a known tight range: 0–100 or 0.0–1.0.\n"
        "    • Result is a rating or score on a well-defined fixed scale (e.g. 1–5 stars, 0–10 points).\n"
        "    • Result is an age (0–120), probability (0–1), or other universally bounded numeric.\n"
        "  Do NOT include if:\n"
        "    • The range would be very wide or unknown (e.g., 0–1,000,000 for counts or totals).\n"
        "    • You are uncertain what numeric range applies.\n"
        "    • The question asks for counts, sums, or averages without a fixed domain bound.\n"
        "  Set numeric_min and numeric_max to the tightest known domain constraint.\n"
        "  Examples:\n"
        "    Question: 'What is the average review rating for each restaurant in the city?'\n"
        "    Ratings are on a 1–5 scale → output must be between 1 and 5.\n"
        "    → include 'scale' key:\n"
        "      scale={numeric_min: 1, numeric_max: 5,\n"
        "             fix_hint: 'Verify AVG(rating) is between 1 and 5. Check the correct rating column is used.'}\n\n"
        "    Question: 'What percentage of flights were delayed in 2022?'\n"
        "    A percentage must be between 0 and 100.\n"
        "    → include 'scale' key:\n"
        "      scale={numeric_min: 0, numeric_max: 100,\n"
        "             fix_hint: 'The result must be in [0, 100]. Use COUNT(*FILTER*) / COUNT(*) * 100.0.'}\n\n"
        "boundary: Include ONLY when the question defines a specific, bounded time window\n"
        "  where getting the exact boundary wrong would change the answer.\n"
        "  Signals (ALL of these must hold, not just one):\n"
        "    • The question references a specific, delimited period: 'in Q1 2022',\n"
        "      'during 2020–2022', 'between March and June', 'in the first half of 2021'\n"
        "    • The boundary is non-trivial: a wrong date range (wrong quarter, wrong year span)\n"
        "      would meaningfully change which rows are returned.\n"
        "  Do NOT include for:\n"
        "    • Simple year-equality filters: 'hired in 2015', 'year = 2020'\n"
        "    • Open-ended conditions: 'after 2015', 'before 2020', 'since 2019'\n"
        "    • Questions where the date is just a filter field, not a precise time window\n"
        "  Provide boundary_description (the exact time window) and expected_outcome\n"
        "  (what a correct SQL date filter should look like, with specific date values).\n"
        "  Example (multi-year range — boundary matters):\n"
        "    Question: 'How many prescriptions were issued in the first quarter of 2022?'\n"
        "    → include 'boundary' key:\n"
        "      boundary={boundary_description: 'Q1 2022 (January 1 – March 31, 2022)',\n"
        "                expected_outcome: 'WHERE clause filters issue_date BETWEEN 2022-01-01 AND 2022-03-31',\n"
        "                fix_hint: \"Add WHERE issue_date BETWEEN '2022-01-01' AND '2022-03-31'\"}\n\n"
        "  Example (simple year filter — do NOT include boundary):\n"
        "    Question: 'Show all employees hired in 2015.'\n"
        "    → do NOT include 'boundary' key (open equality filter, not a bounded window)\n\n"
        "symmetry: Include when the question asks for a total that should equal the sum\n"
        "  of identifiable sub-groups, or when the answer decomposes into named parts.\n"
        "  Signals:\n"
        "    • 'total X … for each category / type / group'\n"
        "    • 'broken down by', 'split by'\n"
        "    • 'sum of A and B' where A and B are named sub-groups of the same entity\n"
        "  Provide a verification_sql that computes the grand total from the raw tables.\n"
        "  Example:\n"
        "    Question: 'What is the total loan amount issued, broken down by loan type?'\n"
        "    Sub-group sums (personal + mortgage + auto) must equal the grand total.\n"
        "    → include 'symmetry' key:\n"
        "      symmetry={verification_sql: 'SELECT SUM(amount) FROM loans',\n"
        "                fix_hint: 'Verify that GROUP BY loan_type sums equal SELECT SUM(amount) FROM loans.'}\n\n"
        "━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "- Do NOT include a grain test — that is generated by a separate call.\n"
        "- Include each test only when the listed signals are clearly present.\n"
        "- For boundary and scale, all signal conditions must hold (see individual sections).\n"
        "- For null, duplicate, ordering, symmetry, a single clear signal is sufficient.\n"
        "- IMPORTANT: Only include a test if you can fill ALL its REQUIRED fields.\n"
        "  If you cannot fill a required field, omit the entire test key instead.\n"
        "- 0–6 tests total (grain is excluded from this call).\n"
        "- All verification_sql queries must be valid SQLite SELECT queries against the raw database tables.\n"
        "- fix_hint must be concrete: name specific SQL clauses or expressions to add or change.\n"
    ),
    cache=True,
)

# ---------------------------------------------------------------------------
# Column alignment tool — unchanged
# ---------------------------------------------------------------------------

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
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "Your confidence in the expected_column_count. "
                    "'high'   — the question explicitly lists every output column "
                    "(e.g. 'List the name and GPA of each student'). "
                    "'medium' — you are confident in the count but the wording is somewhat implicit "
                    "(e.g. 'What is the name of the school with the highest SAT score?'). "
                    "'low'    — the question is genuinely ambiguous about which columns belong in "
                    "SELECT and a wrong count would penalise correct candidates; the "
                    "column_alignment test will be SKIPPED for this question."
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
        "required": ["reasoning", "confidence", "expected_column_count", "column_descriptions"],
    },
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
        "SPECIAL CASES (use confidence 'low' to skip the test):\n"
        "  'Show all information about X' / 'Show all details of X' / 'Show all fields of X'\n"
        "    → confidence: 'low' (column count is undefined — skip the test).\n"
        "  'For each X, show [something]' → count only the output items after 'show';\n"
        "    X itself is NOT an output column unless explicitly named in the SELECT output.\n"
        "    Example: 'For each department, show the number of employees' → 1 (just the count).\n"
        "    Example: 'For each department, show the department name and employee count' → 2.\n"
        "  'What is the X and Y for each Z?' → 2 (X and Y only; Z is the grouping key, not output).\n"
        "  Questions with vague wording like 'relevant info', 'necessary details', 'key data'\n"
        "    → confidence: 'low' (skip the test).\n\n"
        "Set confidence:\n"
        "  'high'   — the question explicitly lists every output column "
        "(e.g. 'List the name and GPA').\n"
        "  'medium' — you are confident in the count but wording is somewhat implicit "
        "(e.g. 'What is the school name...').\n"
        "  'low'    — the question is genuinely ambiguous about which columns belong in "
        "SELECT; the column_alignment test will be SKIPPED for this question.\n\n"
        "Now analyze the given question and evidence, then call column_alignment_spec "
        "with your step-by-step reasoning, confidence, the final expected_column_count, "
        "and a column_descriptions list with one short semantic label per column."
    ),
    cache=True,
)

# ---------------------------------------------------------------------------
# Backward-compat aliases (for existing test patches and imports)
# ---------------------------------------------------------------------------

_PLAN_TOOL = _OPTIONAL_TESTS_TOOL
_PLAN_SYSTEM = _OPTIONAL_TESTS_SYSTEM
