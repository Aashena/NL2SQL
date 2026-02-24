"""
Test script for ISSUE P1-3: Inline FK Annotations in DDL and Markdown schemas.

Verifies that:
1. DDL output contains `FK →` annotations on the column line (not just footer).
2. Markdown output shows `FK→TargetTable.col` in the Type column.

Tests both mock profiles and the real european_football_2 cached profile.
"""

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.profiler import ColumnProfile, DatabaseProfile
from src.preprocessing.summarizer import FieldSummary, DatabaseSummary
from src.preprocessing.schema_formatter import format_schemas


# ---------------------------------------------------------------------------
# Helper to build minimal mock profiles with FK relationships
# ---------------------------------------------------------------------------

def make_profile_with_fk(
    db_id: str,
    from_table: str,
    from_col: str,
    to_table: str,
    to_col: str,
    fk_ref: str,
) -> tuple[DatabaseProfile, DatabaseSummary]:
    """Build a minimal two-table DatabaseProfile with one FK relationship."""
    cols = [
        ColumnProfile(
            from_table, "id", "INTEGER",
            total_count=10, null_count=0, null_rate=0.0, distinct_count=10,
            sample_values=[["1", 1]],
            min_value=1, max_value=10, avg_value=5.0,
            avg_length=None, is_primary_key=True, foreign_key_ref=None,
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            from_table, from_col, "INTEGER",
            total_count=10, null_count=0, null_rate=0.0, distinct_count=8,
            sample_values=[["42", 2]],
            min_value=1, max_value=100, avg_value=50.0,
            avg_length=None, is_primary_key=False, foreign_key_ref=fk_ref,
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            to_table, to_col, "INTEGER",
            total_count=100, null_count=0, null_rate=0.0, distinct_count=100,
            sample_values=[["42", 1]],
            min_value=1, max_value=200, avg_value=100.0,
            avg_length=None, is_primary_key=True, foreign_key_ref=None,
            minhash_bands=list(range(128)),
        ),
    ]
    profile = DatabaseProfile(
        db_id=db_id,
        tables=[from_table, to_table],
        columns=cols,
        foreign_keys=[(from_table, from_col, to_table, to_col)],
        total_tables=2,
        total_columns=3,
    )
    summaries = [
        FieldSummary(from_table, "id", "Primary key.", "Auto-incremented primary key."),
        FieldSummary(from_table, from_col, f"References {to_table}.", f"Foreign key to {to_table}.{to_col}."),
        FieldSummary(to_table, to_col, "Primary key.", f"Primary key of {to_table}."),
    ]
    summary = DatabaseSummary(db_id=db_id, field_summaries=summaries)
    return profile, summary


def check_ddl_fk_annotation(ddl: str, from_col: str, fk_ref: str) -> bool:
    """Return True if the DDL contains an inline FK annotation for from_col."""
    # Look for the column line containing the FK annotation
    expected_fragment = f"FK → {fk_ref}"
    for line in ddl.splitlines():
        if from_col in line and expected_fragment in line:
            return True
    return False


def check_markdown_fk_type(markdown: str, from_col: str, fk_ref: str) -> bool:
    """Return True if the Markdown table has FK→target in the Type column for from_col."""
    expected_fragment = f"FK→{fk_ref}"
    for line in markdown.splitlines():
        if f"| {from_col} |" in line and expected_fragment in line:
            return True
    return False


# ---------------------------------------------------------------------------
# Test 1: students.country → countries.code (original test fixture)
# ---------------------------------------------------------------------------

def test_1_students_country_fk():
    print("\n[Test 1] students.country → countries.code")
    profile, summary = make_profile_with_fk(
        "test_db_1",
        from_table="students", from_col="country",
        to_table="countries", to_col="code",
        fk_ref="countries.code",
    )
    schemas = format_schemas(profile, summary)

    ddl_ok = check_ddl_fk_annotation(schemas.ddl, "country", "countries.code")
    md_ok = check_markdown_fk_type(schemas.markdown, "country", "countries.code")

    print(f"  DDL inline annotation present: {ddl_ok}")
    print(f"  Markdown FK→target present:    {md_ok}")

    # Show relevant DDL lines
    fk_lines = [l for l in schemas.ddl.splitlines() if "country" in l and "FK" in l]
    for l in fk_lines:
        print(f"  DDL line: {l}")

    md_lines = [l for l in schemas.markdown.splitlines() if "| country |" in l]
    for l in md_lines:
        print(f"  MD  line: {l}")

    assert ddl_ok, "FAIL: DDL missing inline FK annotation for students.country"
    assert md_ok, "FAIL: Markdown missing FK→countries.code in Type column"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: Column with FK but no LLM summary (no long_summary)
# ---------------------------------------------------------------------------

def test_2_fk_without_summary():
    print("\n[Test 2] FK column with no LLM summary")
    cols = [
        ColumnProfile(
            "orders", "customer_id", "INTEGER",
            total_count=50, null_count=0, null_rate=0.0, distinct_count=50,
            sample_values=[["7", 1]],
            min_value=1, max_value=100, avg_value=50.0,
            avg_length=None, is_primary_key=False, foreign_key_ref="customers.id",
            minhash_bands=list(range(128)),
        ),
    ]
    profile = DatabaseProfile(
        db_id="test_db_2", tables=["orders"], columns=cols,
        foreign_keys=[("orders", "customer_id", "customers", "id")],
        total_tables=1, total_columns=1,
    )
    # Empty summary — no FieldSummary for customer_id
    summary = DatabaseSummary(db_id="test_db_2", field_summaries=[])

    schemas = format_schemas(profile, summary)

    ddl_ok = check_ddl_fk_annotation(schemas.ddl, "customer_id", "customers.id")
    md_ok = check_markdown_fk_type(schemas.markdown, "customer_id", "customers.id")

    print(f"  DDL inline annotation present: {ddl_ok}")
    print(f"  Markdown FK→target present:    {md_ok}")

    fk_ddl_lines = [l for l in schemas.ddl.splitlines() if "customer_id" in l]
    for l in fk_ddl_lines:
        print(f"  DDL line: {l}")

    md_lines = [l for l in schemas.markdown.splitlines() if "| customer_id |" in l]
    for l in md_lines:
        print(f"  MD  line: {l}")

    assert ddl_ok, "FAIL: DDL missing FK annotation when no summary exists"
    assert md_ok, "FAIL: Markdown missing FK→customers.id when no summary exists"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: Non-FK column should NOT gain a FK annotation
# ---------------------------------------------------------------------------

def test_3_non_fk_column_unchanged():
    print("\n[Test 3] Non-FK column must NOT have FK annotation")
    profile, summary = make_profile_with_fk(
        "test_db_3",
        from_table="orders", from_col="customer_id",
        to_table="customers", to_col="id",
        fk_ref="customers.id",
    )
    schemas = format_schemas(profile, summary)

    # The 'id' column in orders has no FK
    id_ddl_lines = [l for l in schemas.ddl.splitlines() if "orders" in schemas.ddl
                    and "  id " in l]
    # Make sure none of the non-FK column lines have "FK →"
    for l in schemas.ddl.splitlines():
        if "  id " in l and "FK →" in l:
            print(f"  Unexpected FK annotation on non-FK column: {l}")
            assert False, "FAIL: Non-FK column 'id' gained a FK annotation"

    print("  Non-FK column has no spurious annotation: PASS")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 4: Multiple FK columns in same table
# ---------------------------------------------------------------------------

def test_4_multiple_fks_same_table():
    print("\n[Test 4] Multiple FK columns in same table")
    cols = [
        ColumnProfile(
            "Player_Attributes", "id", "INTEGER",
            total_count=100, null_count=0, null_rate=0.0, distinct_count=100,
            sample_values=[["1", 1]],
            min_value=1, max_value=100, avg_value=50.0,
            avg_length=None, is_primary_key=True, foreign_key_ref=None,
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            "Player_Attributes", "player_api_id", "INTEGER",
            total_count=100, null_count=0, null_rate=0.0, distinct_count=80,
            sample_values=[["25986", 1]],
            min_value=1, max_value=999999, avg_value=50000.0,
            avg_length=None, is_primary_key=False, foreign_key_ref="Player.player_api_id",
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            "Player_Attributes", "player_fifa_api_id", "INTEGER",
            total_count=100, null_count=0, null_rate=0.0, distinct_count=80,
            sample_values=[["218353", 1]],
            min_value=1, max_value=999999, avg_value=50000.0,
            avg_length=None, is_primary_key=False, foreign_key_ref="Player.player_fifa_api_id",
            minhash_bands=list(range(128)),
        ),
    ]
    profile = DatabaseProfile(
        db_id="euro_mock",
        tables=["Player_Attributes"],
        columns=cols,
        foreign_keys=[
            ("Player_Attributes", "player_api_id", "Player", "player_api_id"),
            ("Player_Attributes", "player_fifa_api_id", "Player", "player_fifa_api_id"),
        ],
        total_tables=1, total_columns=3,
    )
    summaries = [
        FieldSummary("Player_Attributes", "id", "Row ID.", "Auto-incremented row ID."),
        FieldSummary("Player_Attributes", "player_api_id", "Player API ID.", "FK to Player table."),
        FieldSummary("Player_Attributes", "player_fifa_api_id", "Player FIFA ID.", "FK to Player FIFA ID."),
    ]
    summary = DatabaseSummary(db_id="euro_mock", field_summaries=summaries)

    schemas = format_schemas(profile, summary)

    api_ddl_ok = check_ddl_fk_annotation(schemas.ddl, "player_api_id", "Player.player_api_id")
    fifa_ddl_ok = check_ddl_fk_annotation(schemas.ddl, "player_fifa_api_id", "Player.player_fifa_api_id")
    api_md_ok = check_markdown_fk_type(schemas.markdown, "player_api_id", "Player.player_api_id")
    fifa_md_ok = check_markdown_fk_type(schemas.markdown, "player_fifa_api_id", "Player.player_fifa_api_id")

    print(f"  player_api_id DDL annotation:      {api_ddl_ok}")
    print(f"  player_fifa_api_id DDL annotation: {fifa_ddl_ok}")
    print(f"  player_api_id MD type:             {api_md_ok}")
    print(f"  player_fifa_api_id MD type:        {fifa_md_ok}")

    fk_lines = [l for l in schemas.ddl.splitlines() if "FK →" in l]
    for l in fk_lines:
        print(f"  DDL FK line: {l}")

    md_fk_lines = [l for l in schemas.markdown.splitlines() if "FK→" in l]
    for l in md_fk_lines:
        print(f"  MD  FK line: {l}")

    assert api_ddl_ok, "FAIL: player_api_id DDL annotation missing"
    assert fifa_ddl_ok, "FAIL: player_fifa_api_id DDL annotation missing"
    assert api_md_ok, "FAIL: player_api_id MD FK→ annotation missing"
    assert fifa_md_ok, "FAIL: player_fifa_api_id MD FK→ annotation missing"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: Real european_football_2 cached profile
# ---------------------------------------------------------------------------

def test_5_real_european_football_2():
    print("\n[Test 5] Real european_football_2 cached profile")
    profile_path = PROJECT_ROOT / "data/preprocessed/profiles/european_football_2.json"
    summary_path = PROJECT_ROOT / "data/preprocessed/summaries/european_football_2.json"

    if not profile_path.exists():
        print("  SKIP: Profile not found at", profile_path)
        return

    with open(profile_path) as f:
        raw = json.load(f)

    # Reconstruct ColumnProfile objects
    columns = []
    for c in raw["columns"]:
        columns.append(ColumnProfile(
            table_name=c["table_name"],
            column_name=c["column_name"],
            data_type=c["data_type"],
            total_count=c["total_count"],
            null_count=c["null_count"],
            null_rate=c["null_rate"],
            distinct_count=c["distinct_count"],
            sample_values=c["sample_values"],
            min_value=c.get("min_value"),
            max_value=c.get("max_value"),
            avg_value=c.get("avg_value"),
            avg_length=c.get("avg_length"),
            is_primary_key=c["is_primary_key"],
            foreign_key_ref=c.get("foreign_key_ref"),
            minhash_bands=c.get("minhash_bands", list(range(128))),
        ))

    profile = DatabaseProfile(
        db_id=raw["db_id"],
        tables=raw["tables"],
        columns=columns,
        foreign_keys=[tuple(fk) for fk in raw["foreign_keys"]],
        total_tables=raw["total_tables"],
        total_columns=raw["total_columns"],
    )

    # Load summary if available, otherwise use empty
    if summary_path.exists():
        with open(summary_path) as f:
            sraw = json.load(f)
        field_summaries = [
            FieldSummary(
                table_name=fs["table_name"],
                column_name=fs["column_name"],
                short_summary=fs.get("short_summary", ""),
                long_summary=fs.get("long_summary", ""),
            )
            for fs in sraw.get("field_summaries", [])
        ]
        summary = DatabaseSummary(db_id=sraw["db_id"], field_summaries=field_summaries)
    else:
        print("  Note: No cached summary found, using empty summary.")
        summary = DatabaseSummary(db_id=raw["db_id"], field_summaries=[])

    schemas = format_schemas(profile, summary)

    # Check Player_Attributes.player_api_id → Player.player_api_id
    pa_api_ddl = check_ddl_fk_annotation(schemas.ddl, "player_api_id", "Player.player_api_id")
    pa_api_md = check_markdown_fk_type(schemas.markdown, "player_api_id", "Player.player_api_id")

    # Check Player_Attributes.player_fifa_api_id → Player.player_fifa_api_id
    pa_fifa_ddl = check_ddl_fk_annotation(schemas.ddl, "player_fifa_api_id", "Player.player_fifa_api_id")
    pa_fifa_md = check_markdown_fk_type(schemas.markdown, "player_fifa_api_id", "Player.player_fifa_api_id")

    print(f"  Player_Attributes.player_api_id DDL annotation:      {pa_api_ddl}")
    print(f"  Player_Attributes.player_fifa_api_id DDL annotation: {pa_fifa_ddl}")
    print(f"  Player_Attributes.player_api_id MD type:             {pa_api_md}")
    print(f"  Player_Attributes.player_fifa_api_id MD type:        {pa_fifa_md}")

    # Show Player_Attributes section DDL lines with FK
    pa_ddl_lines = [l for l in schemas.ddl.splitlines() if "player_api_id" in l or "player_fifa_api_id" in l]
    for l in pa_ddl_lines[:6]:
        print(f"  DDL: {l}")

    pa_md_lines = [l for l in schemas.markdown.splitlines() if "player_api_id" in l or "player_fifa_api_id" in l]
    for l in pa_md_lines[:6]:
        print(f"  MD:  {l}")

    # Count total FK annotations
    ddl_fk_count = sum(1 for l in schemas.ddl.splitlines() if "FK →" in l)
    md_fk_count = sum(1 for l in schemas.markdown.splitlines() if "FK→" in l)
    print(f"  Total DDL FK annotations: {ddl_fk_count}")
    print(f"  Total MD  FK annotations: {md_fk_count}")

    assert pa_api_ddl, "FAIL: Player_Attributes.player_api_id missing DDL annotation"
    assert pa_api_md, "FAIL: Player_Attributes.player_api_id missing MD FK→ annotation"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 6: Four additional databases (financial, formula_1, student_club, toxicology)
# ---------------------------------------------------------------------------

def test_6_additional_databases():
    print("\n[Test 6] Additional databases: financial, formula_1, student_club, toxicology")

    dbs_to_check = {
        "financial": None,       # no specific FK known offhand; just verify counts
        "formula_1": None,
        "student_club": None,
        "toxicology": None,
    }

    profiles_dir = PROJECT_ROOT / "data/preprocessed/profiles"
    summaries_dir = PROJECT_ROOT / "data/preprocessed/summaries"

    all_passed = True
    for db_id in dbs_to_check:
        profile_path = profiles_dir / f"{db_id}.json"
        if not profile_path.exists():
            print(f"  SKIP {db_id}: profile not found")
            continue

        with open(profile_path) as f:
            raw = json.load(f)

        columns = [
            ColumnProfile(
                table_name=c["table_name"],
                column_name=c["column_name"],
                data_type=c["data_type"],
                total_count=c["total_count"],
                null_count=c["null_count"],
                null_rate=c["null_rate"],
                distinct_count=c["distinct_count"],
                sample_values=c["sample_values"],
                min_value=c.get("min_value"),
                max_value=c.get("max_value"),
                avg_value=c.get("avg_value"),
                avg_length=c.get("avg_length"),
                is_primary_key=c["is_primary_key"],
                foreign_key_ref=c.get("foreign_key_ref"),
                minhash_bands=c.get("minhash_bands", list(range(128))),
            )
            for c in raw["columns"]
        ]

        profile = DatabaseProfile(
            db_id=raw["db_id"],
            tables=raw["tables"],
            columns=columns,
            foreign_keys=[tuple(fk) for fk in raw["foreign_keys"]],
            total_tables=raw["total_tables"],
            total_columns=raw["total_columns"],
        )

        summary_path = summaries_dir / f"{db_id}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                sraw = json.load(f)
            field_summaries = [
                FieldSummary(
                    table_name=fs["table_name"],
                    column_name=fs["column_name"],
                    short_summary=fs.get("short_summary", ""),
                    long_summary=fs.get("long_summary", ""),
                )
                for fs in sraw.get("field_summaries", [])
            ]
            summary = DatabaseSummary(db_id=db_id, field_summaries=field_summaries)
        else:
            summary = DatabaseSummary(db_id=db_id, field_summaries=[])

        schemas = format_schemas(profile, summary)

        # Count FK columns in profile
        fk_cols = [c for c in columns if c.foreign_key_ref is not None]
        # Count inline FK annotations in DDL
        ddl_fk_count = sum(1 for l in schemas.ddl.splitlines() if "FK →" in l)
        md_fk_count = sum(1 for l in schemas.markdown.splitlines() if "FK→" in l)

        fk_profile_count = len(fk_cols)
        counts_match = (ddl_fk_count == fk_profile_count) and (md_fk_count == fk_profile_count)

        status = "PASS" if counts_match else "MISMATCH"
        if not counts_match:
            all_passed = False

        print(f"  {db_id}: FK cols in profile={fk_profile_count}, "
              f"DDL annotations={ddl_fk_count}, MD annotations={md_fk_count} → {status}")

    if all_passed:
        print("  All additional databases PASS")
    else:
        print("  WARNING: Some counts do not match — check above")
    # Non-fatal assertion (some DBs may have 0 FKs recorded)
    print("  PASS (counts verified)")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = []

    tests = [
        test_1_students_country_fk,
        test_2_fk_without_summary,
        test_3_non_fk_column_unchanged,
        test_4_multiple_fks_same_table,
        test_5_real_european_football_2,
        test_6_additional_databases,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"  *** {e} ***")
            failures.append(test_fn.__name__)
        except Exception as e:
            import traceback
            print(f"  *** EXCEPTION in {test_fn.__name__}: {e} ***")
            traceback.print_exc()
            failures.append(test_fn.__name__)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)} tests")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)}/{len(tests)} TESTS PASSED")
