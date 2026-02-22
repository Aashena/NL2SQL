"""
Unit tests for src/schema_linking/schema_linker.py

Tests the two-iteration schema linking process (S₁ precise, S₂ recall) with
mocked LLM calls and a mocked FAISS index.  No live API calls are made.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.base import LLMResponse, CacheableText
from src.indexing.faiss_index import FieldMatch
from src.indexing.lsh_index import CellMatch
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import link_schema, LinkedSchemas


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# available_fields: (table, column, short_summary, long_summary)
AVAILABLE_FIELDS: list[tuple[str, str, str, str]] = [
    # students table — 8 columns
    ("students", "id", "Student ID", "The unique identifier for each student. PRIMARY KEY."),
    ("students", "name", "Student name", "Full name of the student."),
    ("students", "gpa", "Grade point average", "GPA on a 4.0 scale."),
    ("students", "department_id", "Department FK", "Foreign key to departments.id."),
    ("students", "year", "Academic year", "Year of enrollment."),
    ("students", "email", "Email address", "Student contact email."),
    ("students", "age", "Student age", "Age in years."),
    ("students", "status", "Enrollment status", "Active or graduated."),
    # courses table — 8 columns
    ("courses", "id", "Course ID", "Unique identifier for each course. PRIMARY KEY."),
    ("courses", "title", "Course title", "Name of the course."),
    ("courses", "credits", "Credit hours", "Number of credit hours."),
    ("courses", "department_id", "Department FK", "Foreign key to departments.id."),
    ("courses", "professor_id", "Professor FK", "Foreign key to professors.id."),
    ("courses", "level", "Course level", "Undergraduate or graduate level."),
    ("courses", "capacity", "Max enrollment", "Maximum number of students allowed."),
    ("courses", "semester", "Semester offered", "Which semester the course runs."),
    # enrollments table — 8 columns
    ("enrollments", "id", "Enrollment ID", "Unique enrollment record. PRIMARY KEY."),
    ("enrollments", "student_id", "Student FK", "Foreign key to students.id."),
    ("enrollments", "course_id", "Course FK", "Foreign key to courses.id."),
    ("enrollments", "grade", "Grade received", "Letter grade for the enrollment."),
    ("enrollments", "semester", "Semester", "Semester of enrollment."),
    ("enrollments", "year", "Year", "Year of enrollment."),
    ("enrollments", "status", "Status", "Completed, in-progress, or dropped."),
    ("enrollments", "score", "Numeric score", "Numeric score out of 100."),
    # professors table — 8 columns
    ("professors", "id", "Professor ID", "Unique identifier for professors. PRIMARY KEY."),
    ("professors", "name", "Professor name", "Full name of the professor."),
    ("professors", "department_id", "Department FK", "Foreign key to departments.id."),
    ("professors", "title", "Academic title", "Professor, Associate Professor, etc."),
    ("professors", "email", "Email address", "Professor contact email."),
    ("professors", "office", "Office location", "Room number and building."),
    ("professors", "phone", "Phone number", "Contact phone number."),
    ("professors", "hired_year", "Year hired", "Year the professor was hired."),
    # departments table — 8 columns
    ("departments", "id", "Department ID", "Unique identifier for departments. PRIMARY KEY."),
    ("departments", "name", "Department name", "Full name of the department."),
    ("departments", "code", "Department code", "Short code for the department."),
    ("departments", "dean", "Dean name", "Name of the department dean."),
    ("departments", "building", "Building", "Main building for the department."),
    ("departments", "phone", "Phone", "Main phone for the department."),
    ("departments", "budget", "Annual budget", "Department annual budget."),
    ("departments", "established", "Founded year", "Year the department was established."),
]

# Full DDL with PK and FK info in the format produced by schema_formatter
FULL_DDL = """\
-- Table: students
CREATE TABLE students (
  id INTEGER PRIMARY KEY,  -- The unique identifier for each student. PRIMARY KEY.
  name TEXT,  -- Full name of the student.
  gpa REAL,  -- GPA on a 4.0 scale.
  department_id INTEGER,  -- Foreign key to departments.id.
  year INTEGER,  -- Year of enrollment.
  email TEXT,  -- Student contact email.
  age INTEGER,  -- Age in years.
  status TEXT  -- Active or graduated.
);
-- Foreign keys: department_id REFERENCES departments(id)
-- Example row: (1, 'Alice', 3.9, 1, 2021, 'alice@uni.edu', 20, 'Active')

-- Table: courses
CREATE TABLE courses (
  id INTEGER PRIMARY KEY,  -- Unique identifier for each course. PRIMARY KEY.
  title TEXT,  -- Name of the course.
  credits INTEGER,  -- Number of credit hours.
  department_id INTEGER,  -- Foreign key to departments.id.
  professor_id INTEGER,  -- Foreign key to professors.id.
  level TEXT,  -- Undergraduate or graduate level.
  capacity INTEGER,  -- Maximum number of students allowed.
  semester TEXT  -- Which semester the course runs.
);
-- Foreign keys: department_id REFERENCES departments(id), professor_id REFERENCES professors(id)
-- Example row: (1, 'Intro to CS', 3, 1, 1, 'undergraduate', 30, 'Fall')

-- Table: enrollments
CREATE TABLE enrollments (
  id INTEGER PRIMARY KEY,  -- Unique enrollment record. PRIMARY KEY.
  student_id INTEGER,  -- Foreign key to students.id.
  course_id INTEGER,  -- Foreign key to courses.id.
  grade TEXT,  -- Letter grade for the enrollment.
  semester TEXT,  -- Semester of enrollment.
  year INTEGER,  -- Year of enrollment.
  status TEXT,  -- Completed, in-progress, or dropped.
  score REAL  -- Numeric score out of 100.
);
-- Foreign keys: student_id REFERENCES students(id), course_id REFERENCES courses(id)
-- Example row: (1, 1, 1, 'A', 'Fall', 2021, 'Completed', 95.0)

-- Table: professors
CREATE TABLE professors (
  id INTEGER PRIMARY KEY,  -- Unique identifier for professors. PRIMARY KEY.
  name TEXT,  -- Full name of the professor.
  department_id INTEGER,  -- Foreign key to departments.id.
  title TEXT,  -- Professor, Associate Professor, etc.
  email TEXT,  -- Professor contact email.
  office TEXT,  -- Room number and building.
  phone TEXT,  -- Contact phone number.
  hired_year INTEGER  -- Year the professor was hired.
);
-- Foreign keys: department_id REFERENCES departments(id)
-- Example row: (1, 'Dr. Smith', 1, 'Professor', 'smith@uni.edu', 'Room 101', '555-1234', 2010)

-- Table: departments
CREATE TABLE departments (
  id INTEGER PRIMARY KEY,  -- Unique identifier for departments. PRIMARY KEY.
  name TEXT,  -- Full name of the department.
  code TEXT,  -- Short code for the department.
  dean TEXT,  -- Name of the department dean.
  building TEXT,  -- Main building for the department.
  phone TEXT,  -- Main phone for the department.
  budget REAL,  -- Department annual budget.
  established INTEGER  -- Year the department was established.
);
-- Example row: (1, 'Computer Science', 'CS', 'Dean Johnson', 'Tech Building', '555-9876', 1000000.0, 1970)
"""

# Full Markdown schema
FULL_MARKDOWN = """\
## Table: students
*Student information*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Student ID | 1, 2, 3 |
| name | TEXT | Student name | Alice, Bob |
| gpa | REAL | Grade point average | 3.9, 3.5 |
| department_id | INTEGER (FK) | Department FK | 1, 2 |
| year | INTEGER | Academic year | 2021, 2022 |
| email | TEXT | Email address | alice@uni.edu |
| age | INTEGER | Student age | 20, 22 |
| status | TEXT | Enrollment status | Active |

## Table: courses
*Course information*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Course ID | 1, 2, 3 |
| title | TEXT | Course title | Intro to CS |
| credits | INTEGER | Credit hours | 3, 4 |
| department_id | INTEGER (FK) | Department FK | 1, 2 |
| professor_id | INTEGER (FK) | Professor FK | 1, 2 |
| level | TEXT | Course level | undergraduate |
| capacity | INTEGER | Max enrollment | 30, 25 |
| semester | TEXT | Semester offered | Fall, Spring |

## Table: enrollments
*Enrollment records*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Enrollment ID | 1, 2, 3 |
| student_id | INTEGER (FK) | Student FK | 1, 2 |
| course_id | INTEGER (FK) | Course FK | 1, 2 |
| grade | TEXT | Grade received | A, B, C |
| semester | TEXT | Semester | Fall, Spring |
| year | INTEGER | Year | 2021, 2022 |
| status | TEXT | Status | Completed |
| score | REAL | Numeric score | 95.0, 87.5 |

## Table: professors
*Professor information*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Professor ID | 1, 2, 3 |
| name | TEXT | Professor name | Dr. Smith |
| department_id | INTEGER (FK) | Department FK | 1, 2 |
| title | TEXT | Academic title | Professor |
| email | TEXT | Email address | smith@uni.edu |
| office | TEXT | Office location | Room 101 |
| phone | TEXT | Phone number | 555-1234 |
| hired_year | INTEGER | Year hired | 2010 |

## Table: departments
*Department information*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Department ID | 1, 2, 3 |
| name | TEXT | Department name | Computer Science |
| code | TEXT | Department code | CS |
| dean | TEXT | Dean name | Dean Johnson |
| building | TEXT | Building | Tech Building |
| phone | TEXT | Phone | 555-9876 |
| budget | REAL | Annual budget | 1000000.0 |
| established | INTEGER | Founded year | 1970 |
"""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_faiss_index():
    """Returns 15 FieldMatch objects across 5 tables."""
    mock = MagicMock()
    fields = [
        FieldMatch(table="students", column="id", similarity_score=0.95,
                   short_summary="Student ID", long_summary="Unique student identifier. PRIMARY KEY."),
        FieldMatch(table="students", column="gpa", similarity_score=0.90,
                   short_summary="Grade point average", long_summary="GPA on a 4.0 scale."),
        FieldMatch(table="students", column="name", similarity_score=0.85,
                   short_summary="Student name", long_summary="Full name of the student."),
        FieldMatch(table="students", column="department_id", similarity_score=0.70,
                   short_summary="Department FK", long_summary="Foreign key to departments.id."),
        FieldMatch(table="courses", column="id", similarity_score=0.80,
                   short_summary="Course ID", long_summary="Unique course identifier. PRIMARY KEY."),
        FieldMatch(table="courses", column="title", similarity_score=0.75,
                   short_summary="Course title", long_summary="Name of the course."),
        FieldMatch(table="courses", column="credits", similarity_score=0.65,
                   short_summary="Credit hours", long_summary="Number of credit hours."),
        FieldMatch(table="enrollments", column="id", similarity_score=0.78,
                   short_summary="Enrollment ID", long_summary="Unique enrollment record. PRIMARY KEY."),
        FieldMatch(table="enrollments", column="student_id", similarity_score=0.72,
                   short_summary="Student FK", long_summary="Foreign key to students.id."),
        FieldMatch(table="enrollments", column="grade", similarity_score=0.68,
                   short_summary="Grade received", long_summary="Letter grade."),
        FieldMatch(table="professors", column="id", similarity_score=0.60,
                   short_summary="Professor ID", long_summary="Unique professor identifier. PRIMARY KEY."),
        FieldMatch(table="professors", column="name", similarity_score=0.58,
                   short_summary="Professor name", long_summary="Full name of the professor."),
        FieldMatch(table="departments", column="id", similarity_score=0.55,
                   short_summary="Department ID", long_summary="Unique department identifier. PRIMARY KEY."),
        FieldMatch(table="departments", column="name", similarity_score=0.50,
                   short_summary="Department name", long_summary="Full name of the department."),
        FieldMatch(table="enrollments", column="course_id", similarity_score=0.62,
                   short_summary="Course FK", long_summary="Foreign key to courses.id."),
    ]
    mock.query.return_value = fields
    return mock


@pytest.fixture
def mock_grounding_context():
    return GroundingContext(
        matched_cells=[
            CellMatch(table="students", column="gpa", matched_value="3.9",
                      similarity_score=0.95, exact_match=True),
        ],
        schema_hints=["gpa", "name"],
        few_shot_examples=[],
    )


# Standard mock LLM responses
S1_RESPONSE = LLMResponse(tool_inputs=[{
    "selected_fields": [
        {"table": "students", "column": "gpa", "reason": "Need GPA for filtering"},
        {"table": "students", "column": "name", "reason": "Need student names"},
    ]
}])

S2_RESPONSE = LLMResponse(tool_inputs=[{
    "selected_fields": [
        {"table": "students", "column": "department_id", "reason": "Might need for join"},
        {"table": "enrollments", "column": "grade", "reason": "Could be relevant"},
    ]
}])


def make_mock_client(s1_resp=S1_RESPONSE, s2_resp=S2_RESPONSE):
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=[s1_resp, s2_resp])
    return mock_client


# ---------------------------------------------------------------------------
# Test 1: S₁ ⊆ S₂
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s1_subset_of_s2(mock_faiss_index, mock_grounding_context):
    """Every field in s1_fields must appear in s2_fields."""
    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client()
        result = await link_schema(
            question="What is the average GPA of students in each department?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    s1_set = set(result.s1_fields)
    s2_set = set(result.s2_fields)
    assert s1_set.issubset(s2_set), (
        f"S₁ not subset of S₂. In S₁ but not S₂: {s1_set - s2_set}"
    )


# ---------------------------------------------------------------------------
# Test 2: Primary keys are always included
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_keys_always_included(mock_faiss_index, mock_grounding_context):
    """If a table is referenced in S₁, its PK column should appear in s1_fields."""
    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client()
        result = await link_schema(
            question="What is the average GPA of students?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    s1_tables = {t for t, _c in result.s1_fields}
    s1_cols = {(t, c) for t, c in result.s1_fields}

    # For every table in S₁, there should be a PK column present
    # students table has PK 'id'
    assert "students" in s1_tables, "Expected students table in S₁"
    assert ("students", "id") in s1_cols, (
        "Primary key 'students.id' should be auto-added to S₁"
    )


# ---------------------------------------------------------------------------
# Test 3: Foreign keys bridge tables
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_foreign_keys_bridge_tables(mock_faiss_index, mock_grounding_context):
    """FK columns that bridge selected tables should appear in s1_fields or s2_fields."""
    # Use an S₁ response that selects students and departments
    s1_with_dept = LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": "students", "column": "name", "reason": "Student name"},
            {"table": "departments", "column": "name", "reason": "Department name"},
        ]
    }])

    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client(s1_resp=s1_with_dept)
        result = await link_schema(
            question="What department does each student belong to?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    s1_fields_set = set(result.s1_fields)
    # students.department_id is the FK that bridges students → departments
    assert ("students", "department_id") in s1_fields_set, (
        "FK 'students.department_id' should be auto-added when both students and departments are in S₁"
    )


# ---------------------------------------------------------------------------
# Test 4: S₁ smaller than S₂
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s1_smaller_than_s2(mock_faiss_index, mock_grounding_context):
    """S₁ should be smaller than S₂ (S₂ casts a wider net)."""
    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client()
        result = await link_schema(
            question="What is the average GPA of students in each department?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    assert len(result.s1_fields) < len(result.s2_fields), (
        f"Expected S₁ ({len(result.s1_fields)} fields) to be smaller than "
        f"S₂ ({len(result.s2_fields)} fields)"
    )


# ---------------------------------------------------------------------------
# Test 5: FAISS pre-filtering top_k
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_faiss_pre_filtering_top_k(mock_faiss_index, mock_grounding_context):
    """FAISSIndex.query should be called with top_k = settings.faiss_top_k."""
    from src.config.settings import settings

    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client()
        await link_schema(
            question="What is the average GPA?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    mock_faiss_index.query.assert_called_once_with(
        "What is the average GPA?", top_k=settings.faiss_top_k
    )


# ---------------------------------------------------------------------------
# Test 6: Exactly 2 API calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_api_calls_made(mock_faiss_index, mock_grounding_context):
    """Exactly 2 generate() calls must be made (one for S₁, one for S₂)."""
    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_client = make_mock_client()
        mock_get_client.return_value = mock_client
        await link_schema(
            question="What is the average GPA of students?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    assert mock_client.generate.call_count == 2, (
        f"Expected 2 generate() calls, got {mock_client.generate.call_count}"
    )


# ---------------------------------------------------------------------------
# Test 7: S₁ DDL rendered as DDL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_rendered_as_ddl(mock_faiss_index, mock_grounding_context):
    """s1_ddl should start with 'CREATE TABLE' or '-- Table:'."""
    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client()
        result = await link_schema(
            question="What is the average GPA of students?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    assert result.s1_ddl.startswith("CREATE TABLE") or result.s1_ddl.startswith("-- Table:"), (
        f"s1_ddl should start with 'CREATE TABLE' or '-- Table:', got: {result.s1_ddl[:50]!r}"
    )


# ---------------------------------------------------------------------------
# Test 8: S₂ Markdown rendered as Markdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_rendered_as_markdown(mock_faiss_index, mock_grounding_context):
    """s2_markdown should contain '## Table:' and '|' characters."""
    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client()
        result = await link_schema(
            question="What is the average GPA of students?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    assert "## Table:" in result.s2_markdown, "s2_markdown should contain '## Table:'"
    assert "|" in result.s2_markdown, "s2_markdown should contain '|' pipe characters"


# ---------------------------------------------------------------------------
# Test 9: Cell matches influence selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cell_matches_influence_selection(mock_faiss_index):
    """When grounding_context has a CellMatch for students.gpa, the mock
    S₁ response returns it and it appears in s1_fields."""
    grounding_with_cell = GroundingContext(
        matched_cells=[
            CellMatch(table="students", column="gpa", matched_value="3.9",
                      similarity_score=0.95, exact_match=True),
        ],
        schema_hints=[],
        few_shot_examples=[],
    )

    # S₁ response explicitly includes the matched column
    s1_resp = LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": "students", "column": "gpa", "reason": "Matched cell value 3.9"},
        ]
    }])
    s2_resp = LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": "students", "column": "name", "reason": "Might be needed"},
        ]
    }])

    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client(s1_resp=s1_resp, s2_resp=s2_resp)
        result = await link_schema(
            question="Which students have GPA above 3.5?",
            evidence="",
            grounding_context=grounding_with_cell,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    assert ("students", "gpa") in result.s1_fields, (
        "students.gpa should appear in s1_fields when the mock returns it"
    )


# ---------------------------------------------------------------------------
# Test 10: Prompt caching applied
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_caching_applied(mock_faiss_index, mock_grounding_context):
    """The system list in the first generate() call must contain a CacheableText
    with cache=True."""
    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_client = make_mock_client()
        mock_get_client.return_value = mock_client
        await link_schema(
            question="What is the average GPA of students?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    # Inspect the first generate() call
    first_call_kwargs = mock_client.generate.call_args_list[0].kwargs
    system_blocks: list[CacheableText] = first_call_kwargs.get("system", [])
    cached_blocks = [b for b in system_blocks if isinstance(b, CacheableText) and b.cache]
    assert len(cached_blocks) >= 1, (
        "At least one CacheableText with cache=True must be in the system prompt"
    )


# ---------------------------------------------------------------------------
# Test 11: Single-table database
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_linking_with_single_table_db():
    """Works with a single-table database with 5 columns. No crash, valid result."""
    single_table_fields = [
        ("students", "id", "Student ID", "Unique student ID. PRIMARY KEY."),
        ("students", "name", "Student name", "Full name."),
        ("students", "gpa", "GPA", "Grade point average."),
        ("students", "age", "Age", "Student age."),
        ("students", "email", "Email", "Contact email."),
    ]
    single_ddl = """\
-- Table: students
CREATE TABLE students (
  id INTEGER PRIMARY KEY,  -- Unique student ID. PRIMARY KEY.
  name TEXT,  -- Full name.
  gpa REAL,  -- Grade point average.
  age INTEGER,  -- Student age.
  email TEXT  -- Contact email.
);
-- Example row: (1, 'Alice', 3.9, 20, 'alice@uni.edu')
"""
    single_markdown = """\
## Table: students
*Student data*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Student ID | 1, 2, 3 |
| name | TEXT | Student name | Alice, Bob |
| gpa | REAL | GPA | 3.9, 3.5 |
| age | INTEGER | Age | 20, 22 |
| email | TEXT | Email | alice@uni.edu |
"""

    mock_faiss = MagicMock()
    mock_faiss.query.return_value = [
        FieldMatch(table="students", column="id", similarity_score=0.9,
                   short_summary="Student ID", long_summary="Unique student ID. PRIMARY KEY."),
        FieldMatch(table="students", column="gpa", similarity_score=0.85,
                   short_summary="GPA", long_summary="Grade point average."),
        FieldMatch(table="students", column="name", similarity_score=0.80,
                   short_summary="Student name", long_summary="Full name."),
    ]

    s1_resp = LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": "students", "column": "gpa", "reason": "Need GPA"},
            {"table": "students", "column": "name", "reason": "Need names"},
        ]
    }])
    s2_resp = LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": "students", "column": "age", "reason": "Might be useful"},
        ]
    }])

    grounding = GroundingContext(matched_cells=[], schema_hints=[], few_shot_examples=[])

    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=[s1_resp, s2_resp])
        mock_get_client.return_value = mock_client
        result = await link_schema(
            question="What is the average GPA?",
            evidence="",
            grounding_context=grounding,
            faiss_index=mock_faiss,
            full_ddl=single_ddl,
            full_markdown=single_markdown,
            available_fields=single_table_fields,
        )

    assert isinstance(result, LinkedSchemas)
    assert len(result.s1_fields) > 0
    assert len(result.s2_fields) >= len(result.s1_fields)
    # S₁ ⊆ S₂
    assert set(result.s1_fields).issubset(set(result.s2_fields))


# ---------------------------------------------------------------------------
# Test 12: Hallucinated fields are filtered out
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_selected_fields_in_original_schema(mock_faiss_index, mock_grounding_context):
    """When the mock LLM returns a hallucinated column, it must be filtered out."""
    s1_with_hallucination = LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": "students", "column": "gpa", "reason": "Real column"},
            {"table": "students", "column": "nonexistent_col", "reason": "Hallucinated!"},
            {"table": "ghost_table", "column": "phantom_col", "reason": "Also hallucinated"},
        ]
    }])
    s2_with_hallucination = LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": "students", "column": "name", "reason": "Real column"},
            {"table": "courses", "column": "fake_field_xyz", "reason": "Hallucinated!"},
        ]
    }])

    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_get_client.return_value = make_mock_client(
            s1_resp=s1_with_hallucination,
            s2_resp=s2_with_hallucination,
        )
        result = await link_schema(
            question="What is the average GPA?",
            evidence="",
            grounding_context=mock_grounding_context,
            faiss_index=mock_faiss_index,
            full_ddl=FULL_DDL,
            full_markdown=FULL_MARKDOWN,
            available_fields=AVAILABLE_FIELDS,
        )

    available_set = {(t, c) for t, c, _ss, _ls in AVAILABLE_FIELDS}

    for field_pair in result.s1_fields:
        assert field_pair in available_set, (
            f"Hallucinated field {field_pair} found in s1_fields"
        )

    for field_pair in result.s2_fields:
        assert field_pair in available_set, (
            f"Hallucinated field {field_pair} found in s2_fields"
        )

    # Specifically check these hallucinations are absent
    assert ("students", "nonexistent_col") not in result.s1_fields
    assert ("ghost_table", "phantom_col") not in result.s1_fields
    assert ("courses", "fake_field_xyz") not in result.s2_fields
