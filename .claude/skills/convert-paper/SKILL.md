---
name: convert-paper
description: Convert a scientific PDF paper from a URL to well-formatted markdown
argument-hint: <pdf-url>
---

Convert a scientific PDF paper to well-formatted markdown.

Paper URL: $ARGUMENTS

## Steps

1. Run the PDF text extraction script using the NL2SQL conda environment:
   ```
   source ~/miniforge3/etc/profile.d/conda.sh && conda activate NL2SQL && python3 .claude/skills/convert-paper/pdf_to_text.py "$ARGUMENTS"
   ```
   The script prints the path to the extracted text file.

2. Read the extracted text file.

3. Convert the raw text into clean, well-formatted markdown following these rules:
   - `#` for the paper title, `##` for main sections, `###` for subsections, `####` for sub-subsections
   - Format authors and affiliations clearly
   - Mark the abstract with its own `## Abstract` section
   - Use LaTeX notation (`$...$` and `$$...$$`) for mathematical expressions and equations
   - Format tables using markdown table syntax
   - Use fenced code blocks with language tags (```sql, ```python, etc.) for code/queries
   - Preserve figure and table captions (prefix with **Figure N:** or **Table N:**)
   - Use **bold** and *italic* appropriately
   - Use bullet points and numbered lists where the paper uses them
   - Format references as a numbered list
   - Clean up PDF extraction artifacts: fix broken hyphenation, remove odd spacing, merge split paragraphs
   - Include ALL content — do not truncate or summarize any section including appendices

4. Derive a filename slug from the paper title (lowercase, underscores, no special chars) and write the markdown to `references/<slug>.md`.

5. Report the output file path when done.
