"""Download a PDF from a URL and extract its text content."""

import sys
import tempfile
import urllib.request

import pymupdf


def download_pdf(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(dest, "wb") as f:
            f.write(resp.read())


def extract_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n\n".join(pages)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pdf_url>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    print(f"Downloading PDF from {url}...", file=sys.stderr)
    download_pdf(url, tmp_path)

    print("Extracting text...", file=sys.stderr)
    text = extract_text(tmp_path)

    output_path = tempfile.mktemp(suffix=".txt")
    with open(output_path, "w") as f:
        f.write(text)

    print(output_path)


if __name__ == "__main__":
    main()
