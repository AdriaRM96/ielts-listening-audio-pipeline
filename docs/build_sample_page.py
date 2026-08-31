#!/usr/bin/env python3
"""Regenerates the collapsible Transcript / Questions & answer key sections in
docs/index.html from docs/sample-test/ (the source of truth), so the live
demo page can never silently drift out of sync with the actual sample test.

Run this after any change to docs/sample-test/partN.txt or
docs/sample-test/questions_and_answers.md:

    python docs/build_sample_page.py

Only the marked <!-- BEGIN/END PART N DETAILS --> regions are touched —
everything else in index.html (styling, audio players, intro, footer) is
left exactly as-is.

Requires the `markdown` package (pip install markdown) — dev-only, not a
runtime dependency of the pipeline itself.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

try:
    import markdown
except ImportError as exc:
    raise SystemExit(
        "The 'markdown' package is required to run this script: pip install markdown"
    ) from exc

DOCS_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = DOCS_DIR / "sample-test"
INDEX_PATH = DOCS_DIR / "index.html"
PARTS = (1, 2, 3, 4)

PART_HEADING_RE = re.compile(r"^# Part (\d) — .+$", re.MULTILINE)


def _split_qa_by_part(qa_text: str) -> dict[int, str]:
    """Split questions_and_answers.md into {part_number: markdown_body}."""
    matches = list(PART_HEADING_RE.finditer(qa_text))
    if not matches:
        raise ValueError("No '# Part N — ...' headings found in questions_and_answers.md")

    sections: dict[int, str] = {}
    for i, m in enumerate(matches):
        part = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(qa_text)
        sections[part] = qa_text[start:end].strip()
    return sections


def _render_transcript_block(part: int) -> str:
    text = (SAMPLE_DIR / f"part{part}.txt").read_text(encoding="utf-8").strip()
    escaped = html.escape(text)
    return f"<details>\n<summary>Transcript</summary>\n<pre>{escaped}</pre>\n</details>"


def _render_qa_block(part: int, qa_sections: dict[int, str]) -> str:
    if part not in qa_sections:
        raise ValueError(f"questions_and_answers.md has no section for Part {part}")
    # The source content uses runs of underscores as literal fill-in-the-blank
    # markers (e.g. "1. __________"), never as intentional emphasis — but
    # Markdown parses "__..__" as <strong>/<em>. Escape every underscore so
    # blanks render as literal underscores instead of garbled nested tags.
    body_source = qa_sections[part].replace("_", "\\_")
    body_html = markdown.markdown(body_source, extensions=["tables"])
    return (
        "<details>\n<summary>Questions &amp; answer key</summary>\n"
        f'<div class="qa-content">\n{body_html}\n</div>\n</details>'
    )


def build() -> None:
    qa_text = (SAMPLE_DIR / "questions_and_answers.md").read_text(encoding="utf-8")
    qa_sections = _split_qa_by_part(qa_text)

    index_html = INDEX_PATH.read_text(encoding="utf-8")

    for part in PARTS:
        transcript_block = _render_transcript_block(part)
        qa_block = _render_qa_block(part, qa_sections)

        begin = f"<!-- BEGIN PART {part} DETAILS -->"
        end = f"<!-- END PART {part} DETAILS -->"
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(index_html):
            raise ValueError(f"Could not find markers {begin} / {end} in {INDEX_PATH}")

        replacement = f"{begin}\n    {transcript_block}\n    {qa_block}\n    {end}"
        index_html = pattern.sub(replacement, index_html)

    INDEX_PATH.write_text(index_html, encoding="utf-8")
    print(f"Regenerated collapsible sections for Parts {', '.join(map(str, PARTS))} in {INDEX_PATH}")


if __name__ == "__main__":
    build()
