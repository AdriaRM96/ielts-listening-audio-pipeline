"""Parse IELTS Listening transcript files produced by the ielts-listening-generator skill.

Fixed input format (do not change):
    # GENDER: Name=female
    # GENDER: Name=male
    ...blank line optional...
    Name: spoken line
    Name: spoken line
    ...

Files are named part1.txt through part4.txt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

GENDER_LINE_RE = re.compile(
    r"^#\s*GENDER:\s*(?P<name>[^=]+?)\s*=\s*(?P<gender>male|female)\s*$",
    re.IGNORECASE,
)
TURN_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z .]{0,29}):\s+(?P<text>.+)$")
PART_FILENAME_RE = re.compile(r"^part(?P<num>[1-4])\.txt$", re.IGNORECASE)

VALID_GENDERS = {"male", "female"}


class ScriptParseError(Exception):
    """Raised when a transcript file is missing, empty, or malformed."""


@dataclass
class Turn:
    speaker: str
    gender: str | None
    text: str


@dataclass
class ParsedScript:
    part: int
    genders: dict[str, str] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        """Unique speaker names in order of first appearance."""
        return list(dict.fromkeys(t.speaker for t in self.turns))

    @property
    def is_dialogue(self) -> bool:
        """True if the script has more than one distinct speaker."""
        return len(self.speakers) > 1


def part_number_from_filename(path: str | Path) -> int:
    """Extract the part number (1-4) from a 'partN.txt' filename.

    Raises ScriptParseError if the filename doesn't match the expected pattern.
    """
    name = Path(path).name
    m = PART_FILENAME_RE.match(name)
    if not m:
        raise ScriptParseError(
            f"Filename '{name}' does not match the expected 'partN.txt' pattern (N = 1-4)."
        )
    return int(m.group("num"))


def parse_script(text: str, part: int) -> ParsedScript:
    """Parse transcript text into a ParsedScript.

    The optional '# GENDER: Name=gender' header block must appear before any
    spoken turns (blank lines and other '#' comments in the header are
    tolerated and skipped). Every non-blank line after the header must match
    'Speaker: text' — a stray line that doesn't is treated as a malformed
    file and raises ScriptParseError rather than being silently dropped or
    misread as dialogue.
    """
    if not text.strip():
        raise ScriptParseError(f"Part {part}: transcript file is empty.")

    lines = text.splitlines()
    genders: dict[str, str] = {}
    i = 0

    # Header block: consume leading blank lines and '#' comments.
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("#"):
            m = GENDER_LINE_RE.match(stripped)
            if m:
                genders[m.group("name").strip()] = m.group("gender").lower()
            i += 1
            continue
        break

    turns: list[Turn] = []
    for lineno, raw_line in enumerate(lines[i:], start=i + 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        m = TURN_RE.match(stripped)
        if not m:
            raise ScriptParseError(
                f"Part {part}, line {lineno}: expected 'Speaker: text' format, "
                f"got: {stripped!r}"
            )
        speaker = m.group("name").strip()
        turns.append(Turn(speaker=speaker, gender=genders.get(speaker), text=m.group("text").strip()))

    if not turns:
        raise ScriptParseError(f"Part {part}: no spoken turns found after the header.")

    return ParsedScript(part=part, genders=genders, turns=turns)


def parse_script_file(path: str | Path) -> ParsedScript:
    """Read and parse a partN.txt file, inferring the part number from its filename."""
    path = Path(path)
    part = part_number_from_filename(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ScriptParseError(f"Part {part}: file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ScriptParseError(f"Part {part}: file is not valid UTF-8 text: {path}") from exc
    return parse_script(text, part)
