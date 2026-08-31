"""Generates IELTS Listening mock test transcripts using Gemini on Vertex AI.

This is optional: the pipeline works perfectly well with transcripts you
already have (from the ielts-listening-generator skill, or written by
hand). This module exists for `--generate`, which writes a fresh script
with Gemini when you don't already have one on hand.

Uses the same GOOGLE_APPLICATION_CREDENTIALS service-account key already
set up for Text-to-Speech — no separate credentials mechanism. That key's
IAM role needs to be extended with "Vertex AI User" (see the README).
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .parser import ParsedScript, ScriptParseError, parse_script

MODEL_ID = "gemini-2.5-flash"
LOCATION = "us-central1"

MAX_ATTEMPTS = 3
RECENT_TOPICS_PER_PART = 6
LOG_MAX_ENTRIES = 20

GENERATION_LOG_PATH = Path(__file__).resolve().parent.parent / "script_generation_log.json"

TOPIC_MARKER = "<<<TOPIC_CATEGORY>>>"
SCRIPT_MARKER = "<<<SCRIPT>>>"
QUESTIONS_MARKER = "<<<QUESTIONS>>>"
ANSWERS_MARKER = "<<<ANSWERS>>>"

SECTION_SPECS: dict[int, str] = {
    1: (
        "Section 1 — everyday transactional conversation, exactly 2 speakers "
        "(e.g. booking a service, an enquiry, form-filling). Question types: "
        "form completion, note completion — heavy on spelling and number "
        "accuracy. This is the easiest section; information is given fairly "
        "directly, but spelling/number distractors are common."
    ),
    2: (
        "Section 2 — everyday monologue, exactly 1 speaker (e.g. a guided "
        "tour, a talk about a facility or service, an announcement, an "
        "induction talk). Question types: multiple choice, matching (e.g. "
        "facilities to locations), note/table completion. Easy-medium "
        "difficulty; longer uninterrupted stretches require sustained "
        "attention rather than back-and-forth tracking."
    ),
    3: (
        "Section 3 — academic conversation, 2 to 4 speakers (e.g. students "
        "discussing an assignment or project, with or without a tutor). "
        "Question types: matching opinions to speakers, multiple choice, "
        "short-answer. Medium-hard difficulty — speakers may disagree or "
        "revise their views mid-conversation; this is where the "
        "self-correction/distractor pattern matters most."
    ),
    4: (
        "Section 4 — academic monologue/lecture, exactly 1 speaker, in a "
        "more formal, written-like register than Sections 1-3. Question "
        "types: note/summary completion (no word bank — the hardest "
        "variant), short-answer. This is the hardest section: dense "
        "information delivered at a steady, uninterrupted pace with no "
        "back-and-forth to re-anchor attention."
    ),
}

SYSTEM_PROMPT = """You are an expert IELTS Listening test writer, generating one section at a time of an original, exam-realistic IELTS Listening mock test.

## Global rules

- Strict UK English in all question/instruction text. Note in your script generation which native-English accent the dialogue is written for, so voice selection can match it (default to British English unless a topic naturally suggests otherwise).
- Never reuse or imitate a real IELTS recording — invent the scenario and dialogue fresh every time.
- Numbers, dates, spelling of names, and proper nouns are the highest-frequency trap across all sections — include at least one such detail.
- Question order must follow audio chronology, not question-type grouping. Trace the script top to bottom and number questions in the order their answers are spoken, even if that interleaves or reorders question types.

## Script-writing rules

- Natural spoken register, not written prose: include false starts, self-corrections, and fillers where natural (e.g. "It's on Tuesday — sorry, actually, Wednesday — at the main hall").
- The self-correction/distractor pattern is non-optional: at least once in this section, a speaker must say something that sounds like an answer, then correct it (e.g. "the cost is forty pounds — oh wait, that went up last month, it's forty five now"). This is the single most-tested feature of real IELTS Listening audio; skipping it makes the exercise unrealistically easy.

## Speaker-label format — critical, a downstream script parses this mechanically

- The transcript must start with one "# GENDER: Name=female" or "# GENDER: Name=male" line per speaker (top of file), then a blank line, then the dialogue turns as "Name: spoken line".
- Fix the cast (speaker names) before writing dialogue. Use the exact same spelling and capitalisation on every turn — "Sarah" must never become "sarah" or "Dr Sarah" partway through.
- Monologues use exactly one speaker label, repeated on every line, with zero variation. Critically: never write a bare line break in the middle of one speaker's turn, even for a long paragraph — each newline in the transcript must start a new "Name: text" line. Either keep a long turn on a single unbroken line, or, if you split it across several lines for readability, repeat "Name: " at the start of every one of those lines.
- Speaker labels must be 1 to 30 characters, letters and single spaces only — no periods, parentheses, numbers, or hyphens (e.g. "Doctor Hale", never "Dr. Hale").
- Never start a spoken line with a capitalised word followed by a colon (e.g. "Important: ..."), since that would be misread as a new speaker label.

## Output format

Return exactly these four blocks, in this order, with no commentary before, after, or between them:

<<<TOPIC_CATEGORY>>>
A short "category - specific topic" line, e.g. "booking/enquiry - car hire". One line only.
<<<SCRIPT>>>
The full transcript for this section, following every rule above.
<<<QUESTIONS>>>
The matching question set for this section (following the question types and chronology rule above), including word-limit instructions where relevant.
<<<ANSWERS>>>
The answer key for the questions above.
"""

QUIZ_SYSTEM_PROMPT = """You are an expert IELTS Listening test writer. You will be given the transcript for one section of an IELTS Listening test (already written — do not modify, rewrite, or comment on it). Your only job is to write the matching question set and answer key for it.

## Rules

- Question order must follow audio chronology, not question-type grouping. Trace the transcript top to bottom and number questions in the order their answers are spoken, even if that interleaves or reorders question types.
- Include word-limit instructions where relevant (e.g. "Write NO MORE THAN TWO WORDS").
- Base every question strictly on information actually present in the transcript — never invent details not in the text.

## Output format

Return exactly these two blocks, with no commentary before, after, or between them:

<<<QUESTIONS>>>
The matching question set for this section, following the question types and chronology rule above.
<<<ANSWERS>>>
The answer key for the questions above.
"""


class ScriptGenerationError(Exception):
    """Raised when Gemini generation fails, or its output can't be validated after retries."""


@dataclass
class GeneratedSection:
    script: ParsedScript
    script_text: str
    topic_category: str
    questions: str
    answers: str


def _get_client():
    """Construct a Gemini client on Vertex AI, reusing the TTS service-account key."""
    try:
        from google import genai
    except ImportError as exc:
        raise ScriptGenerationError(
            "google-genai is not installed. Run: pip install -r requirements.txt"
        ) from exc

    load_dotenv()
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not key_path:
        raise ScriptGenerationError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. --generate reuses the same "
            "service-account key set up for Text-to-Speech — see the README's "
            "'Enable automatic script generation with Gemini' section."
        )
    if not Path(key_path).is_file():
        raise ScriptGenerationError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to a file that doesn't exist: {key_path}"
        )

    try:
        with open(key_path, encoding="utf-8") as f:
            project_id = json.load(f)["project_id"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ScriptGenerationError(
            f"Could not read a 'project_id' field from the service-account key at {key_path}"
        ) from exc

    try:
        return genai.Client(vertexai=True, project=project_id, location=LOCATION)
    except Exception as exc:  # noqa: BLE001 - surface any auth/client-construction failure clearly
        raise ScriptGenerationError(f"Could not create a Gemini client: {exc}") from exc


def _load_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    try:
        with log_path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_log(entries: list[dict], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(entries[-LOG_MAX_ENTRIES:], f, indent=2)


def _recent_topics(part: int, log_path: Path) -> list[str]:
    entries = _load_log(log_path)
    return [e["topic_category"] for e in entries if e.get("part") == part][-RECENT_TOPICS_PER_PART:]


def _record_topic(part: int, topic_category: str, log_path: Path) -> None:
    entries = _load_log(log_path)
    entries.append({
        "part": part,
        "topic_category": topic_category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_log(entries, log_path)


_BLOCK_RE = re.compile(
    r"<<<TOPIC_CATEGORY>>>\s*\n(?P<topic>.*?)\n<<<SCRIPT>>>\s*\n(?P<script>.*?)\n"
    r"<<<QUESTIONS>>>\s*\n(?P<questions>.*?)\n<<<ANSWERS>>>\s*\n(?P<answers>.*)",
    re.DOTALL,
)

_QUIZ_BLOCK_RE = re.compile(
    r"<<<QUESTIONS>>>\s*\n(?P<questions>.*?)\n<<<ANSWERS>>>\s*\n(?P<answers>.*)",
    re.DOTALL,
)

_CAST_SIZE_RULES: dict[int, tuple[int, int]] = {1: (2, 2), 2: (1, 1), 3: (2, 4), 4: (1, 1)}


def _parse_response(text: str, part: int) -> tuple[str, str, str, str]:
    """Split a raw Gemini response into (topic_category, script_text, questions, answers)."""
    m = _BLOCK_RE.search(text)
    if not m:
        raise ScriptParseError(
            "Response did not contain the four required blocks "
            "(<<<TOPIC_CATEGORY>>>, <<<SCRIPT>>>, <<<QUESTIONS>>>, <<<ANSWERS>>>)."
        )
    topic = m.group("topic").strip()
    script_text = m.group("script").strip() + "\n"
    questions = m.group("questions").strip()
    answers = m.group("answers").strip()

    if not questions:
        raise ScriptParseError("The <<<QUESTIONS>>> block was empty.")
    if not answers:
        raise ScriptParseError("The <<<ANSWERS>>> block was empty.")

    return topic, script_text, questions, answers


def _parse_quiz_response(text: str) -> tuple[str, str]:
    """Split a raw Gemini quiz-only response into (questions, answers)."""
    m = _QUIZ_BLOCK_RE.search(text)
    if not m:
        raise ScriptParseError(
            "Response did not contain the two required blocks (<<<QUESTIONS>>>, <<<ANSWERS>>>)."
        )
    questions = m.group("questions").strip()
    answers = m.group("answers").strip()

    if not questions:
        raise ScriptParseError("The <<<QUESTIONS>>> block was empty.")
    if not answers:
        raise ScriptParseError("The <<<ANSWERS>>> block was empty.")

    return questions, answers


def _validate_cast_size(script: ParsedScript, part: int) -> None:
    lo, hi = _CAST_SIZE_RULES[part]
    count = len(script.speakers)
    if not (lo <= count <= hi):
        expected = f"exactly {lo}" if lo == hi else f"{lo}-{hi}"
        raise ScriptParseError(
            f"Part {part} must have {expected} speaker(s), but the script has {count} "
            f"({', '.join(script.speakers)})."
        )


def _user_prompt(part: int, topic_hint: str | None, recent_topics: list[str]) -> str:
    lines = [
        f"Generate Part {part} of a fresh IELTS Listening mock test now.",
        "",
        SECTION_SPECS[part],
    ]
    if recent_topics:
        lines.append(
            "\nAvoid repeating these recently-used topics for this section: "
            + "; ".join(recent_topics) + "."
        )
    if topic_hint:
        lines.append(f"\nTry to incorporate this theme if it fits naturally: {topic_hint}.")
    return "\n".join(lines)


def generate_section(
    part: int,
    topic_hint: str | None = None,
    log_path: Path = GENERATION_LOG_PATH,
) -> GeneratedSection:
    """Generate, validate, and return one section, retrying on validation failure."""
    from google.genai import types

    client = _get_client()
    recent_topics = _recent_topics(part, log_path)

    contents: list = [_user_prompt(part, topic_hint, recent_topics)]
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    # This is straightforward creative writing + format-following, not a
                    # reasoning task — Gemini 2.5's default "thinking" mode burns tens of
                    # thousands of tokens here for no benefit and can crowd out the actual
                    # response. max_output_tokens is a hard safety cap either way.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=4096,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK/network error with context
            raise ScriptGenerationError(f"Gemini request failed for Part {part}: {exc}") from exc

        usage = response.usage_metadata
        print(
            f"  [Gemini] Part {part} attempt {attempt}: "
            f"{usage.prompt_token_count} input + {usage.candidates_token_count} output tokens",
            file=sys.stderr,
        )

        response_text = response.text or ""

        try:
            topic, script_text, questions, answers = _parse_response(response_text, part)
            script = parse_script(script_text, part)
            _validate_cast_size(script, part)
        except ScriptParseError as exc:
            last_error = exc
            print(f"  [Gemini] Part {part} attempt {attempt} failed validation: {exc}", file=sys.stderr)
            if attempt == MAX_ATTEMPTS:
                break
            contents.append(response_text)
            contents.append(
                "Your last response didn't match the required format: "
                f"{exc}\nFix this and resend the complete response — all four blocks, "
                "for this same section."
            )
            continue

        _record_topic(part, topic, log_path)
        return GeneratedSection(
            script=script,
            script_text=script_text,
            topic_category=topic,
            questions=questions,
            answers=answers,
        )

    raise ScriptGenerationError(
        f"Part {part}: Gemini's output still didn't validate after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


def generate_full_test(
    topic_hint: str | None = None,
    log_path: Path = GENERATION_LOG_PATH,
    parts: list[int] | None = None,
) -> dict[int, GeneratedSection]:
    """Generate one or more sections. Defaults to all 4 parts."""
    parts = parts or [1, 2, 3, 4]
    return {part: generate_section(part, topic_hint, log_path) for part in parts}


def generate_quiz(part: int, script_text: str) -> tuple[str, str]:
    """Generate a matching question set + answer key for an already-written script.

    Used for --quiz on a hand-written or previously-generated transcript,
    where the dialogue itself is fixed and only needs a matching quiz.
    """
    from google.genai import types

    client = _get_client()
    contents: list = [
        f"{SECTION_SPECS[part]}\n\nHere is the transcript for this section:\n\n{script_text}"
    ]
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=QUIZ_SYSTEM_PROMPT,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=4096,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK/network error with context
            raise ScriptGenerationError(f"Gemini request failed for Part {part} quiz: {exc}") from exc

        usage = response.usage_metadata
        print(
            f"  [Gemini] Part {part} quiz attempt {attempt}: "
            f"{usage.prompt_token_count} input + {usage.candidates_token_count} output tokens",
            file=sys.stderr,
        )

        response_text = response.text or ""

        try:
            return _parse_quiz_response(response_text)
        except ScriptParseError as exc:
            last_error = exc
            print(f"  [Gemini] Part {part} quiz attempt {attempt} failed validation: {exc}", file=sys.stderr)
            if attempt == MAX_ATTEMPTS:
                break
            contents.append(response_text)
            contents.append(
                f"Your last response didn't match the required format: {exc}\n"
                "Fix this and resend the complete response — both blocks."
            )
            continue

    raise ScriptGenerationError(
        f"Part {part} quiz: Gemini's output still didn't validate after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )
