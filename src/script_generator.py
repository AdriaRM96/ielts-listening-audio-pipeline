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
QUESTION_TYPES_MARKER = "<<<QUESTION_TYPES>>>"
SCRIPT_MARKER = "<<<SCRIPT>>>"
QUESTIONS_MARKER = "<<<QUESTIONS>>>"
ANSWERS_MARKER = "<<<ANSWERS>>>"

# Parts 1 and 4 lean heavily on one dominant type in the real exam (dominant,
# occasional secondary type). Parts 2 and 3 genuinely mix 2-3 types with
# similar frequency in practice, so they get real avoidance instead of a soft
# nudge — see _type_variety_guidance().
DOMINANT_TYPE_PARTS: dict[int, tuple[str, str]] = {
    1: ("form/note completion", "matching"),
    4: ("note/summary completion", "short-answer"),
}
BALANCED_TYPE_PARTS = {2, 3}

DOMINANT_TYPE_LOOKBACK = 5
BALANCED_TYPE_LOOKBACK = 2
MIN_HISTORY_FOR_TYPE_NUDGE = 3

SECTION_SPECS: dict[int, str] = {
    1: (
        "Section 1 — everyday transactional conversation, exactly 2 speakers "
        "(e.g. booking a service, an enquiry, form-filling). Question types: "
        "form completion and note completion are the dominant types in real "
        "IELTS Listening — use one or both as the default, heavy on spelling "
        "and number accuracy. Matching appears only occasionally, as the "
        "exception rather than the norm. This is the easiest section; "
        "information is given fairly directly, but spelling/number "
        "distractors are common."
    ),
    2: (
        "Section 2 — everyday monologue, exactly 1 speaker (e.g. a guided "
        "tour, a talk about a facility or service, an announcement, an "
        "induction talk). Question types: multiple choice, matching (e.g. "
        "facilities to locations, or map/plan labelling), and note/table "
        "completion all appear with genuinely similar frequency in real "
        "tests — combine 2-3 of them across the section rather than using "
        "just one throughout. Easy-medium difficulty; longer uninterrupted "
        "stretches require sustained attention rather than back-and-forth "
        "tracking."
    ),
    3: (
        "Section 3 — academic conversation, 2 to 4 speakers (e.g. students "
        "discussing an assignment or project, with or without a tutor). "
        "Question types: multiple choice and matching opinions to speakers "
        "are the two dominant types here, appearing with similar frequency; "
        "short-answer appears too, but less often. Combine 2-3 types across "
        "the section rather than using just one throughout. Medium-hard "
        "difficulty — speakers may disagree or revise their views "
        "mid-conversation; this is where the self-correction/distractor "
        "pattern matters most."
    ),
    4: (
        "Section 4 — academic monologue/lecture, exactly 1 speaker, in a "
        "more formal, written-like register than Sections 1-3. Question "
        "types: note/summary completion (no word bank) is the classic "
        "dominant type here — it's specifically what makes this section the "
        "hardest — and should be the default. Short-answer appears "
        "sometimes as a secondary type, not a replacement. This is the "
        "hardest section: dense information delivered at a steady, "
        "uninterrupted pace with no back-and-forth to re-anchor attention."
    ),
}

SYSTEM_PROMPT = """You are an expert IELTS Listening test writer, generating one section at a time of an original, exam-realistic IELTS Listening mock test.

## Global rules

- Strict UK English in all question/instruction text. Note in your script generation which native-English accent the dialogue is written for, so voice selection can match it (default to British English unless a topic naturally suggests otherwise).
- Never reuse or imitate a real IELTS recording — invent the scenario and dialogue fresh every time.
- Numbers, dates, spelling of names, and proper nouns are the highest-frequency trap across all sections — include at least one such detail.
- Question order must follow audio chronology, not question-type grouping. Trace the script top to bottom and number questions in the order their answers are spoken, even if that interleaves or reorders question types.
- Every section has exactly 10 questions — never more, never fewer. Real IELTS Listening is always 40 questions total, 10 per part, numbered continuously across the whole test with no gaps (Part 1 = 1-10, Part 2 = 11-20, Part 3 = 21-30, Part 4 = 31-40). Before finishing, count your own questions and confirm there are exactly 10, numbered correctly for this part.

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

Return exactly these five blocks, in this order, with no commentary before, after, or between them:

<<<TOPIC_CATEGORY>>>
A short "category - specific topic" line, e.g. "booking/enquiry - car hire". One line only.
<<<QUESTION_TYPES>>>
A short comma-separated list of the question type(s) you actually used in this section, e.g. "form completion, note completion". One line only.
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
- Write exactly 10 questions — never more, never fewer. You'll be told the exact question numbers to use for this section; before finishing, count your own questions and confirm there are exactly 10, using precisely that range.

## Output format

Return exactly these three blocks, in this order, with no commentary before, after, or between them:

<<<QUESTION_TYPES>>>
A short comma-separated list of the question type(s) you actually used, e.g. "multiple choice, matching". One line only.
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
    question_types: list[str]
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
    return [
        e["topic_category"] for e in entries
        if e.get("part") == part and e.get("topic_category")
    ][-RECENT_TOPICS_PER_PART:]


def _recent_question_types(part: int, log_path: Path, limit: int) -> list[list[str]]:
    entries = _load_log(log_path)
    return [
        e["question_types"] for e in entries
        if e.get("part") == part and e.get("question_types")
    ][-limit:]


def _parse_question_types(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _record_generation(
    part: int,
    topic_category: str | None,
    question_types: list[str],
    log_path: Path,
) -> None:
    """Record one successful generation for cross-run topic/type variety.

    topic_category is None for --quiz (the transcript's topic isn't
    Gemini's to name — only the question types it chose are tracked there).
    """
    entries = _load_log(log_path)
    entries.append({
        "part": part,
        "topic_category": topic_category,
        "question_types": question_types,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_log(entries, log_path)


_BLOCK_RE = re.compile(
    r"<<<TOPIC_CATEGORY>>>\s*\n(?P<topic>.*?)\n<<<QUESTION_TYPES>>>\s*\n(?P<question_types>.*?)\n"
    r"<<<SCRIPT>>>\s*\n(?P<script>.*?)\n"
    r"<<<QUESTIONS>>>\s*\n(?P<questions>.*?)\n<<<ANSWERS>>>\s*\n(?P<answers>.*)",
    re.DOTALL,
)

_QUIZ_BLOCK_RE = re.compile(
    r"<<<QUESTION_TYPES>>>\s*\n(?P<question_types>.*?)\n"
    r"<<<QUESTIONS>>>\s*\n(?P<questions>.*?)\n<<<ANSWERS>>>\s*\n(?P<answers>.*)",
    re.DOTALL,
)

_CAST_SIZE_RULES: dict[int, tuple[int, int]] = {1: (2, 2), 2: (1, 1), 3: (2, 4), 4: (1, 1)}


def _parse_response(text: str, part: int) -> tuple[str, list[str], str, str, str]:
    """Split a raw Gemini response into (topic_category, question_types, script_text, questions, answers)."""
    m = _BLOCK_RE.search(text)
    if not m:
        raise ScriptParseError(
            "Response did not contain the five required blocks "
            "(<<<TOPIC_CATEGORY>>>, <<<QUESTION_TYPES>>>, <<<SCRIPT>>>, <<<QUESTIONS>>>, <<<ANSWERS>>>)."
        )
    topic = m.group("topic").strip()
    question_types = _parse_question_types(m.group("question_types"))
    script_text = m.group("script").strip() + "\n"
    questions = m.group("questions").strip()
    answers = m.group("answers").strip()

    if not question_types:
        raise ScriptParseError("The <<<QUESTION_TYPES>>> block was empty.")
    if not questions:
        raise ScriptParseError("The <<<QUESTIONS>>> block was empty.")
    if not answers:
        raise ScriptParseError("The <<<ANSWERS>>> block was empty.")

    return topic, question_types, script_text, questions, answers


def _parse_quiz_response(text: str) -> tuple[list[str], str, str]:
    """Split a raw Gemini quiz-only response into (question_types, questions, answers)."""
    m = _QUIZ_BLOCK_RE.search(text)
    if not m:
        raise ScriptParseError(
            "Response did not contain the three required blocks "
            "(<<<QUESTION_TYPES>>>, <<<QUESTIONS>>>, <<<ANSWERS>>>)."
        )
    question_types = _parse_question_types(m.group("question_types"))
    questions = m.group("questions").strip()
    answers = m.group("answers").strip()

    if not question_types:
        raise ScriptParseError("The <<<QUESTION_TYPES>>> block was empty.")
    if not questions:
        raise ScriptParseError("The <<<QUESTIONS>>> block was empty.")
    if not answers:
        raise ScriptParseError("The <<<ANSWERS>>> block was empty.")

    return question_types, questions, answers


def _validate_cast_size(script: ParsedScript, part: int) -> None:
    lo, hi = _CAST_SIZE_RULES[part]
    count = len(script.speakers)
    if not (lo <= count <= hi):
        expected = f"exactly {lo}" if lo == hi else f"{lo}-{hi}"
        raise ScriptParseError(
            f"Part {part} must have {expected} speaker(s), but the script has {count} "
            f"({', '.join(script.speakers)})."
        )


_QUESTION_NUMBER_RE = re.compile(r"(?<!\d)(\d{1,2})(?!\d)(?:[.\):]|(?=\s[A-Z_]))")


def _expected_question_range(part: int) -> range:
    """Real IELTS Listening is always 40 questions, 10 per part, numbered continuously."""
    start = 10 * (part - 1) + 1
    return range(start, start + 10)


def _validate_question_count(questions_text: str, part: int) -> None:
    """Every IELTS Listening section has exactly 10 questions — check none are missing.

    Gemini can silently under-generate a section (e.g. stop at question 7 of a
    10-question block) without any other part of the format looking wrong, so
    this can't be caught by the transcript parser — it needs its own check.
    """
    expected = set(_expected_question_range(part))
    found = {
        int(m.group(1))
        for m in _QUESTION_NUMBER_RE.finditer(questions_text)
        if int(m.group(1)) in expected
    }
    missing = sorted(expected - found)
    if missing:
        lo, hi = min(expected), max(expected)
        raise ScriptParseError(
            f"Part {part} question set is missing question(s) {missing} — IELTS Listening "
            f"requires exactly 10 questions per part, numbered {lo}-{hi} continuously with no gaps."
        )


def _type_variety_guidance(part: int, log_path: Path) -> str:
    """Cross-run question-type variety, matching real IELTS's uneven per-part distribution.

    Parts 1/4 lean heavily on one dominant type in the real exam — this only
    adds a soft nudge toward the occasional secondary type, and only once
    there's enough history to show it's been genuinely absent for a while
    (never on a near-empty log, and never a requirement to alternate).

    Parts 2/3 are genuinely balanced in practice, so this actively discourages
    repeating the exact same type combination as recent runs — the same
    avoid-repeats approach _recent_topics() already uses for topics.
    """
    if part in DOMINANT_TYPE_PARTS:
        dominant, secondary = DOMINANT_TYPE_PARTS[part]
        recent = _recent_question_types(part, log_path, DOMINANT_TYPE_LOOKBACK)
        if len(recent) < MIN_HISTORY_FOR_TYPE_NUDGE:
            return ""
        secondary_seen = any(secondary.lower() in (t.lower() for t in types) for types in recent)
        if secondary_seen:
            return ""
        dominant_cap = dominant[0].upper() + dominant[1:]
        return (
            f"\nThe last {len(recent)} Part {part} sections have all stuck to {dominant} "
            f"without any {secondary} — if it fits naturally this time, you could work in a "
            f"touch of {secondary} as well, but only if it doesn't feel forced. {dominant_cap} "
            "should still be the primary type either way."
        )

    if part in BALANCED_TYPE_PARTS:
        recent = _recent_question_types(part, log_path, BALANCED_TYPE_LOOKBACK)
        if not recent:
            return ""
        combos = "; ".join(", ".join(types) for types in recent)
        return (
            "\nAvoid repeating the same combination of question types used recently for this "
            f"section: {combos}. Choose a different mix from the types listed above."
        )

    return ""


def _user_prompt(
    part: int,
    topic_hint: str | None,
    recent_topics: list[str],
    log_path: Path,
) -> str:
    expected = _expected_question_range(part)
    lines = [
        f"Generate Part {part} of a fresh IELTS Listening mock test now.",
        "",
        SECTION_SPECS[part],
        f"\nNumber the questions {expected.start}-{expected[-1]} (exactly 10 questions).",
    ]
    type_guidance = _type_variety_guidance(part, log_path)
    if type_guidance:
        lines.append(type_guidance)
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

    contents: list = [_user_prompt(part, topic_hint, recent_topics, log_path)]
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
            topic, question_types, script_text, questions, answers = _parse_response(response_text, part)
            script = parse_script(script_text, part)
            _validate_cast_size(script, part)
            _validate_question_count(questions, part)
        except ScriptParseError as exc:
            last_error = exc
            print(f"  [Gemini] Part {part} attempt {attempt} failed validation: {exc}", file=sys.stderr)
            if attempt == MAX_ATTEMPTS:
                break
            contents.append(response_text)
            contents.append(
                "Your last response didn't match the required format: "
                f"{exc}\nFix this and resend the complete response — all five blocks, "
                "for this same section."
            )
            continue

        _record_generation(part, topic, question_types, log_path)
        return GeneratedSection(
            script=script,
            script_text=script_text,
            topic_category=topic,
            question_types=question_types,
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


def generate_quiz(
    part: int,
    script_text: str,
    log_path: Path = GENERATION_LOG_PATH,
) -> tuple[str, str]:
    """Generate a matching question set + answer key for an already-written script.

    Used for --quiz on a hand-written or previously-generated transcript,
    where the dialogue itself is fixed and only needs a matching quiz. Gets
    the same cross-run type-variety guidance as generate_section(), and
    records the type(s) chosen to the same log — with no topic_category,
    since the transcript's topic isn't Gemini's to name here.
    """
    from google.genai import types

    client = _get_client()
    expected = _expected_question_range(part)
    type_guidance = _type_variety_guidance(part, log_path)
    contents: list = [
        f"{SECTION_SPECS[part]}\n\n"
        f"Number the questions {expected.start}-{expected[-1]} (exactly 10 questions)."
        f"{type_guidance}\n\n"
        f"Here is the transcript for this section:\n\n{script_text}"
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
            question_types, questions, answers = _parse_quiz_response(response_text)
            _validate_question_count(questions, part)
        except ScriptParseError as exc:
            last_error = exc
            print(f"  [Gemini] Part {part} quiz attempt {attempt} failed validation: {exc}", file=sys.stderr)
            if attempt == MAX_ATTEMPTS:
                break
            contents.append(response_text)
            contents.append(
                f"Your last response didn't match the required format: {exc}\n"
                "Fix this and resend the complete response — all three blocks."
            )
            continue

        _record_generation(part, None, question_types, log_path)
        return questions, answers

    raise ScriptGenerationError(
        f"Part {part} quiz: Gemini's output still didn't validate after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )
