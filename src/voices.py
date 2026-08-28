"""Voice pools and gender/accent-aware assignment for Google Cloud Text-to-Speech.

Voice names below are real Neural2 voices, confirmed against a live call to
the Google Cloud TTS `ListVoices` API. If Google renames or retires one,
update the pools here — nothing else in the codebase needs to change.

Parts 1 and 2 always use standard British English (en-GB), matching the
default register of the exam. Parts 3 and 4 — the harder sections, where the
real IELTS Listening test sometimes features other native English accents —
draw from a wider pool that also includes Australian, Indian, and American
English. en-NZ, en-CA, en-ZA, and en-IE have no Neural2 voices in the Google
Cloud catalogue at time of writing, so they aren't included.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_LANGUAGE_CODE = "en-GB"

# Standard pool used for Parts 1 and 2.
UK_VOICE_POOLS: dict[str, list[str]] = {
    "female": ["en-GB-Neural2-A", "en-GB-Neural2-C", "en-GB-Neural2-F"],
    "male": ["en-GB-Neural2-B", "en-GB-Neural2-D"],
}

# Extended pool used for Parts 3 and 4.
ACCENT_VOICE_POOLS: dict[str, dict[str, list[str]]] = {
    "en-GB": UK_VOICE_POOLS,
    "en-AU": {
        "female": ["en-AU-Neural2-A", "en-AU-Neural2-C"],
        "male": ["en-AU-Neural2-B", "en-AU-Neural2-D"],
    },
    "en-IN": {
        "female": ["en-IN-Neural2-A", "en-IN-Neural2-D"],
        "male": ["en-IN-Neural2-B", "en-IN-Neural2-C"],
    },
    "en-US": {
        "female": ["en-US-Neural2-C", "en-US-Neural2-E"],
        "male": ["en-US-Neural2-A", "en-US-Neural2-D"],
    },
}

# Parts where accent variation beyond en-GB is allowed.
ACCENT_PARTS = {3, 4}

# voice_name -> the language_code it must be synthesized with.
VOICE_LOCALE: dict[str, str] = {
    voice: locale
    for locale, pools in ACCENT_VOICE_POOLS.items()
    for voices in pools.values()
    for voice in voices
}

USAGE_LOG_PATH = Path(__file__).resolve().parent.parent / "voice_usage_log.json"


def _pools_for_part(part: int) -> dict[str, list[str]]:
    """Return the {gender: [voice_names]} pool available for this part."""
    if part not in ACCENT_PARTS:
        return UK_VOICE_POOLS

    merged: dict[str, list[str]] = {"female": [], "male": []}
    for pools in ACCENT_VOICE_POOLS.values():
        for gender, voices in pools.items():
            merged[gender].extend(voices)
    return merged


def _all_voices_for_part(part: int) -> list[str]:
    return [v for voices in _pools_for_part(part).values() for v in voices]


def _load_usage_log(path: Path = USAGE_LOG_PATH) -> dict:
    if not path.exists():
        return {"counts": {}}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"counts": {}}
    data.setdefault("counts", {})
    return data


def _save_usage_log(log: dict, path: Path = USAGE_LOG_PATH) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, sort_keys=True)


def _pick_least_used(pool: list[str], counts: dict[str, int]) -> str:
    return min(pool, key=lambda v: counts.get(v, 0))


def _with_locale(voice_name: str) -> tuple[str, str]:
    return voice_name, VOICE_LOCALE.get(voice_name, DEFAULT_LANGUAGE_CODE)


def assign_dialogue_voices(
    speakers: list[str],
    genders: dict[str, str],
    part: int,
    usage_log_path: Path = USAGE_LOG_PATH,
) -> dict[str, tuple[str, str]]:
    """Map each dialogue speaker to a distinct (voice_name, language_code).

    Each speaker is matched to the least-used voice in their declared
    gender's pool for this part (Parts 3/4 draw from a wider set of accents;
    Parts 1/2 stay en-GB only), avoiding voices already used by another
    speaker in this same script. Falls back to the full pool for the part
    (any gender) with a WARNING to stderr when a speaker's gender is
    missing, unrecognised, or when their gender's pool is exhausted by
    earlier speakers in the same script.
    """
    pools = _pools_for_part(part)
    all_voices = _all_voices_for_part(part)

    log = _load_usage_log(usage_log_path)
    counts = log["counts"]
    for v in all_voices:
        counts.setdefault(v, 0)

    used: set[str] = set()
    mapping: dict[str, tuple[str, str]] = {}

    for speaker in speakers:
        gender = genders.get(speaker)
        pool: list[str] = []

        if gender in pools:
            pool = [v for v in pools[gender] if v not in used]
            if not pool:
                print(
                    f"WARNING: no unused {gender} voice left for speaker '{speaker}' "
                    "— falling back to the full voice pool regardless of gender.",
                    file=sys.stderr,
                )
        elif gender is not None:
            print(
                f"WARNING: speaker '{speaker}' has unrecognised gender '{gender}' "
                "(expected 'male' or 'female') — ignoring gender for this speaker.",
                file=sys.stderr,
            )
        else:
            print(
                f"WARNING: speaker '{speaker}' has no gender declared in the transcript "
                "header — picking from the full voice pool.",
                file=sys.stderr,
            )

        if not pool:
            pool = [v for v in all_voices if v not in used] or all_voices

        voice = _pick_least_used(pool, counts)
        mapping[speaker] = _with_locale(voice)
        used.add(voice)
        counts[voice] = counts.get(voice, 0) + 1

    _save_usage_log(log, usage_log_path)
    return mapping


def assign_narrator_voice(
    gender: str | None,
    part: int,
    usage_log_path: Path = USAGE_LOG_PATH,
) -> tuple[str, str]:
    """Pick a single least-used (voice_name, language_code) for a monologue narrator.

    Filters to the declared gender's pool for this part when recognised;
    falls back to the full pool for the part with a WARNING when the gender
    is missing or unrecognised.
    """
    pools = _pools_for_part(part)
    all_voices = _all_voices_for_part(part)

    log = _load_usage_log(usage_log_path)
    counts = log["counts"]
    for v in all_voices:
        counts.setdefault(v, 0)

    pool = pools.get(gender) if gender else None
    if pool is None:
        if gender is not None:
            print(
                f"WARNING: narrator has unrecognised gender '{gender}' "
                "(expected 'male' or 'female') — picking from the full voice pool.",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: narrator has no gender declared in the transcript header "
                "— picking from the full voice pool.",
                file=sys.stderr,
            )
        pool = all_voices

    voice = _pick_least_used(pool, counts)
    counts[voice] = counts.get(voice, 0) + 1
    _save_usage_log(log, usage_log_path)
    return _with_locale(voice)
