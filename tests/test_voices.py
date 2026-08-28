import pytest

from src.voices import (
    ACCENT_PARTS,
    ACCENT_VOICE_POOLS,
    UK_VOICE_POOLS,
    VOICE_LOCALE,
    assign_dialogue_voices,
    assign_narrator_voice,
)


@pytest.fixture
def usage_log_path(tmp_path):
    return tmp_path / "voice_usage_log.json"


def _all_uk_voices():
    return [v for voices in UK_VOICE_POOLS.values() for v in voices]


def _all_accent_voices():
    return [
        v
        for pools in ACCENT_VOICE_POOLS.values()
        for voices in pools.values()
        for v in voices
    ]


# --- Parts 1/2: en-GB only -------------------------------------------------


def test_dialogue_voices_respect_declared_gender(usage_log_path):
    speakers = ["Examiner", "Candidate"]
    genders = {"Examiner": "male", "Candidate": "female"}
    mapping = assign_dialogue_voices(speakers, genders, part=1, usage_log_path=usage_log_path)

    examiner_voice, examiner_lang = mapping["Examiner"]
    candidate_voice, candidate_lang = mapping["Candidate"]
    assert examiner_voice in UK_VOICE_POOLS["male"]
    assert candidate_voice in UK_VOICE_POOLS["female"]
    assert examiner_lang == "en-GB"
    assert candidate_lang == "en-GB"


def test_part_1_and_2_never_use_non_uk_accents(usage_log_path):
    speakers = ["A", "B"]
    genders = {"A": "male", "B": "female"}
    for part in (1, 2):
        mapping = assign_dialogue_voices(speakers, genders, part=part, usage_log_path=usage_log_path)
        for voice_name, language_code in mapping.values():
            assert voice_name in _all_uk_voices()
            assert language_code == "en-GB"


def test_dialogue_voices_are_distinct_per_speaker(usage_log_path):
    speakers = ["A", "B", "C"]
    genders = {"A": "male", "B": "male", "C": "male"}
    mapping = assign_dialogue_voices(speakers, genders, part=1, usage_log_path=usage_log_path)

    voice_names = [v for v, _ in mapping.values()]
    assert len(set(voice_names)) == 3


def test_dialogue_voices_fallback_to_full_pool_when_gender_missing(usage_log_path, capsys):
    mapping = assign_dialogue_voices(["Unknown"], {}, part=2, usage_log_path=usage_log_path)

    voice_name, _ = mapping["Unknown"]
    assert voice_name in _all_uk_voices()
    captured = capsys.readouterr()
    assert "no gender declared" in captured.err


def test_dialogue_voices_fallback_to_full_pool_when_unrecognised_gender(usage_log_path, capsys):
    mapping = assign_dialogue_voices(
        ["Speaker"], {"Speaker": "unspecified"}, part=2, usage_log_path=usage_log_path
    )

    voice_name, _ = mapping["Speaker"]
    assert voice_name in _all_uk_voices()
    captured = capsys.readouterr()
    assert "unrecognised gender" in captured.err


def test_dialogue_voices_warn_and_fallback_when_gender_pool_exhausted(usage_log_path, capsys):
    # Part 1/2 has only 2 male UK voices; a 3rd male speaker must fall back
    # to the full pool with a warning rather than crashing.
    speakers = ["A", "B", "C"]
    genders = {"A": "male", "B": "male", "C": "male"}
    mapping = assign_dialogue_voices(speakers, genders, part=1, usage_log_path=usage_log_path)

    voice_names = [v for v, _ in mapping.values()]
    assert len(set(voice_names)) == 3
    captured = capsys.readouterr()
    assert "no unused male voice left" in captured.err


def test_dialogue_voices_rotate_least_used_across_calls(usage_log_path):
    assign_dialogue_voices(
        ["A", "B"], {"A": "male", "B": "male"}, part=1, usage_log_path=usage_log_path
    )
    mapping = assign_dialogue_voices(["C"], {"C": "male"}, part=1, usage_log_path=usage_log_path)
    voice_name, _ = mapping["C"]
    assert voice_name in UK_VOICE_POOLS["male"]


def test_narrator_voice_respects_gender(usage_log_path):
    voice_name, language_code = assign_narrator_voice("female", part=2, usage_log_path=usage_log_path)
    assert voice_name in UK_VOICE_POOLS["female"]
    assert language_code == "en-GB"


def test_narrator_voice_fallback_when_gender_none(usage_log_path, capsys):
    voice_name, _ = assign_narrator_voice(None, part=2, usage_log_path=usage_log_path)
    assert voice_name in _all_uk_voices()
    captured = capsys.readouterr()
    assert "no gender declared" in captured.err


def test_narrator_voice_rotation_picks_least_used(usage_log_path):
    first, _ = assign_narrator_voice("male", part=2, usage_log_path=usage_log_path)
    second, _ = assign_narrator_voice("male", part=2, usage_log_path=usage_log_path)
    # Only 2 male UK voices; two consecutive single-narrator picks should
    # not repeat the same voice.
    assert first != second


# --- Parts 3/4: accent variation --------------------------------------------


def test_part_3_and_4_can_use_non_uk_accents(usage_log_path):
    # With enough distinct male speakers to exhaust en-GB's 2-voice male
    # pool, later speakers should spill into other accent pools rather than
    # falling back to "any gender" — proving the wider pool is actually used.
    speakers = [f"Speaker{i}" for i in range(6)]
    genders = {s: "male" for s in speakers}
    mapping = assign_dialogue_voices(speakers, genders, part=3, usage_log_path=usage_log_path)

    voice_names = {v for v, _ in mapping.values()}
    assert len(voice_names) == 6
    non_uk_used = voice_names - set(UK_VOICE_POOLS["male"]) - set(UK_VOICE_POOLS["female"])
    assert non_uk_used, "expected at least one non-UK voice once the UK pool is exhausted"


def test_part_3_and_4_language_code_matches_chosen_voice(usage_log_path):
    speakers = [f"Speaker{i}" for i in range(6)]
    genders = {s: "male" for s in speakers}
    mapping = assign_dialogue_voices(speakers, genders, part=4, usage_log_path=usage_log_path)

    for voice_name, language_code in mapping.values():
        assert voice_name.startswith(language_code)
        assert voice_name in VOICE_LOCALE
        assert VOICE_LOCALE[voice_name] == language_code


def test_narrator_accent_pool_for_part_4(usage_log_path):
    # Exhaust en-GB's single female narrator picks across repeated calls;
    # eventually a non-UK voice should be picked as the least-used option.
    seen_voices = set()
    for _ in range(10):
        voice_name, _ = assign_narrator_voice("female", part=4, usage_log_path=usage_log_path)
        seen_voices.add(voice_name)

    non_uk_used = seen_voices - set(UK_VOICE_POOLS["female"])
    assert non_uk_used, "expected rotation to eventually reach into non-UK narrator voices"


def test_accent_parts_constant():
    assert ACCENT_PARTS == {3, 4}
