from unittest.mock import MagicMock, patch

import pytest

from src.script_generator import (
    BALANCED_TYPE_LOOKBACK,
    DOMINANT_TYPE_LOOKBACK,
    MAX_ATTEMPTS,
    MIN_HISTORY_FOR_TYPE_NUDGE,
    QUIZ_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ScriptGenerationError,
    _expected_question_range,
    _parse_question_types,
    _parse_quiz_response,
    _parse_response,
    _recent_question_types,
    _recent_topics,
    _record_generation,
    _type_variety_guidance,
    _user_prompt,
    _validate_cast_size,
    _validate_question_count,
    generate_full_test,
    generate_quiz,
    generate_section,
)
from src.parser import ScriptParseError, parse_script


def _ten_questions(start: int = 1) -> str:
    return "\n".join(f"{n}. Question {n}?" for n in range(start, start + 10))


def _ten_answers(start: int = 1) -> str:
    return "\n".join(f"{n}. Answer {n}" for n in range(start, start + 10))


VALID_PART1_RESPONSE = f"""<<<TOPIC_CATEGORY>>>
booking/enquiry - car hire
<<<QUESTION_TYPES>>>
form completion, note completion
<<<SCRIPT>>>
# GENDER: Agent=male
# GENDER: Client=female

Agent: How can I help you today?
Client: I'd like to book a car for next week.
<<<QUESTIONS>>>
{_ten_questions(1)}
<<<ANSWERS>>>
{_ten_answers(1)}
"""

MISSING_QUESTIONS_PART1_RESPONSE = """<<<TOPIC_CATEGORY>>>
booking/enquiry - car hire
<<<QUESTION_TYPES>>>
form completion, note completion
<<<SCRIPT>>>
# GENDER: Agent=male
# GENDER: Client=female

Agent: How can I help you today?
Client: I'd like to book a car for next week.
<<<QUESTIONS>>>
1. What does the client want to book?
2. When does the client want it?
<<<ANSWERS>>>
1. A car
2. Next week
"""

INVALID_PART1_RESPONSE = """<<<TOPIC_CATEGORY>>>
booking/enquiry - car hire
<<<QUESTION_TYPES>>>
form completion
<<<SCRIPT>>>
This line has no speaker colon prefix at all so it will fail parsing.
<<<QUESTIONS>>>
1. What does the client want to book?
<<<ANSWERS>>>
1. A car
"""

WRONG_CAST_SIZE_PART1_RESPONSE = """<<<TOPIC_CATEGORY>>>
booking/enquiry - hotel
<<<QUESTION_TYPES>>>
form completion
<<<SCRIPT>>>
# GENDER: Agent=male

Agent: Hello, how can I help?
<<<QUESTIONS>>>
1. Something.
<<<ANSWERS>>>
1. Something.
"""


def _fake_response(text: str, prompt_tokens: int = 100, output_tokens: int = 200):
    response = MagicMock()
    response.text = text
    response.usage_metadata = MagicMock(
        prompt_token_count=prompt_tokens, candidates_token_count=output_tokens
    )
    return response


# --- prompt content -----------------------------------------------------


def test_system_prompt_contains_required_rule_fragments():
    assert "self-correction" in SYSTEM_PROMPT
    assert "1 to 30 characters" in SYSTEM_PROMPT
    assert "# GENDER:" in SYSTEM_PROMPT
    assert "<<<TOPIC_CATEGORY>>>" in SYSTEM_PROMPT
    assert "<<<QUESTION_TYPES>>>" in SYSTEM_PROMPT
    assert "<<<SCRIPT>>>" in SYSTEM_PROMPT
    assert "<<<QUESTIONS>>>" in SYSTEM_PROMPT
    assert "<<<ANSWERS>>>" in SYSTEM_PROMPT
    assert "capitalised word followed by a colon" in SYSTEM_PROMPT


def test_quiz_system_prompt_includes_question_types_block():
    assert "<<<QUESTION_TYPES>>>" in QUIZ_SYSTEM_PROMPT
    assert "<<<QUESTIONS>>>" in QUIZ_SYSTEM_PROMPT
    assert "<<<ANSWERS>>>" in QUIZ_SYSTEM_PROMPT
    assert "<<<SCRIPT>>>" not in QUIZ_SYSTEM_PROMPT


# --- response parsing -----------------------------------------------------


def test_parse_response_valid():
    topic, question_types, script_text, questions, answers = _parse_response(
        VALID_PART1_RESPONSE, part=1
    )
    assert topic == "booking/enquiry - car hire"
    assert question_types == ["form completion", "note completion"]
    assert "Agent: How can I help you today?" in script_text
    assert "1. Question 1?" in questions
    assert "1. Answer 1" in answers


def test_parse_response_missing_blocks_raises():
    with pytest.raises(ScriptParseError, match="five required blocks"):
        _parse_response("just some random text", part=1)


def test_parse_response_empty_questions_raises():
    text = """<<<TOPIC_CATEGORY>>>
booking/enquiry - car hire
<<<QUESTION_TYPES>>>
form completion
<<<SCRIPT>>>
# GENDER: Agent=male
# GENDER: Client=female

Agent: How can I help you today?
Client: I'd like to book a car for next week.
<<<QUESTIONS>>>
<<<ANSWERS>>>
1. A car
"""
    with pytest.raises(ScriptParseError, match="QUESTIONS"):
        _parse_response(text, part=1)


def test_parse_response_empty_question_types_raises():
    text = """<<<TOPIC_CATEGORY>>>
booking/enquiry - car hire
<<<QUESTION_TYPES>>>
<<<SCRIPT>>>
# GENDER: Agent=male
# GENDER: Client=female

Agent: How can I help you today?
Client: I'd like to book a car for next week.
<<<QUESTIONS>>>
1. What does the client want to book?
<<<ANSWERS>>>
1. A car
"""
    with pytest.raises(ScriptParseError, match="QUESTION_TYPES"):
        _parse_response(text, part=1)


def test_parse_question_types_splits_and_strips():
    assert _parse_question_types("form completion, note completion") == [
        "form completion",
        "note completion",
    ]
    assert _parse_question_types("  multiple choice ,matching  ") == [
        "multiple choice",
        "matching",
    ]
    assert _parse_question_types("") == []


def test_validate_cast_size_dialogue_ok():
    script = parse_script(
        "# GENDER: A=male\n# GENDER: B=female\nA: hi\nB: hello\n", part=1
    )
    _validate_cast_size(script, part=1)  # should not raise


def test_validate_cast_size_wrong_count_raises():
    script = parse_script("A: hi\n", part=1)
    with pytest.raises(ScriptParseError, match="exactly 2"):
        _validate_cast_size(script, part=1)


def test_validate_cast_size_monologue_ok():
    script = parse_script("Narrator: hello\n", part=2)
    _validate_cast_size(script, part=2)


def test_validate_cast_size_section3_range():
    script = parse_script("A: hi\nB: hey\nC: hello\n", part=3)
    _validate_cast_size(script, part=3)  # 3 speakers, within 2-4


# --- question count validation -----------------------------------------


def test_expected_question_range_per_part():
    assert list(_expected_question_range(1)) == list(range(1, 11))
    assert list(_expected_question_range(2)) == list(range(11, 21))
    assert list(_expected_question_range(3)) == list(range(21, 31))
    assert list(_expected_question_range(4)) == list(range(31, 41))


def test_validate_question_count_accepts_full_set():
    text = "\n".join(f"{n}. Question?" for n in range(11, 21))
    _validate_question_count(text, part=2)  # should not raise


def test_validate_question_count_rejects_gap():
    # This is the exact real-world bug: Part 2 stopped at 17 instead of 20.
    text = "\n".join(f"{n}. Question?" for n in list(range(11, 18)))
    with pytest.raises(ScriptParseError, match=r"\[18, 19, 20\]"):
        _validate_question_count(text, part=2)


def test_validate_question_count_handles_bare_number_format():
    # Real Gemini output mixes "N. text" and "N text" (no punctuation) for
    # multiple-choice stems — both must be recognised.
    text = "\n".join(f"{n} What is the answer?" for n in range(1, 11))
    _validate_question_count(text, part=1)  # should not raise


def test_validate_question_count_ignores_prose_numbers():
    # Numbers embedded in ordinary sentences (prices, counts, times) must not
    # be mistaken for question markers.
    text = "\n".join(f"{n}. Question?" for n in range(1, 11))
    text += "\nWe have over 50 classes a week, open 24 hours, with a 20% discount."
    _validate_question_count(text, part=1)  # should not raise — no false positive triggered


# --- topic + question-type log -------------------------------------------


def test_recent_topics_and_record_round_trip(tmp_path):
    log_path = tmp_path / "log.json"
    assert _recent_topics(1, log_path) == []

    _record_generation(1, "booking/enquiry - car hire", ["form completion"], log_path)
    _record_generation(1, "booking/enquiry - hotel", ["note completion"], log_path)
    _record_generation(2, "guided tour - museum", ["multiple choice"], log_path)

    assert _recent_topics(1, log_path) == ["booking/enquiry - car hire", "booking/enquiry - hotel"]
    assert _recent_topics(2, log_path) == ["guided tour - museum"]
    assert _recent_topics(3, log_path) == []


def test_recent_topics_caps_at_six(tmp_path):
    log_path = tmp_path / "log.json"
    for i in range(10):
        _record_generation(1, f"topic-{i}", ["form completion"], log_path)
    assert _recent_topics(1, log_path) == [f"topic-{i}" for i in range(4, 10)]


def test_recent_topics_skips_entries_with_no_topic(tmp_path):
    # --quiz records question_types with topic_category=None (the transcript's
    # topic isn't Gemini's to name) — those entries must not crash or pollute
    # the topic-avoidance list used by generate_section's own prompt.
    log_path = tmp_path / "log.json"
    _record_generation(1, None, ["form completion"], log_path)  # from --quiz
    _record_generation(1, "booking/enquiry - car hire", ["note completion"], log_path)

    assert _recent_topics(1, log_path) == ["booking/enquiry - car hire"]


def test_recent_question_types_round_trip(tmp_path):
    log_path = tmp_path / "log.json"
    assert _recent_question_types(1, log_path, limit=5) == []

    _record_generation(1, "topic a", ["form completion"], log_path)
    _record_generation(1, "topic b", ["form completion", "note completion"], log_path)
    _record_generation(2, "topic c", ["multiple choice"], log_path)

    assert _recent_question_types(1, log_path, limit=5) == [
        ["form completion"],
        ["form completion", "note completion"],
    ]
    assert _recent_question_types(2, log_path, limit=5) == [["multiple choice"]]
    assert _recent_question_types(3, log_path, limit=5) == []


def test_recent_question_types_respects_limit(tmp_path):
    log_path = tmp_path / "log.json"
    for i in range(10):
        _record_generation(1, f"topic-{i}", [f"type-{i}"], log_path)
    assert _recent_question_types(1, log_path, limit=3) == [
        ["type-7"], ["type-8"], ["type-9"],
    ]


# --- type variety guidance: dominant-type parts (1, 4) ---------------------


def test_type_guidance_dominant_part_no_history_is_silent(tmp_path):
    log_path = tmp_path / "log.json"
    assert _type_variety_guidance(1, log_path) == ""


def test_type_guidance_dominant_part_below_min_history_is_silent(tmp_path):
    log_path = tmp_path / "log.json"
    for _ in range(MIN_HISTORY_FOR_TYPE_NUDGE - 1):
        _record_generation(1, "topic", ["form completion"], log_path)
    assert _type_variety_guidance(1, log_path) == ""


def test_type_guidance_dominant_part_secondary_recently_seen_is_silent(tmp_path):
    log_path = tmp_path / "log.json"
    for _ in range(4):
        _record_generation(1, "topic", ["form completion"], log_path)
    _record_generation(1, "topic", ["matching"], log_path)  # secondary seen recently
    assert _type_variety_guidance(1, log_path) == ""


def test_type_guidance_dominant_part_nudges_when_secondary_absent(tmp_path):
    # This is the exact scenario in the task: 5 recent "form completion"
    # entries for Part 1, zero "matching" — should produce a soft nudge, not
    # a hard requirement.
    log_path = tmp_path / "log.json"
    for _ in range(5):
        _record_generation(1, "topic", ["form completion"], log_path)

    guidance = _type_variety_guidance(1, log_path)

    assert guidance != ""
    assert "matching" in guidance
    assert "form/note completion" in guidance
    assert "5 Part 1 sections" in guidance
    # A nudge, not a requirement — "could", not "must"/"always".
    assert "could work in a touch of" in guidance
    assert "should still be the primary type" in guidance


def test_type_guidance_dominant_part4_uses_correct_secondary(tmp_path):
    log_path = tmp_path / "log.json"
    for _ in range(4):
        _record_generation(4, "topic", ["note/summary completion"], log_path)

    guidance = _type_variety_guidance(4, log_path)

    assert "short-answer" in guidance
    assert "note/summary completion" in guidance


# --- type variety guidance: balanced-type parts (2, 3) ---------------------


def test_type_guidance_balanced_part_no_history_is_silent(tmp_path):
    log_path = tmp_path / "log.json"
    assert _type_variety_guidance(2, log_path) == ""


def test_type_guidance_balanced_part_avoids_recent_combination(tmp_path):
    log_path = tmp_path / "log.json"
    _record_generation(2, "topic", ["multiple choice", "matching"], log_path)

    guidance = _type_variety_guidance(2, log_path)

    assert "multiple choice, matching" in guidance
    assert "Avoid repeating" in guidance


def test_type_guidance_balanced_part_respects_lookback(tmp_path):
    log_path = tmp_path / "log.json"
    for i in range(5):
        _record_generation(3, "topic", [f"type-{i}"], log_path)

    guidance = _type_variety_guidance(3, log_path)

    # Only the last BALANCED_TYPE_LOOKBACK entries should be referenced.
    assert f"type-{4}" in guidance  # most recent
    assert f"type-{3}" in guidance  # second most recent
    assert "type-0" not in guidance


def test_type_guidance_unknown_part_is_silent(tmp_path):
    log_path = tmp_path / "log.json"
    assert _type_variety_guidance(5, log_path) == ""


# --- _user_prompt integration ----------------------------------------------


def test_user_prompt_part1_five_recent_form_completion_zero_matching(tmp_path):
    """Exact scenario requested for review: Part 1, 5 recent 'form completion'
    entries, 0 recent 'matching' — the prompt Gemini would actually receive."""
    log_path = tmp_path / "log.json"
    for _ in range(5):
        _record_generation(1, "some past topic", ["form completion"], log_path)

    prompt = _user_prompt(1, topic_hint=None, recent_topics=[], log_path=log_path)

    assert "could work in a touch of matching" in prompt
    assert "form/note completion" in prompt
    assert "should still be the primary type" in prompt


def test_user_prompt_includes_type_guidance_for_balanced_part(tmp_path):
    log_path = tmp_path / "log.json"
    _record_generation(2, "topic", ["multiple choice"], log_path)

    prompt = _user_prompt(2, topic_hint=None, recent_topics=[], log_path=log_path)

    assert "Avoid repeating the same combination" in prompt


# --- generate_section: success, retry, exhaustion -------------------------


def test_generate_section_success_first_try(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_PART1_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    result = generate_section(1, log_path=log_path)

    assert result.topic_category == "booking/enquiry - car hire"
    assert result.question_types == ["form completion", "note completion"]
    assert result.script.part == 1
    assert result.script.speakers == ["Agent", "Client"]
    fake_client.models.generate_content.assert_called_once()
    assert _recent_topics(1, log_path) == ["booking/enquiry - car hire"]
    assert _recent_question_types(1, log_path, limit=5) == [["form completion", "note completion"]]


def test_generate_section_retries_then_succeeds(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        _fake_response(INVALID_PART1_RESPONSE),
        _fake_response(VALID_PART1_RESPONSE),
    ]
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    result = generate_section(1, log_path=log_path)

    assert result.topic_category == "booking/enquiry - car hire"
    assert fake_client.models.generate_content.call_count == 2

    # The retry call must include the prior bad output and the parser error as context.
    second_call_contents = fake_client.models.generate_content.call_args_list[1].kwargs["contents"]
    assert INVALID_PART1_RESPONSE in second_call_contents
    assert any("didn't match the required format" in c for c in second_call_contents if isinstance(c, str))


def test_generate_section_wrong_cast_size_triggers_retry(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        _fake_response(WRONG_CAST_SIZE_PART1_RESPONSE),
        _fake_response(VALID_PART1_RESPONSE),
    ]
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    result = generate_section(1, log_path=log_path)

    assert fake_client.models.generate_content.call_count == 2
    assert result.script.speakers == ["Agent", "Client"]


def test_generate_section_exhausts_retries_raises(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(INVALID_PART1_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    with pytest.raises(ScriptGenerationError, match=f"after {MAX_ATTEMPTS} attempts"):
        generate_section(1, log_path=log_path)

    assert fake_client.models.generate_content.call_count == MAX_ATTEMPTS
    # A section that never validated must not pollute the topic-variety log.
    assert _recent_topics(1, log_path) == []


def test_generate_section_missing_questions_triggers_retry(tmp_path, monkeypatch):
    """The real bug this guards against: Gemini under-generated Part 2's quiz
    (only 7 of 10 questions, silently skipping 18-20) with no other part of
    the format looking wrong — nothing else would have caught this."""
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        _fake_response(MISSING_QUESTIONS_PART1_RESPONSE),
        _fake_response(VALID_PART1_RESPONSE),
    ]
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    result = generate_section(1, log_path=log_path)

    assert fake_client.models.generate_content.call_count == 2
    second_call_contents = fake_client.models.generate_content.call_args_list[1].kwargs["contents"]
    assert any("missing question" in c for c in second_call_contents if isinstance(c, str))
    assert "3" in str(second_call_contents)  # names the missing question numbers


def test_generate_section_wraps_api_errors(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    with pytest.raises(ScriptGenerationError, match="quota exceeded"):
        generate_section(1, log_path=log_path)


def test_generate_section_includes_topic_hint_and_recent_topics(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    _record_generation(1, "booking/enquiry - hotel", ["form completion"], log_path)

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_PART1_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    generate_section(1, topic_hint="a bicycle rental shop", log_path=log_path)

    call_contents = fake_client.models.generate_content.call_args.kwargs["contents"]
    prompt = call_contents[0]
    assert "a bicycle rental shop" in prompt
    assert "booking/enquiry - hotel" in prompt


# --- generate_full_test ----------------------------------------------------


def test_generate_full_test_defaults_to_all_four_parts(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_PART1_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)
    # generate_section validates cast size and question numbering against
    # `part`, but our fixture always returns the same fixed Part-1-shaped
    # content regardless of which part is being requested — bypass both
    # part-specific checks so this test can focus on call count/keys.
    monkeypatch.setattr("src.script_generator._validate_cast_size", lambda script, part: None)
    monkeypatch.setattr("src.script_generator._validate_question_count", lambda questions, part: None)

    result = generate_full_test(log_path=log_path)

    assert set(result.keys()) == {1, 2, 3, 4}
    assert fake_client.models.generate_content.call_count == 4


def test_generate_full_test_single_section(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_PART1_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    result = generate_full_test(log_path=log_path, parts=[1])

    assert set(result.keys()) == {1}
    fake_client.models.generate_content.assert_called_once()


# --- generate_quiz ----------------------------------------------------------


VALID_QUIZ_RESPONSE = f"""<<<QUESTION_TYPES>>>
form completion, note completion
<<<QUESTIONS>>>
{_ten_questions(1)}
<<<ANSWERS>>>
{_ten_answers(1)}
"""

MISSING_QUESTIONS_QUIZ_RESPONSE = """<<<QUESTION_TYPES>>>
form completion
<<<QUESTIONS>>>
1. What does the client want to book?
2. When does the client want it?
<<<ANSWERS>>>
1. A car
2. Next week
"""

INVALID_QUIZ_RESPONSE = "no markers here at all"


def test_parse_quiz_response_valid():
    question_types, questions, answers = _parse_quiz_response(VALID_QUIZ_RESPONSE)
    assert question_types == ["form completion", "note completion"]
    assert "1. Question 1?" in questions
    assert "1. Answer 1" in answers


def test_parse_quiz_response_missing_blocks_raises():
    with pytest.raises(ScriptParseError, match="three required blocks"):
        _parse_quiz_response(INVALID_QUIZ_RESPONSE)


def test_generate_quiz_success(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_QUIZ_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    questions, answers = generate_quiz(
        1, "Agent: How can I help?\nClient: I'd like a car.\n", log_path=log_path
    )

    assert "1. Question 1?" in questions
    assert "1. Answer 1" in answers
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert "Agent: How can I help?" in call_kwargs["contents"][0]
    assert "1-10" in call_kwargs["contents"][0]


def test_generate_quiz_records_question_types_with_no_topic(tmp_path, monkeypatch):
    # --quiz has no topic of its own to record — only the question types.
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_QUIZ_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    generate_quiz(1, "Agent: hi\nClient: hello\n", log_path=log_path)

    assert _recent_question_types(1, log_path, limit=5) == [
        ["form completion", "note completion"]
    ]
    assert _recent_topics(1, log_path) == []  # topic_category was None, correctly excluded


def test_generate_quiz_prompt_includes_type_guidance(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    for _ in range(5):
        _record_generation(1, None, ["form completion"], log_path)

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_QUIZ_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    generate_quiz(1, "Agent: hi\nClient: hello\n", log_path=log_path)

    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert "could work in a touch of matching" in call_kwargs["contents"][0]


def test_generate_quiz_retries_then_succeeds(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        _fake_response(INVALID_QUIZ_RESPONSE),
        _fake_response(VALID_QUIZ_RESPONSE),
    ]
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    questions, answers = generate_quiz(1, "Agent: hi\nClient: hello\n")

    assert fake_client.models.generate_content.call_count == 2
    assert "1. Question 1?" in questions


def test_generate_quiz_missing_questions_triggers_retry(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        _fake_response(MISSING_QUESTIONS_QUIZ_RESPONSE),
        _fake_response(VALID_QUIZ_RESPONSE),
    ]
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    questions, answers = generate_quiz(1, "Agent: hi\nClient: hello\n")

    assert fake_client.models.generate_content.call_count == 2
    second_call_contents = fake_client.models.generate_content.call_args_list[1].kwargs["contents"]
    assert any("missing question" in c for c in second_call_contents if isinstance(c, str))


def test_generate_quiz_exhausts_retries_raises(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(INVALID_QUIZ_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    with pytest.raises(ScriptGenerationError, match=f"after {MAX_ATTEMPTS} attempts"):
        generate_quiz(1, "Agent: hi\nClient: hello\n")

    assert fake_client.models.generate_content.call_count == MAX_ATTEMPTS


def test_generate_quiz_wraps_api_errors(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    with pytest.raises(ScriptGenerationError, match="quota exceeded"):
        generate_quiz(1, "Agent: hi\nClient: hello\n")
