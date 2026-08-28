from unittest.mock import MagicMock, patch

import pytest

from src.script_generator import (
    MAX_ATTEMPTS,
    SYSTEM_PROMPT,
    ScriptGenerationError,
    _parse_response,
    _recent_topics,
    _record_topic,
    _validate_cast_size,
    generate_full_test,
    generate_section,
)
from src.parser import ScriptParseError, parse_script


VALID_PART1_RESPONSE = """<<<TOPIC_CATEGORY>>>
booking/enquiry - car hire
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

INVALID_PART1_RESPONSE = """<<<TOPIC_CATEGORY>>>
booking/enquiry - car hire
<<<SCRIPT>>>
This line has no speaker colon prefix at all so it will fail parsing.
<<<QUESTIONS>>>
1. What does the client want to book?
<<<ANSWERS>>>
1. A car
"""

WRONG_CAST_SIZE_PART1_RESPONSE = """<<<TOPIC_CATEGORY>>>
booking/enquiry - hotel
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
    assert "<<<SCRIPT>>>" in SYSTEM_PROMPT
    assert "<<<QUESTIONS>>>" in SYSTEM_PROMPT
    assert "<<<ANSWERS>>>" in SYSTEM_PROMPT
    assert "capitalised word followed by a colon" in SYSTEM_PROMPT


# --- response parsing -----------------------------------------------------


def test_parse_response_valid():
    topic, script_text, questions, answers = _parse_response(VALID_PART1_RESPONSE, part=1)
    assert topic == "booking/enquiry - car hire"
    assert "Agent: How can I help you today?" in script_text
    assert "What does the client want to book" in questions
    assert "A car" in answers


def test_parse_response_missing_blocks_raises():
    with pytest.raises(ScriptParseError, match="four required blocks"):
        _parse_response("just some random text", part=1)


def test_parse_response_empty_questions_raises():
    text = VALID_PART1_RESPONSE.replace(
        "1. What does the client want to book?", ""
    ).replace("<<<QUESTIONS>>>\n\n", "<<<QUESTIONS>>>\n")
    with pytest.raises(ScriptParseError, match="QUESTIONS"):
        _parse_response(text, part=1)


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


# --- topic log --------------------------------------------------------


def test_recent_topics_and_record_round_trip(tmp_path):
    log_path = tmp_path / "log.json"
    assert _recent_topics(1, log_path) == []

    _record_topic(1, "booking/enquiry - car hire", log_path)
    _record_topic(1, "booking/enquiry - hotel", log_path)
    _record_topic(2, "guided tour - museum", log_path)

    assert _recent_topics(1, log_path) == ["booking/enquiry - car hire", "booking/enquiry - hotel"]
    assert _recent_topics(2, log_path) == ["guided tour - museum"]
    assert _recent_topics(3, log_path) == []


def test_recent_topics_caps_at_six(tmp_path):
    log_path = tmp_path / "log.json"
    for i in range(10):
        _record_topic(1, f"topic-{i}", log_path)
    assert _recent_topics(1, log_path) == [f"topic-{i}" for i in range(4, 10)]


# --- generate_section: success, retry, exhaustion -------------------------


def test_generate_section_success_first_try(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(VALID_PART1_RESPONSE)
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    result = generate_section(1, log_path=log_path)

    assert result.topic_category == "booking/enquiry - car hire"
    assert result.script.part == 1
    assert result.script.speakers == ["Agent", "Client"]
    fake_client.models.generate_content.assert_called_once()
    assert _recent_topics(1, log_path) == ["booking/enquiry - car hire"]


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


def test_generate_section_wraps_api_errors(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
    monkeypatch.setattr("src.script_generator._get_client", lambda: fake_client)

    with pytest.raises(ScriptGenerationError, match="quota exceeded"):
        generate_section(1, log_path=log_path)


def test_generate_section_includes_topic_hint_and_recent_topics(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    _record_topic(1, "booking/enquiry - hotel", log_path)

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
    # generate_section validates against `part`, but our fixture text is only
    # valid as a 2-speaker script — restrict this test to checking call count/keys.
    monkeypatch.setattr("src.script_generator._validate_cast_size", lambda script, part: None)

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
