import pytest

from src.parser import (
    ScriptParseError,
    part_number_from_filename,
    parse_script,
)


def test_part_number_from_filename_valid():
    assert part_number_from_filename("transcripts/part1.txt") == 1
    assert part_number_from_filename("part4.txt") == 4


def test_part_number_from_filename_invalid():
    with pytest.raises(ScriptParseError):
        part_number_from_filename("section1.txt")
    with pytest.raises(ScriptParseError):
        part_number_from_filename("part5.txt")


def test_parse_dialogue_with_gender_header():
    text = """# GENDER: Examiner=male
# GENDER: Candidate=female

Examiner: Good morning. Can you tell me your name, please?
Candidate: Yes, my name is Sarah Thompson.
"""
    script = parse_script(text, part=1)
    assert script.genders == {"Examiner": "male", "Candidate": "female"}
    assert script.speakers == ["Examiner", "Candidate"]
    assert script.is_dialogue is True
    assert script.turns[0].text == "Good morning. Can you tell me your name, please?"
    assert script.turns[0].gender == "male"
    assert script.turns[1].gender == "female"


def test_parse_monologue_without_gender_header():
    text = "Narrator: Welcome to Part 2. In this section you will hear a talk.\n"
    script = parse_script(text, part=2)
    assert script.genders == {}
    assert script.speakers == ["Narrator"]
    assert script.is_dialogue is False
    assert script.turns[0].gender is None


def test_parse_empty_file_raises():
    with pytest.raises(ScriptParseError):
        parse_script("", part=1)
    with pytest.raises(ScriptParseError):
        parse_script("   \n\n  ", part=1)


def test_parse_header_only_no_turns_raises():
    text = "# GENDER: Narrator=female\n"
    with pytest.raises(ScriptParseError):
        parse_script(text, part=2)


def test_parse_malformed_line_raises():
    text = """# GENDER: Narrator=female
This line has no speaker colon prefix at all.
"""
    with pytest.raises(ScriptParseError):
        parse_script(text, part=2)


def test_parse_ignores_unrecognised_gender_value_in_header():
    # A header line with an invalid gender value is simply not captured —
    # the speaker falls through to "no gender declared" downstream.
    text = """# GENDER: Narrator=unknown
Narrator: Hello there.
"""
    script = parse_script(text, part=2)
    assert script.genders == {}


def test_parse_gender_header_must_precede_turns():
    # A '# GENDER:' line appearing after spoken turns have started is not
    # part of the header block and is rejected as a malformed turn line.
    text = """Narrator: Hello there.
# GENDER: Narrator=female
"""
    with pytest.raises(ScriptParseError):
        parse_script(text, part=2)


def test_speakers_preserves_order_of_first_appearance():
    text = """Examiner: Line one.
Candidate: Line two.
Examiner: Line three.
Candidate: Line four.
"""
    script = parse_script(text, part=3)
    assert script.speakers == ["Examiner", "Candidate"]
