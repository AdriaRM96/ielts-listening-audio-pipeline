from unittest.mock import MagicMock, patch

import pytest

from src.build_test import main, next_test_number, run_build
from src.script_generator import ScriptGenerationError
from src.tts_client import TTSClientError


def test_next_test_number_empty_dir(tmp_path):
    assert next_test_number(tmp_path) == 1


def test_next_test_number_creates_missing_dir(tmp_path):
    fresh = tmp_path / "does_not_exist_yet"
    assert next_test_number(fresh) == 1
    assert fresh.is_dir()


def test_next_test_number_sequential(tmp_path):
    (tmp_path / "test1").mkdir()
    (tmp_path / "test2").mkdir()
    assert next_test_number(tmp_path) == 3


def test_next_test_number_handles_gaps(tmp_path):
    # test2 was deleted manually at some point; next number must still be
    # max(existing) + 1, not len(existing) + 1 (which would collide with test3).
    (tmp_path / "test1").mkdir()
    (tmp_path / "test3").mkdir()
    assert next_test_number(tmp_path) == 4


def test_next_test_number_ignores_unrelated_dirs(tmp_path):
    (tmp_path / "test1").mkdir()
    (tmp_path / "not_a_test_dir").mkdir()
    (tmp_path / "testX").mkdir()
    assert next_test_number(tmp_path) == 2


def test_run_build_missing_input_dir(tmp_path):
    with pytest.raises(TTSClientError, match="not found"):
        run_build(tmp_path / "nope", tmp_path / "output")


def test_run_build_skips_missing_and_malformed_parts(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    (input_dir / "part2.txt").write_text("Narrator: A short monologue about museums.\n")
    (input_dir / "part3.txt").write_text("this is not valid speaker format at all\n")
    # part1.txt and part4.txt are simply absent.

    fake_client = MagicMock()
    monkeypatch.setattr("src.build_test.get_client", lambda: fake_client)
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice",
        lambda gender, part: ("en-GB-Neural2-A", "en-GB"),
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )

    test_dir, built_texts = run_build(input_dir, output_dir)

    assert test_dir == output_dir / "test1"
    assert (test_dir / "part2.mp3").read_bytes() == b"fake-mp3-bytes"
    assert not (test_dir / "part1.mp3").exists()
    assert not (test_dir / "part3.mp3").exists()
    assert not (test_dir / "part4.mp3").exists()
    assert set(built_texts.keys()) == {2}
    assert built_texts[2] == "Narrator: A short monologue about museums.\n"


def test_run_build_copies_transcript_alongside_audio(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    (input_dir / "part2.txt").write_text("Narrator: A short monologue about museums.\n")
    (input_dir / "part3.txt").write_text("this is not valid speaker format at all\n")

    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice",
        lambda gender, part: ("en-GB-Neural2-A", "en-GB"),
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )

    test_dir, _ = run_build(input_dir, output_dir)

    # A part that produced audio gets its source transcript copied alongside it...
    assert (test_dir / "part2.txt").read_text() == "Narrator: A short monologue about museums.\n"
    # ...but a part that was skipped for failing validation does not, since no
    # audio was produced for it either.
    assert not (test_dir / "part3.txt").exists()
    # The original file in input_dir is untouched (copied, not moved) so a
    # subsequent run can still read it.
    assert (input_dir / "part2.txt").exists()


def test_run_build_raises_when_nothing_usable(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())

    with pytest.raises(TTSClientError, match="No usable part files"):
        run_build(input_dir, output_dir)

    # The empty numbered test folder should be cleaned up, not left behind.
    assert not (output_dir / "test1").exists()


def test_run_build_dialogue_requires_ffmpeg(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    (input_dir / "part1.txt").write_text(
        "# GENDER: A=male\n# GENDER: B=female\nA: Hello.\nB: Hi there.\n"
    )

    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr("src.build_test.shutil.which", lambda name: None)

    with pytest.raises(TTSClientError, match="ffmpeg"):
        run_build(input_dir, output_dir)


def _fake_generated_section(part: int, script_text: str, topic: str = "a topic"):
    from src.parser import parse_script

    section = MagicMock()
    section.script = parse_script(script_text, part)
    section.script_text = script_text
    section.topic_category = topic
    section.questions = f"Q for part {part}"
    section.answers = f"A for part {part}"
    return section


def test_main_generate_writes_scripts_and_qa_before_building(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "output"

    fake_sections = {
        1: _fake_generated_section(1, "# GENDER: A=male\n\nA: Hello.\n", "car hire"),
        2: _fake_generated_section(2, "Narrator: Hi.\n", "museum tour"),
    }
    generate_mock = MagicMock(return_value=fake_sections)
    monkeypatch.setattr("src.build_test.generate_full_test", generate_mock)
    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice", lambda gender, part: ("en-GB-Neural2-A", "en-GB")
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )

    main([
        str(input_dir),
        "--generate",
        "--topic",
        "a museum tour",
        "--output-dir",
        str(output_dir),
    ])

    generate_mock.assert_called_once_with("a museum tour", parts=None)
    assert (input_dir / "part1.txt").read_text() == fake_sections[1].script_text
    assert (input_dir / "part2.txt").read_text() == fake_sections[2].script_text

    qa_path = output_dir / "test1" / "questions_and_answers.md"
    assert qa_path.exists()
    qa_text = qa_path.read_text()
    assert "car hire" in qa_text
    assert "Q for part 1" in qa_text
    assert "A for part 2" in qa_text


def test_main_generate_without_topic_passes_none(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "output"

    fake_sections = {1: _fake_generated_section(1, "Narrator: Hi.\n")}
    generate_mock = MagicMock(return_value=fake_sections)
    monkeypatch.setattr("src.build_test.generate_full_test", generate_mock)
    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice", lambda gender, part: ("en-GB-Neural2-A", "en-GB")
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )

    main([str(input_dir), "--generate", "--output-dir", str(output_dir)])

    generate_mock.assert_called_once_with(None, parts=None)


def test_main_generate_with_section_restricts_to_one_part(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "output"

    fake_sections = {3: _fake_generated_section(3, "Narrator: hi\n", "lecture topic")}
    generate_mock = MagicMock(return_value=fake_sections)
    monkeypatch.setattr("src.build_test.generate_full_test", generate_mock)
    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice", lambda gender, part: ("en-GB-Neural2-A", "en-GB")
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )

    main([str(input_dir), "--generate", "--section", "3", "--output-dir", str(output_dir)])

    generate_mock.assert_called_once_with(None, parts=[3])
    assert (input_dir / "part3.txt").exists()
    assert not (input_dir / "part1.txt").exists()


def test_main_generate_error_exits_cleanly(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"

    monkeypatch.setattr(
        "src.build_test.generate_full_test",
        MagicMock(side_effect=ScriptGenerationError("GOOGLE_APPLICATION_CREDENTIALS is not set")),
    )

    with pytest.raises(SystemExit, match="GOOGLE_APPLICATION_CREDENTIALS is not set"):
        main([str(input_dir), "--generate"])


def test_parse_args_generate_and_positional_input_dir_unambiguous():
    from src.build_test import parse_args

    args = parse_args(["my_transcripts", "--generate", "--topic", "a bakery", "--section", "2"])
    assert args.input_dir == "my_transcripts"
    assert args.generate is True
    assert args.topic == "a bakery"
    assert args.section == 2


def test_parse_args_section_defaults_to_none():
    from src.build_test import parse_args

    args = parse_args(["my_transcripts", "--generate"])
    assert args.section is None


def test_parse_args_quiz_flag():
    from src.build_test import parse_args

    args = parse_args(["my_transcripts", "--quiz"])
    assert args.quiz is True
    assert parse_args(["my_transcripts"]).quiz is False


def test_main_quiz_generates_qa_for_hand_written_transcript(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    (input_dir / "part2.txt").write_text("Narrator: A short monologue about museums.\n")

    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice", lambda gender, part: ("en-GB-Neural2-A", "en-GB")
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )
    quiz_mock = MagicMock(return_value=("Q1: What is the talk about?", "1. Museums"))
    monkeypatch.setattr("src.build_test.generate_quiz", quiz_mock)

    main([str(input_dir), "--quiz", "--output-dir", str(output_dir)])

    quiz_mock.assert_called_once_with(2, "Narrator: A short monologue about museums.\n")
    qa_text = (output_dir / "test1" / "questions_and_answers.md").read_text()
    assert "# Part 2" in qa_text
    assert "What is the talk about?" in qa_text
    assert "1. Museums" in qa_text


def test_main_quiz_skipped_for_parts_already_generated_this_run(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "output"

    fake_sections = {1: _fake_generated_section(1, "Narrator: Hi.\n", "car hire")}
    monkeypatch.setattr("src.build_test.generate_full_test", MagicMock(return_value=fake_sections))
    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice", lambda gender, part: ("en-GB-Neural2-A", "en-GB")
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )
    quiz_mock = MagicMock()
    monkeypatch.setattr("src.build_test.generate_quiz", quiz_mock)

    main([str(input_dir), "--generate", "--quiz", "--output-dir", str(output_dir)])

    # Part 1 already got a quiz as part of --generate's own call — --quiz must
    # not spend a second Gemini call regenerating it.
    quiz_mock.assert_not_called()
    qa_text = (output_dir / "test1" / "questions_and_answers.md").read_text()
    assert f"Q for part 1" in qa_text


def test_main_quiz_failure_warns_and_continues(tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    (input_dir / "part2.txt").write_text("Narrator: A short monologue.\n")

    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice", lambda gender, part: ("en-GB-Neural2-A", "en-GB")
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )
    monkeypatch.setattr(
        "src.build_test.generate_quiz",
        MagicMock(side_effect=ScriptGenerationError("Gemini request failed")),
    )

    # Must not raise — audio already succeeded, a quiz failure shouldn't lose it.
    main([str(input_dir), "--quiz", "--output-dir", str(output_dir)])

    assert (output_dir / "test1" / "part2.mp3").exists()
    assert not (output_dir / "test1" / "questions_and_answers.md").exists()
    captured = capsys.readouterr()
    assert "could not generate a quiz for part 2" in captured.err


def test_main_without_quiz_flag_no_qa_file_for_hand_written(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    (input_dir / "part2.txt").write_text("Narrator: A short monologue.\n")

    monkeypatch.setattr("src.build_test.get_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.build_test.assign_narrator_voice", lambda gender, part: ("en-GB-Neural2-A", "en-GB")
    )
    monkeypatch.setattr(
        "src.build_test.synthesize",
        lambda client, text, voice_name, language_code=None: b"fake-mp3-bytes",
    )

    main([str(input_dir), "--output-dir", str(output_dir)])

    assert not (output_dir / "test1" / "questions_and_answers.md").exists()
