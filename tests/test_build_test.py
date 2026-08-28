from unittest.mock import MagicMock, patch

import pytest

from src.build_test import next_test_number, run_build
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

    test_dir = run_build(input_dir, output_dir)

    assert test_dir == output_dir / "test1"
    assert (test_dir / "part2.mp3").read_bytes() == b"fake-mp3-bytes"
    assert not (test_dir / "part1.mp3").exists()
    assert not (test_dir / "part3.mp3").exists()
    assert not (test_dir / "part4.mp3").exists()


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
