from unittest.mock import MagicMock, patch

import pytest

from src.tts_client import TTSClientError, load_credentials_path, synthesize


def test_load_credentials_path_missing_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with patch("src.tts_client.load_dotenv"):
        with pytest.raises(TTSClientError, match="not set"):
            load_credentials_path()


def test_load_credentials_path_missing_file(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/key.json")
    with patch("src.tts_client.load_dotenv"):
        with pytest.raises(TTSClientError, match="doesn't exist"):
            load_credentials_path()


def test_load_credentials_path_valid_file(monkeypatch, tmp_path):
    key_file = tmp_path / "key.json"
    key_file.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_file))
    with patch("src.tts_client.load_dotenv"):
        assert load_credentials_path() == str(key_file)


def test_synthesize_rejects_empty_text():
    client = MagicMock()
    with pytest.raises(TTSClientError, match="empty text"):
        synthesize(client, "   ", "en-GB-Neural2-A")
    client.synthesize_speech.assert_not_called()


def test_synthesize_returns_audio_bytes():
    client = MagicMock()
    client.synthesize_speech.return_value = MagicMock(audio_content=b"fake-mp3-bytes")

    result = synthesize(client, "Hello there.", "en-GB-Neural2-A")

    assert result == b"fake-mp3-bytes"
    client.synthesize_speech.assert_called_once()


def test_synthesize_wraps_api_errors():
    client = MagicMock()
    client.synthesize_speech.side_effect = RuntimeError("quota exceeded")

    with pytest.raises(TTSClientError, match="quota exceeded"):
        synthesize(client, "Hello there.", "en-GB-Neural2-A")
