"""Thin wrapper around the Google Cloud Text-to-Speech client."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class TTSClientError(Exception):
    """Raised for credential or API-call failures talking to Google Cloud TTS."""


def load_credentials_path() -> str:
    """Read GOOGLE_APPLICATION_CREDENTIALS from .env and validate it points to a file.

    The google-cloud-texttospeech client reads this same env var itself when
    constructing the client, so this function's job is purely to fail with a
    clear message *before* that happens, rather than surfacing whatever
    generic auth error the SDK produces.
    """
    load_dotenv()
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not path:
        raise TTSClientError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. Copy .env.example to .env "
            "and set it to the path of your downloaded service-account JSON key."
        )
    if not Path(path).is_file():
        raise TTSClientError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to a file that doesn't exist: {path}"
        )
    return path


def get_client():
    """Construct and return an authenticated TextToSpeechClient."""
    load_credentials_path()
    try:
        from google.cloud import texttospeech
    except ImportError as exc:
        raise TTSClientError(
            "google-cloud-texttospeech is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        return texttospeech.TextToSpeechClient()
    except Exception as exc:  # noqa: BLE001 - surface any auth/client-construction failure clearly
        raise TTSClientError(f"Could not create a Google Cloud TTS client: {exc}") from exc


def synthesize(
    client,
    text: str,
    voice_name: str,
    language_code: str = "en-GB",
    speaking_rate: float = 1.0,
) -> bytes:
    """Synthesize a single block of text to MP3 bytes using the given voice."""
    if not text or not text.strip():
        raise TTSClientError("Cannot synthesize empty text.")

    from google.cloud import texttospeech

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
    )

    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
    except Exception as exc:  # noqa: BLE001 - wrap any SDK/network error with context
        raise TTSClientError(
            f"Google Cloud TTS request failed for voice '{voice_name}': {exc}"
        ) from exc

    return response.audio_content
