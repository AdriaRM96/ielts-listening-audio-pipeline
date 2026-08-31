"""CLI entry point: generate a full IELTS Listening test's audio from transcripts.

Reads part1.txt-part4.txt from an input folder, synthesizes each with Google
Cloud Text-to-Speech, concatenates dialogue turns with a short pause between
speakers, and writes the result into a fresh, auto-numbered folder under the
output directory (output/test1/, output/test2/, ...) — audio, a copy of the
source transcript, and (if generated) the matching questions and answer key
all land in that same folder, so each testN/ is a complete, self-contained
bundle.
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
from pathlib import Path

from .parser import ParsedScript, ScriptParseError, parse_script_file
from .script_generator import ScriptGenerationError, generate_full_test
from .tts_client import TTSClientError, get_client, synthesize
from .voices import assign_dialogue_voices, assign_narrator_voice

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

PART_FILENAMES = ["part1.txt", "part2.txt", "part3.txt", "part4.txt"]
SILENCE_MS = 500
TEST_DIR_RE = re.compile(r"^test(\d+)$")


def next_test_number(output_root: Path) -> int:
    """Return the next free testN number under output_root.

    Scans existing 'testN' directories and returns max(N) + 1, rather than
    counting entries — so it's correct even when output_root already has
    non-contiguous test folders (e.g. test1 and test3 but no test2, as can
    happen after a manual cleanup) or isn't empty on a fresh clone that
    reuses a populated output directory.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    existing_numbers = []
    for entry in output_root.iterdir():
        if not entry.is_dir():
            continue
        m = TEST_DIR_RE.match(entry.name)
        if m:
            existing_numbers.append(int(m.group(1)))
    return max(existing_numbers, default=0) + 1


def _require_ffmpeg_for_dialogue() -> None:
    if not PYDUB_AVAILABLE:
        raise TTSClientError(
            "pydub is not installed but is required to build multi-speaker dialogue "
            "audio. Run: pip install -r requirements.txt"
        )
    if shutil.which("ffmpeg") is None:
        raise TTSClientError(
            "ffmpeg is required to concatenate dialogue audio but was not found on PATH.\n"
            "  macOS:         brew install ffmpeg\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  Windows:       download from https://ffmpeg.org/download.html and add it to PATH"
        )


def build_part_audio(client, script: ParsedScript, out_path: Path) -> None:
    """Synthesize one parsed part and write it as an mp3 to out_path."""
    if script.is_dialogue:
        _require_ffmpeg_for_dialogue()
        voice_map = assign_dialogue_voices(script.speakers, script.genders, script.part)

        segments = []
        for turn in script.turns:
            voice_name, language_code = voice_map[turn.speaker]
            audio_bytes = synthesize(client, turn.text, voice_name, language_code=language_code)
            segments.append(AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3"))

        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=SILENCE_MS)
        for i, segment in enumerate(segments):
            if i > 0:
                combined += silence
            combined += segment
        combined.export(out_path, format="mp3")

    else:
        narrator = script.turns[0].speaker
        gender = script.genders.get(narrator)
        voice_name, language_code = assign_narrator_voice(gender, script.part)
        full_text = " ".join(turn.text for turn in script.turns)
        audio_bytes = synthesize(client, full_text, voice_name, language_code=language_code)
        out_path.write_bytes(audio_bytes)


def run_build(input_dir: Path, output_root: Path) -> Path:
    """Build all found part files from input_dir into a new numbered test folder.

    Missing part files are skipped with a warning rather than failing the
    whole run; a malformed part file is also skipped with a warning so one
    bad transcript doesn't block the rest of the test. Each successfully
    built part's source .txt is copied alongside its .mp3 in the output
    folder, so the transcript stays with the audio it produced even after
    input_dir's contents are later overwritten by a fresh --generate run.
    """
    if not input_dir.is_dir():
        raise TTSClientError(f"Input folder not found: {input_dir}")

    test_number = next_test_number(output_root)
    test_dir = output_root / f"test{test_number}"
    test_dir.mkdir(parents=True, exist_ok=False)

    client = get_client()
    built_any = False

    for filename in PART_FILENAMES:
        file_path = input_dir / filename
        if not file_path.exists():
            print(f"  [{filename}] SKIPPED — not found in {input_dir}", file=sys.stderr)
            continue

        try:
            script = parse_script_file(file_path)
        except ScriptParseError as exc:
            print(f"  [{filename}] SKIPPED — {exc}", file=sys.stderr)
            continue

        out_path = test_dir / filename.replace(".txt", ".mp3")
        print(f"  [{filename}] synthesizing ({'dialogue' if script.is_dialogue else 'monologue'})...")
        build_part_audio(client, script, out_path)
        shutil.copy2(file_path, test_dir / filename)
        print(f"  [{filename}] -> {out_path}")
        built_any = True

    if not built_any:
        test_dir.rmdir()
        raise TTSClientError(
            f"No usable part files found in {input_dir}. "
            "Expected part1.txt, part2.txt, part3.txt, and/or part4.txt."
        )

    return test_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IELTS Listening test audio from transcript files using Google Cloud TTS."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="transcripts",
        help="Folder containing part1.txt-part4.txt (default: ./transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder to create numbered test<N> subfolders in (default: ./output)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Generate a fresh test with Gemini (Vertex AI) before synthesizing audio "
            "(writes partN.txt into input_dir, overwriting any that exist there)."
        ),
    )
    parser.add_argument(
        "--topic",
        default=None,
        metavar="TOPIC",
        help=(
            "Optional topic hint for --generate, e.g. --topic 'a university library tour'. "
            "Ignored without --generate."
        ),
    )
    parser.add_argument(
        "--section",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        metavar="N",
        help=(
            "Restrict --generate to a single section (1-4) instead of a full 4-part test. "
            "Ignored without --generate."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)

    generated: dict[int, object] = {}
    if args.generate:
        print("Generating a new test with Gemini (Vertex AI)...")
        parts = [args.section] if args.section else None
        try:
            generated = generate_full_test(args.topic, parts=parts)
        except ScriptGenerationError as exc:
            sys.exit(f"ERROR: {exc}")

        input_dir.mkdir(parents=True, exist_ok=True)
        for part_num, section in generated.items():
            path = input_dir / f"part{part_num}.txt"
            path.write_text(section.script_text, encoding="utf-8")
            print(f"  wrote {path}")

    try:
        test_dir = run_build(input_dir, Path(args.output_dir))
    except TTSClientError as exc:
        sys.exit(f"ERROR: {exc}")

    if generated:
        qa_path = test_dir / "questions_and_answers.md"
        qa_lines = []
        for part_num in sorted(generated):
            section = generated[part_num]
            qa_lines.append(f"# Part {part_num} — {section.topic_category}\n")
            qa_lines.append("## Questions\n")
            qa_lines.append(section.questions + "\n")
            qa_lines.append("## Answer key\n")
            qa_lines.append(section.answers + "\n")
        qa_path.write_text("\n".join(qa_lines), encoding="utf-8")
        print(f"  wrote {qa_path}")

    print(f"\nDone. Audio saved to: {test_dir}")


if __name__ == "__main__":
    main()
