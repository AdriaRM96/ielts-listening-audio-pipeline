# IELTS Listening Audio Pipeline

[![Tests](https://github.com/AdriaRM96/ielts-listening-audio-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/AdriaRM96/ielts-listening-audio-pipeline/actions/workflows/tests.yml)

Generates complete IELTS Listening practice tests — transcript, matching questions and answer key, and natural-sounding .mp3 audio — using Gemini and [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech). Distinct voices are assigned automatically by speaker and gender: British English throughout Parts 1 and 2, and a mix of British, Australian, Indian, and American accents in Parts 3 and 4, matching how the real exam sometimes varies accents in its harder sections.

New to Google Cloud or not comfortable with the command line? Follow the full walkthrough in [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

---

## Listen to a real example

**🎧 [Listen to a full 4-part test on the demo page](https://adriarm96.github.io/ielts-listening-audio-pipeline/)** — generated end-to-end by this tool, playable right in your browser, no setup or cloning needed.

Prefer to grab the raw files instead? [Questions and answer key](docs/sample-test/questions_and_answers.md), or download any `.mp3` directly: [part1](docs/sample-test/part1.mp3) · [part2](docs/sample-test/part2.mp3) · [part3](docs/sample-test/part3.mp3) · [part4](docs/sample-test/part4.mp3) (with matching [transcripts](docs/sample-test/)).

---

## Quick start

Assumes you already have Python and a Google Cloud project set up.

```bash
git clone https://github.com/AdriaRM96/ielts-listening-audio-pipeline.git
cd ielts-listening-audio-pipeline
pip install -r requirements.txt
```

Requires a GCP project with the Text-to-Speech API enabled and a service-account key (for `--generate`/`--quiz`, also grant that service account the **Vertex AI User** role). Point the tool at your key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
```

Then either bring your own transcripts (`part1.txt`-`part4.txt`, `# GENDER: Name=male/female` header + `Speaker: line` turns, dropped into `transcripts/`) or generate everything from scratch:

```bash
python run.py                                    # bring your own transcripts
python run.py --generate                         # Gemini writes the test, then synthesizes it
python run.py --generate --topic "a museum tour"  # optional topic nudge
python run.py --quiz                              # generate a matching quiz for a transcript you already have
```

Output lands in `output/testN/` — audio, transcript, and (when generated) `questions_and_answers.md`, bundled together, auto-numbered so nothing gets overwritten. Full console click-through for the GCP setup, ffmpeg install, and a troubleshooting table are in [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

---

## Project structure

```
.
├── run.py                # entry point: python run.py
├── src/
│   ├── parser.py            # reads and validates transcript files
│   ├── voices.py            # gender-aware voice pool and assignment
│   ├── tts_client.py        # Google Cloud Text-to-Speech wrapper
│   ├── script_generator.py  # optional: writes scripts with Gemini (Vertex AI)
│   └── build_test.py        # orchestrates a full test build
├── tests/                 # automated tests (pytest) — no GCP account needed
├── notebooks/
│   └── demo.ipynb         # walkthrough: parsing → voice assignment → synthesis, with playable audio
├── transcripts/           # your working input — part1.txt-part4.txt (or let --generate write them)
├── output/                # each testN/ bundles that run's partN.mp3 + partN.txt (+ questions_and_answers.md if generated)
├── requirements.txt
└── .env.example
```

## Running the tests

The automated tests never call the real Google Cloud API — the network-facing parts are mocked, so they run instantly and don't cost anything:

```bash
pip install -r requirements-dev.txt
pytest
```

## Demo notebook

`notebooks/demo.ipynb` walks through what `run.py` does step by step — parsing a transcript, assigning voices, and calling Google Cloud Text-to-Speech — with the real audio playable inline. It requires Jupyter (`pip install jupyter`) and a working `.env` (see [docs/USER_GUIDE.md](docs/USER_GUIDE.md)), since the synthesis cells call your real Google Cloud project.
