# IELTS Listening Audio Pipeline

[![Tests](https://github.com/AdriaRM96/ielts-listening-audio-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/AdriaRM96/ielts-listening-audio-pipeline/actions/workflows/tests.yml)

Turns IELTS Listening practice transcripts into natural-sounding .mp3 audio using [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech), with distinct voices assigned automatically by speaker and gender — British English throughout Parts 1 and 2, and a mix of British, Australian, Indian, and American accents in Parts 3 and 4, matching how the real exam sometimes varies accents in its harder sections.

You bring the transcripts (four files, `part1.txt` to `part4.txt`), this tool does the rest: it reads each one, picks appropriate voices, and writes out a full set of practice audio files.

---

## What you need before you start

- A computer with Python 3.10 or newer installed
- A free Google account (a normal Gmail account works)
- About 10 minutes for one-time setup

You do **not** need to know how to code. Every step below is copy-paste.

---

## Step 1 — Download this project

Click the green **Code** button at the top of this page → **Download ZIP**, then unzip it somewhere you'll remember (e.g. your Desktop).

If you're comfortable with git instead:

```bash
git clone https://github.com/AdriaRM96/ielts-listening-audio-pipeline.git
cd ielts-listening-audio-pipeline
```

---

## Step 2 — Install the Python packages this tool needs

Open a terminal (Terminal on Mac, Command Prompt or PowerShell on Windows), navigate into the folder you just downloaded, and run:

```bash
pip install -r requirements.txt
```

### Also install ffmpeg (needed for multi-speaker dialogues)

Parts 1 and 3 of an IELTS Listening test are usually dialogues between two or more speakers — this tool stitches those speakers' audio together, which requires a small free tool called `ffmpeg`.

- **Mac**: open Terminal and run `brew install ffmpeg` ([install Homebrew first](https://brew.sh) if you don't have it)
- **Windows**: download from [ffmpeg.org](https://ffmpeg.org/download.html) and follow their "add to PATH" instructions
- **Linux**: `sudo apt install ffmpeg`

Single-speaker parts (2 and 4, usually a monologue) don't need ffmpeg at all.

---

## Step 3 — Get your own Google Cloud credentials

This tool uses your own free Google Cloud account to generate audio. Google gives new accounts free credit, and Text-to-Speech itself has a generous free monthly allowance — for occasional practice-test generation, you are very unlikely to be charged anything.

### 3.1 Create a Google Cloud project

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and sign in with your Google account.
2. Click the project dropdown at the top of the page, then **New Project**.
3. Give it any name (e.g. "IELTS Audio") and click **Create**.
4. Wait a few seconds, then select your new project from the same dropdown.

### 3.2 Enable billing (required by Google, even for free usage)

1. In the left-hand menu, go to **Billing**.
2. Follow the prompts to link a billing account (you'll need a card on file, but Text-to-Speech is very cheap and Google won't charge you without warning).
3. If you're inside your free trial credit, no charge will occur at all for this kind of usage.

### 3.3 Enable the Text-to-Speech API

1. In the search bar at the top of the console, type **Text-to-Speech API** and click on it.
2. Click the blue **Enable** button.

### 3.4 Create a service account (a "robot user" this tool authenticates as)

1. In the search bar, type **Service Accounts** and open that page.
2. Click **Create Service Account**.
3. Give it any name (e.g. `ielts-tts-runner`) and click **Create and Continue**.
4. You can skip the "grant access" and "grant users access" steps — click **Continue**, then **Done**.

### 3.5 Download your credentials key

1. On the Service Accounts page, click the service account you just created.
2. Go to the **Keys** tab.
3. Click **Add Key → Create new key**.
4. Choose **JSON** and click **Create**. A `.json` file will download automatically — **keep this file private, like a password.**
5. Move that downloaded file into this project's folder (or anywhere else on your computer you'll remember).

> If your Google account belongs to an organisation that manages Google Cloud centrally, key creation may be disabled by an admin policy. If you hit a "Key creation is not allowed" error, contact your Cloud admin, or create the project under a personal (non-managed) Google account instead.

### 3.6 Tell the tool where your key is

In this project's folder, copy `.env.example` to a new file named `.env`:

```bash
cp .env.example .env
```

Open `.env` in any text editor and set the path to the JSON file you downloaded, for example:

```
GOOGLE_APPLICATION_CREDENTIALS=/Users/yourname/Desktop/ielts-tts-runner-key.json
```

`.env` is listed in `.gitignore` and is never uploaded anywhere by this tool — it stays on your computer.

---

## Step 4 — Generate your audio

Place your four transcript files (`part1.txt`, `part2.txt`, `part3.txt`, `part4.txt`) into a folder named `transcripts/` in this project (create it if it doesn't exist). These come from the matching IELTS Listening transcript generator — the format is fixed:

```
# GENDER: Examiner=male
# GENDER: Candidate=female

Examiner: Good morning. Can you tell me your name, please?
Candidate: Yes, my name is Sarah Thompson.
```

Then run:

```bash
python run.py
```

That's it. The tool will:
- Read each transcript file it finds (missing files are skipped with a warning, not an error)
- Pick a distinct, gender-matched British voice for each speaker
- Generate the audio for each part
- Save everything into a new, automatically numbered folder: `output/test1/`, then `output/test2/` next time, and so on — so you never overwrite a previous practice test

You'll see progress printed in the terminal as each part is generated.

### Optional: point at a different transcripts folder or output location

```bash
python run.py path/to/my_transcripts --output-dir path/to/my_output
```

---

## How much does this cost?

Google Cloud Text-to-Speech includes a free monthly allowance (millions of characters, refreshed monthly) that comfortably covers casual practice-test generation. A full 4-part IELTS Listening test is typically a few thousand characters — a tiny fraction of that allowance. Check your actual usage any time at **Billing → Reports** in the Google Cloud Console.

---

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS is not set` | You haven't created `.env`, or forgot to fill in the path — see Step 3.6 |
| `GOOGLE_APPLICATION_CREDENTIALS points to a file that doesn't exist` | The path in `.env` is wrong, or the file was moved/renamed |
| `ffmpeg is required to concatenate dialogue audio but was not found on PATH` | Install ffmpeg — see Step 2 |
| A part is `SKIPPED — not found` | That `partN.txt` file isn't in your transcripts folder |
| A part is `SKIPPED — <parsing error>` | That transcript file doesn't match the expected `Speaker: text` format — check it wasn't edited by hand |
| `WARNING: no gender declared` | That speaker has no `# GENDER:` line in the transcript header — a voice is still assigned, just without gender matching |

---

## Project structure

```
.
├── run.py                # entry point: python run.py
├── src/
│   ├── parser.py          # reads and validates transcript files
│   ├── voices.py          # gender-aware voice pool and assignment
│   ├── tts_client.py       # Google Cloud Text-to-Speech wrapper
│   └── build_test.py      # orchestrates a full test build
├── tests/                 # automated tests (pytest) — no GCP account needed
├── notebooks/
│   └── demo.ipynb         # walkthrough: parsing → voice assignment → synthesis, with playable audio
├── transcripts/           # put your part1.txt-part4.txt files here
├── output/                # generated audio appears here (test1/, test2/, ...)
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

`notebooks/demo.ipynb` walks through what `run.py` does step by step — parsing a transcript, assigning voices, and calling Google Cloud Text-to-Speech — with the real audio playable inline. It requires Jupyter (`pip install jupyter`) and a working `.env` (see Step 3), since the synthesis cells call your real Google Cloud project.
