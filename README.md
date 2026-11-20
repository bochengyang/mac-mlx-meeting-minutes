# mac-mlx-meeting-minutes

Record a meeting on your Mac and get a clean, speaker-labeled transcript — entirely on your
own machine. Nothing is uploaded anywhere.

It records two things separately — **your microphone** and the **audio from the other side**
(a call, a video, a podcast…) — and writes a timestamped transcript that shows who said
what.

## What it does

- Records **you** and the **other side** as two separate tracks.
- Turns speech into text **locally** on Apple Silicon.
- Supports custom **hotwords** (`hotwords.txt`) for technical acronyms and names.
- Optionally tells apart multiple people on the other side.
- Removes junk text that speech recognition tends to invent during silent gaps.
- Converts Chinese output to Taiwan Traditional by default (`s2twp`).

In the transcript, lines are labeled **Me** (your mic), **Them** (the other side), or
**Speaker 1 / Speaker 2 / …** when you ask it to separate speakers.

## Quickstart

Core recording + transcription needs no account and no token. You need an Apple Silicon Mac
(macOS 14.4+), Xcode Command Line Tools (`xcode-select --install`), and
[uv](https://docs.astral.sh/uv/).

```bash
./scripts/build_audiotee.sh      # build the system-audio helper (~30s)
uv sync --extra transcribe       # core dependencies — no token needed
```

Grant your terminal two one-time permissions under **System Settings → Privacy & Security**:
**Screen & System Audio Recording → "System Audio Recording Only"**, and **Microphone**.

```bash
uv run python -m meeting_minutes record --transcribe
```

Put on headphones, talk, and press **Enter** to stop — your transcript lands in
`recordings/<timestamp>/minutes.md`. To label multiple remote speakers, add the optional
[diarization setup](docs/pyannote-setup.md).

## Requirements

- A Mac with Apple Silicon, macOS 14.4 or newer.
- Xcode Command Line Tools — `xcode-select --install`.
- [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

## Setup

**1. Build the audio-capture helper**

```bash
./scripts/build_audiotee.sh
```

**2. Install dependencies**

```bash
uv sync --all-extras
```

**3. Let your terminal record audio**

macOS needs two permissions for the terminal you run this from. Terminal.app asks for them
automatically; for other terminals (iTerm2, …) you may need to add them by hand.

- **System Settings → Privacy & Security → Screen & System Audio Recording → "System Audio
  Recording Only"** — turn on your terminal. This is what captures the other side.
  (The Microphone permission alone is *not* enough for this.)
- **System Settings → Privacy & Security → Microphone** — turn on your terminal.

> If the other side's recording comes out silent, this is the missing permission.

**4. (Optional) Custom hotwords dictionary**

Copy `hotwords.example.txt` to `hotwords.txt` (git-ignored so team terms stay local) and add
frequently misheard project names or English acronyms (one per line; `#` comments allowed):

```bash
cp hotwords.example.txt hotwords.txt
```

Keep the list concise (around 10–30 high-value terms) so similar-sounding acronyms don't
interfere with each other. When `hotwords.txt` exists in the working directory, it is loaded
automatically.

**5. (Optional) Separating multiple speakers**

To label several people on the other side, do the one-time setup in
[docs/pyannote-setup.md](docs/pyannote-setup.md). It still runs locally; the one-time login
is only to download the model.

## Using it

Run it from **Terminal.app**, and **wear headphones** so your microphone doesn't also pick
up the other side.

**Record a meeting:**

```bash
uv run python -m meeting_minutes record --transcribe
```

Press **Enter** to stop. It then asks how many people are on the other side:

- Press **Enter** or type `auto` → auto-detect speakers and label them `Speaker 1`, `Speaker 2`, … (needs step 5).
- Type `2` or more → separate using that exact speaker count (needs step 5).
- Type `1` or `no` → skip separation; the other side is labeled `Them`.

You get `recordings/<timestamp>/minutes.md`.

**Transcribe a recording you already made:**

```bash
uv run python -m meeting_minutes transcribe recordings/<timestamp> --diarize
```

**Handy options** *(supported by both `record --transcribe` and `transcribe`)*

| Flag | What it does |
| --- | --- |
| `--model REPO` | speech recognition model repository or local path |
| `--aligner-model REPO` | timestamp aligner model repository or local path, or `none` to skip |
| `--language LANG` | transcription language or `auto` (default: `Chinese`) |
| `--hotwords PATH` | path to hotwords file (defaults to `hotwords.txt` if present) |
| `--no-hotwords` | disable hotwords even if `hotwords.txt` exists |
| `--offline` | only load already-cached local models (no Hugging Face downloads) |
| `--title TITLE` | title for the generated `minutes.md` |
| `--diarize` | auto-detect and separate multiple other-party speakers (`transcribe` only) |
| `--num-speakers N` | exact number of other-party speakers (`transcribe` only; `1` skips separation) |
| `--device {auto,cpu,mps}` | where speaker separation runs; `auto` (default) uses the Apple GPU (MPS) when available, else CPU |
| `--zh-variant none` | turn off the Simplified → Taiwan Traditional Chinese conversion |
| `--no-vad` | keep everything, including noise from silent gaps |

> Once models are cached locally, Hugging Face Hub lookups are skipped automatically. Pass
> `--offline` to fail instead of downloading if a model isn't cached yet.

## Output

Each run creates `recordings/<timestamp>/`:

- `me.wav`, `them.wav` — the two recordings
- `minutes.md` — the transcript

Recordings and downloaded models are kept out of version control.

## Good to know

- **Headphones matter** — without them your microphone also records the other side, and the
  "Me vs. Them" split gets muddled.
- **Transcripts aren't perfect** — quiet or noisy audio can still produce odd lines, and
  people talking over each other are hard to tell apart.
- **Separating speakers uses the Apple GPU (MPS) by default** and is much faster than CPU;
  pass `--device cpu` to force CPU. Long recordings on CPU can take a long time.

## Built with

On-device speech recognition (mlx-audio), voice-activity detection (Silero), speaker
separation (pyannote.audio), Chinese conversion (OpenCC), and system-audio capture
(audiotee). Thanks to those projects.

## License

[MIT](LICENSE) © 楊博丞 (Bocheng Yang)
