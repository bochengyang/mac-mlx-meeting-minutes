# Speaker diarization setup (pyannote)

This tool can label the remote speakers (`Speaker 1`, `Speaker 2`, …) in a recording using
[pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization.

**Privacy:** the model runs **100% locally** — your audio never leaves your machine. A
**one-time** Hugging Face token is required only to *download* the gated model weights; it
is not used during inference. After the first download you can run fully offline.

## Prerequisites

- A free [Hugging Face](https://huggingface.co/join) account.

## One-time setup

1. **Accept the model license.** Sign in to Hugging Face and open
   <https://huggingface.co/pyannote/speaker-diarization-community-1>, then click
   **"Agree and access repository"**.

2. **Create an access token.** Go to <https://huggingface.co/settings/tokens> and create a
   token with **Read** access. Copy it.

3. **Log in locally.** This caches the token under `~/.cache/huggingface`; the tool reads
   the cache and never stores the token itself:

   ```bash
   uv run hf auth login
   # paste your token when prompted
   # (older Hugging Face CLI: uv run huggingface-cli login)
   ```

## Install the diarization dependency

```bash
uv sync --extra diarize
```

## How it is used

When you stop a recording with **Enter**, the tool asks for the number of remote
participants:

- **Enter a number (2 or more)** → diarization runs and the remote speech is labeled
  `Speaker 1 / Speaker 2 / …`. Providing the known participant count is the most reliable mode.
- **Leave it blank, or enter 1** → diarization is skipped; remote speech is labeled simply
  `Them`.

## Verifying it stays local

After the one-time model download, inference makes no network calls. You can confirm with a
firewall (LuLu / Little Snitch) or by disconnecting from the network and running a
transcription with diarization enabled.
