# claudia-listen

Speech to finished text on your own GPU:
[Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) hears, and a small
local language model ([Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it)) cleans the text
up with each person's own word list. Behind OpenAI's
`POST /v1/audio/transcriptions` and a bearer token. It is the dictation of
[Claudia](https://github.com/p4ulcristian/claudia), and
[omarchy-dictation](https://github.com/p4ulcristian/omarchy-dictation)'s
push-to-talk types what it hears.

## How it fits

```
any program ── Bearer token, a clip (+ speaker, context) ──▶ claudia-listen (server.py: Silero VAD + Fluid, GPU)
                                                                   │  the speaker's words
                                                                   ▼
                                                              llama-server (Gemma 4 E4B, GPU, localhost)
            ◀── {"text": cleaned, "raw": heard} ──────────────
```

- **The token**: every request needs `Authorization: Bearer <token>`, or the answer is 401. The
  first install makes one (`~/.config/claudia-listen/token`), and a second key that only
  claudia-listen uses to talk to its cleanup server.
- **Hearing**: Fluid Speech Pro through [transformers](https://github.com/huggingface/transformers)
  (`ParakeetForTDT`, bf16, greedy), a fine-tune of
  [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by FluidVoice Labs,
  CC-BY-4.0. Picked on 2026-11-21 over Musci-ASR-qwen-1019: on 150 FLEURS clips per language (no
  cleanup) it made 4.6 % errors in English against Musci's 6.8 % and 13.9 % in Hungarian against
  39.8 % (Canary-1B-v2: 13.7 % on 50 of them), with about 1.2 GB of GPU memory against 5.8 GB.
- **Languages**: it hears English, Hungarian and 23 more European languages and tells them apart
  itself (`language` is accepted and ignored). Speech mixing two languages in one sentence is not
  measured.
- **Chunks**: Silero VAD finds the speech (a stretch ends at a pause of 250 ms or more, at most
  20 s); stretches next to each other are heard together, up to 30 s from the first one's start to
  the last one's end, so the model hears whole sentences with what came before. Silence gives
  `""`. Whole clips, no stream.
- **Word lists**: one per speaker (`GET`/`PUT /v1/vocabulary/<speaker>`). The speaker's words, a
  caller's `prompt` and `context` go to the cleanup, which spells names and terms from them (the
  speech model takes no prompt). Measured 2026-11-21 on 20 made-up names: Fluid alone heard 9,
  after the cleanup with the list 20. Write the words as they should be written: capitals count.
- **Cleanup**: llama-server runs Gemma 4 E4B (Q4_K_M) on localhost with a LoRA adapter trained for this job (`train/`: 1,700 messy → clean pairs written by Claude; on 49 held-out cases 44 → 49 right, the adapter adds about 0.08 s). It gets the raw text and the
  speaker's words and returns what the speaker meant to write: it drops filler sounds (uh, um, öö),
  filler words that carry no meaning, repeats, false starts and self-corrections ("Monday, no
  wait, Tuesday" becomes "Tuesday"), fixes grammar mistakes, misheard words from the list,
  punctuation and capitals; it keeps the speaker's own words and tone, and never translates or
  answers. A caller's `context` is background for names and meaning; a name the reply brings in
  from nowhere (rare in English and Hungarian by [wordfreq](https://github.com/rspeer/wordfreq), or
  with a capital inside) is put back to the spoken words. A reply that is not the transcript
  give or take a few words (the guard from omarchy-dictation's fixer), a late reply
  (`--cleanup-timeout`) or a server that is down: the raw text goes out as `text`.
- **Memory** cannot creep: the speech model takes about 1.2 GB, capped by `--gpu-fraction`. At
  start a warm-up runs the largest clip there can be (`--batch` chunks of 30 s), and torch keeps
  the memory that took reserved for this process: another program on the GPU can't take the room a clip needs.
  Stopping the server gives it all back. After `--idle-minutes` without a clip the model is
  parked in RAM, the reserved room given back too, and both come back in a second or two.
  llama-server has a fixed context (8192 tokens, one slot): about 3.8 GB, unloaded after the same idle
  time. Out of GPU memory, a clip goes again one chunk at a time; still out, the answer is 503 and
  the caller sends it again. Never on the CPU.
- **Restarts lose no clip**: a new copy loads its model first, then binds the port beside the
  running one (`SO_REUSEPORT`, same user); both answer until the old one stops. A port held by a
  process without `SO_REUSEPORT` is waited for and taken the moment it is free.
- **Speed** on an RTX 5090: hearing about 0.15 s for a 10 s FLEURS clip; cleanup about 0.1-0.8 s.

## Install

```bash
git clone https://github.com/p4ulcristian/claudia-listen && claudia-listen/install.sh
```

It makes the token, sets up `.venv` here, downloads llama.cpp's CUDA build and the models (about
9 GB, to `~/.local/share/claudia-listen/models`, or `LISTEN_MODELS`) and starts two user services:
`claudia-listen` on `127.0.0.1:7861` (`LISTEN_PORT`) and `claudia-listen-cleanup` (llama-server) on
`127.0.0.1:7961` (`LISTEN_CLEANUP_PORT`, default the port + 100).

| variable | |
|---|---|
| `LISTEN_HOSTS` | `"127.0.0.1 <address>"`: more addresses, a VPN's for a proxy in front; one that is not up yet is bound anyway and answers once it is |
| `LISTEN_DATA` | where the word lists are kept (default `~/.local/share/<unit>`) |
| `LISTEN_GPU_FRACTION` | the most of the GPU's memory the speech model may take (default `0.4`) |
| `LISTEN_IDLE_MINUTES` | idle time before both models give their GPU memory back (default `15`) |
| `LISTEN_TOKEN_FILE` | the token somewhere else |
| `LISTEN_UNIT` | the service's name (default `claudia-listen`; the cleanup is `<unit>-cleanup`) |

Beside an older claudia-listen on the same machine (port 7862, its own token and word lists):

```bash
LISTEN_UNIT=claudia-listen-v2 LISTEN_PORT=7862 \
  LISTEN_HOSTS="127.0.0.1 <vpn address>" ./install.sh
```

## API

`POST /v1/audio/transcriptions`, multipart form → `{"text": cleaned, "raw": heard}`:

| field | |
|---|---|
| `file` | the clip: WAV, FLAC, OGG, MP3, MP4/M4A or WebM (anything FFmpeg reads), any sample rate, up to 25 MB |
| `language` | accepted and ignored: the model hears the language itself |
| `speaker` | optional: whose word list the cleanup uses, any string (e.g. a sign-in email); none = no list |
| `cleanup` | optional, `false` skips the cleanup: then `text` is `raw` |
| `prompt` | optional: names and terms, comma separated, for the cleanup (with the speaker's word list) |
| `context` | optional: text around the dictation (a chat's last messages, a document); its last 200 words help the cleanup spell the names and terms in it, as background only. The cleanup never brings in a name that is in none of transcript, word list, `prompt` or `context` |
| `response_format` | `json` (default) or `text`: the cleaned text alone |
| `model` | accepted and ignored, for OpenAI's clients |

`GET /v1/vocabulary/<speaker>` → `{"words": [...]}` (`[]` for a speaker never seen).
`PUT /v1/vocabulary/<speaker>` with `{"words": [...]}` saves the list (trimmed, blanks and repeats
dropped, at most 1000 words of at most 100 characters) and answers it as saved.
`GET /health` → `{"ok": true, "gpu": {...}, "models": {...}}`: the process's GPU memory
(allocated, reserved, peak, the process as the driver counts it, the card's free and total, the
cap) and each model's state (`gpu`, or `ram` while parked; the cleanup `up` or `down`).

One clip at a time; others wait.

```bash
T=$(cat ~/.config/claudia-listen/token)
curl -s -H "Authorization: Bearer $T" -X PUT -H 'content-type: application/json' \
  -d '{"words": ["WireGuard", "Postgres"]}' http://127.0.0.1:7861/v1/vocabulary/someone
curl -s -H "Authorization: Bearer $T" -F file=@hello.wav -F speaker=someone \
  http://127.0.0.1:7861/v1/audio/transcriptions
```
