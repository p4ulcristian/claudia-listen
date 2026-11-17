# iris-listen

Speech to finished text on your own GPU: [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2)
hears, each person's own word list steers it, and a small local language model
([Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it)) cleans the text up. Behind OpenAI's
`POST /v1/audio/transcriptions` and a bearer token. It is the ears of
[Iris](https://github.com/topics/iris); her voice is
[iris-voice](https://github.com/p4ulcristian/iris-voice).
[omarchy-dictation](https://github.com/p4ulcristian/omarchy-dictation)'s
push-to-talk types what it hears.

## How it fits

```
any program ── Bearer token, a clip (+ language, speaker) ──▶ iris-listen (server.py: Silero VAD + Canary, GPU)
                                                                   │  the speaker's words
                                                                   ▼
                                                              llama-server (Gemma 4 E4B, GPU, localhost)
            ◀── {"text": cleaned, "raw": heard} ──────────────
```

- **The token**: every request needs `Authorization: Bearer <token>`, or the answer is 401. The
  first install makes one (`~/.config/iris-listen/token`), and a second key that only
  iris-listen uses to talk to its cleanup server.
- **Hearing**: Canary-1B-v2 through [NeMo](https://github.com/NVIDIA-NeMo/NeMo) (PyTorch, bf16),
  picked for Hungarian (FLEURS: 13.1 % WER, Whisper large-v3 13.4 %). It knows 25 European
  languages: bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk.
- **The language**: Canary cannot tell which language it hears; told the wrong one, it
  translates. Each stretch of speech of a clip without a `language` is written out in each of the
  service's languages (`--languages`, `LISTEN_LANGUAGES`) and the version the model is surest of
  (highest score per token) wins, so a clip that switches language between sentences keeps both.
- **Stretches**: Silero VAD cuts a clip at pauses of 250 ms or more (a stretch is at most 20 s);
  silence gives `""`. Whole clips, no stream.
- **Word lists**: one per speaker (`GET`/`PUT /v1/vocabulary/<speaker>`). NeMo's GPU phrase
  boosting adds the speaker's words to Canary's beam search: each list becomes a boosting tree once
  (about 0.1 s for 1000 words) and is kept, and a clip's speaker picks its tree, so switching
  speakers costs nothing. Write the words as they should be written: capitals count.
- **Cleanup**: llama-server runs Gemma 4 E4B (Q4_K_M) on localhost. It gets the raw text and the
  speaker's words, and fixes misheard words from the list, punctuation and capitals, and drops
  filler sounds (uh, um, öö); it never translates or answers. A reply that is not the transcript
  give or take a few words (the guard from omarchy-dictation's fixer), a late reply
  (`--cleanup-timeout`) or a server that is down: the raw text goes out as `text`.
- **Memory** cannot creep: the speech model takes about 1.9 GB (3.4 GB with the CUDA context),
  capped by `--gpu-fraction`; torch's cache goes back to the GPU after every clip. After
  `--idle-minutes` without a clip the model is parked in RAM (the process keeps 1.5 GB, its CUDA
  context) and comes back in about 0.3 s. llama-server has a fixed context (4096 tokens, one
  slot): 3.7 GB, unloaded after the same idle time. Out of GPU memory, a clip goes again one
  stretch at a time; still out, the answer is 503 and the caller sends it again. Never on the CPU.
- **Speed** on an RTX 5090, two languages, beam 4: hearing about 0.6-1.6 s for a 6-18 s clip,
  cleanup about 0.1-0.4 s.

## Install

```bash
git clone https://github.com/p4ulcristian/iris-listen && iris-listen/install.sh
```

It makes the token, sets up `.venv` here, downloads llama.cpp's CUDA build and the models (about
11 GB, to `~/.local/share/iris-listen/models`, or `LISTEN_MODELS`) and starts two user services:
`iris-listen` on `127.0.0.1:7861` (`LISTEN_PORT`) and `iris-listen-cleanup` (llama-server) on
`127.0.0.1:7961` (`LISTEN_CLEANUP_PORT`, default the port + 100).

| variable | |
|---|---|
| `LISTEN_HOSTS` | `"127.0.0.1 <address>"`: more addresses, a VPN's for a proxy in front; one that is not up yet is bound anyway and answers once it is |
| `LISTEN_LANGUAGES` | the languages a clip without one may be in (default `en`), e.g. `"hu en"` |
| `LISTEN_DATA` | where the word lists are kept (default `~/.local/share/<unit>`) |
| `LISTEN_GPU_FRACTION` | the most of the GPU's memory the speech model may take (default `0.4`) |
| `LISTEN_IDLE_MINUTES` | idle time before both models give their GPU memory back (default `15`) |
| `LISTEN_TOKEN_FILE` | the token somewhere else |
| `LISTEN_UNIT` | the service's name (default `iris-listen`; the cleanup is `<unit>-cleanup`) |

Beside an older iris-listen on the same machine (port 7862, its own token and word lists):

```bash
LISTEN_UNIT=iris-listen-v2 LISTEN_PORT=7862 LISTEN_LANGUAGES="hu en" \
  LISTEN_HOSTS="127.0.0.1 <vpn address>" ./install.sh
```

## API

`POST /v1/audio/transcriptions`, multipart form → `{"text": cleaned, "raw": heard}`:

| field | |
|---|---|
| `file` | the clip: WAV, FLAC, OGG, MP3, MP4/M4A or WebM (anything FFmpeg reads), any sample rate, up to 25 MB |
| `language` | optional, e.g. `en`: forces that language (speech in another one comes out translated); without it the service picks among its languages |
| `speaker` | optional: whose word list steers hearing and cleanup, any string (e.g. a sign-in email); none = no list |
| `cleanup` | optional, `false` skips the cleanup: then `text` is `raw` |
| `prompt` | accepted and ignored: Canary takes no context |
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
T=$(cat ~/.config/iris-listen/token)
curl -s -H "Authorization: Bearer $T" -X PUT -H 'content-type: application/json' \
  -d '{"words": ["WireGuard", "Postgres"]}' http://127.0.0.1:7861/v1/vocabulary/someone
curl -s -H "Authorization: Bearer $T" -F file=@hello.wav -F speaker=someone \
  http://127.0.0.1:7861/v1/audio/transcriptions
```
