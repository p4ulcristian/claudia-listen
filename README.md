# iris-listen

English speech to finished text on your own GPU:
[Musci-ASR-qwen-1019](https://huggingface.co/Musci-research/Musci-ASR-qwen-1019-hf) hears, each
person's own word list steers it, and a small local language model
([Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it)) cleans the text up. Behind OpenAI's
`POST /v1/audio/transcriptions` and a bearer token. It is the ears of
[Iris](https://github.com/topics/iris); her voice is
[iris-voice](https://github.com/p4ulcristian/iris-voice).
[omarchy-dictation](https://github.com/p4ulcristian/omarchy-dictation)'s
push-to-talk types what it hears.

## How it fits

```
any program ── Bearer token, a clip (+ language, speaker) ──▶ iris-listen (server.py: Silero VAD + Musci, GPU)
                                                                   │  the speaker's words
                                                                   ▼
                                                              llama-server (Gemma 4 E4B, GPU, localhost)
            ◀── {"text": cleaned, "raw": heard} ──────────────
```

- **The token**: every request needs `Authorization: Bearer <token>`, or the answer is 401. The
  first install makes one (`~/.config/iris-listen/token`), and a second key that only
  iris-listen uses to talk to its cleanup server.
- **Hearing**: Musci-ASR-qwen-1019 through [transformers](https://github.com/huggingface/transformers)
  (bf16), a fine-tune of [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) for
  English. Picked for English on 2026-11-18: the best open model on Hugging Face's
  [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (5.70 %
  average WER, #5 of 69 behind four closed services; Canary-1B-v2 8.43 %, #50). On FLEURS English
  (394 clips) it made 6.32 % errors against Canary-1B-v2's 8.33 %, in 0.77 s a clip against 1.07 s.
- **English only**: the language is set in the model's prompt, so it never guesses one. A
  `language` other than `en` is refused (400); speech in another language comes out as whatever
  English the model makes of it.
- **Chunks**: Silero VAD finds the speech (a stretch ends at a pause of 250 ms or more, at most
  20 s); stretches next to each other are heard together, up to 30 s from the first one's start to
  the last one's end, so the model hears whole sentences with what came before. Silence gives
  `""`. Whole clips, no stream.
- **Word lists**: one per speaker (`GET`/`PUT /v1/vocabulary/<speaker>`). The speaker's words and
  a caller's `prompt` go into the model's system prompt as context, which Qwen3-ASR leans towards
  for names and terms. Write the words as they should be written: capitals count.
- **Cleanup**: llama-server runs Gemma 4 E4B (Q4_K_M) on localhost. It gets the raw text and the
  speaker's words, and fixes misheard words from the list, punctuation and capitals, and drops
  filler sounds (uh, um, öö); it never translates or answers. A reply that is not the transcript
  give or take a few words (the guard from omarchy-dictation's fixer), a late reply
  (`--cleanup-timeout`) or a server that is down: the raw text goes out as `text`.
- **Memory** cannot creep: the speech model takes about 3.9 GB (4.5 GB with the CUDA context),
  capped by `--gpu-fraction`; torch's cache goes back to the GPU after every clip. After
  `--idle-minutes` without a clip the model is parked in RAM and comes back in a second or two.
  llama-server has a fixed context (4096 tokens, one slot): 3.7 GB, unloaded after the same idle
  time. Out of GPU memory, a clip goes again one chunk at a time; still out, the answer is 503 and
  the caller sends it again. Never on the CPU.
- **Restarts lose no clip**: a new copy loads its model first, then binds the port beside the
  running one (`SO_REUSEPORT`, same user); both answer until the old one stops. A port held by a
  process without `SO_REUSEPORT` is waited for and taken the moment it is free.
- **Speed** on an RTX 5090: hearing about 0.8 s for a 10 s clip, 3.6 s for 65 s; cleanup about
  0.1-0.8 s.

## Install

```bash
git clone https://github.com/p4ulcristian/iris-listen && iris-listen/install.sh
```

It makes the token, sets up `.venv` here, downloads llama.cpp's CUDA build and the models (about
9 GB, to `~/.local/share/iris-listen/models`, or `LISTEN_MODELS`) and starts two user services:
`iris-listen` on `127.0.0.1:7861` (`LISTEN_PORT`) and `iris-listen-cleanup` (llama-server) on
`127.0.0.1:7961` (`LISTEN_CLEANUP_PORT`, default the port + 100).

| variable | |
|---|---|
| `LISTEN_HOSTS` | `"127.0.0.1 <address>"`: more addresses, a VPN's for a proxy in front; one that is not up yet is bound anyway and answers once it is |
| `LISTEN_DATA` | where the word lists are kept (default `~/.local/share/<unit>`) |
| `LISTEN_GPU_FRACTION` | the most of the GPU's memory the speech model may take (default `0.4`) |
| `LISTEN_IDLE_MINUTES` | idle time before both models give their GPU memory back (default `15`) |
| `LISTEN_TOKEN_FILE` | the token somewhere else |
| `LISTEN_UNIT` | the service's name (default `iris-listen`; the cleanup is `<unit>-cleanup`) |

Beside an older iris-listen on the same machine (port 7862, its own token and word lists):

```bash
LISTEN_UNIT=iris-listen-v2 LISTEN_PORT=7862 \
  LISTEN_HOSTS="127.0.0.1 <vpn address>" ./install.sh
```

## API

`POST /v1/audio/transcriptions`, multipart form → `{"text": cleaned, "raw": heard}`:

| field | |
|---|---|
| `file` | the clip: WAV, FLAC, OGG, MP3, MP4/M4A or WebM (anything FFmpeg reads), any sample rate, up to 25 MB |
| `language` | optional: `en` or none; any other is refused (400): English only |
| `speaker` | optional: whose word list steers hearing and cleanup, any string (e.g. a sign-in email); none = no list |
| `cleanup` | optional, `false` skips the cleanup: then `text` is `raw` |
| `prompt` | optional: names and terms, comma separated, the model leans towards (with the speaker's word list) |
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
