# iris-listen

Speech to text on your own GPU: [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2)
behind OpenAI's `POST /v1/audio/transcriptions` and a bearer token. It is the
ears of [Iris](https://github.com/topics/iris); her voice is
[iris-voice](https://github.com/p4ulcristian/iris-voice).
[omarchy-dictation](https://github.com/p4ulcristian/omarchy-dictation)'s
push-to-talk types what it hears.

## How it fits

```
any program ── Bearer token, a clip (+ language) ──▶ iris-listen (server.py, Silero VAD + Canary, GPU)
            ◀── {"text": "..."} ────────────────────────────────
```

- **The token**: every request needs `Authorization: Bearer <token>`, or the
  answer is 401. The token is in `~/.config/iris-listen/token`, made by the
  first install. So the service is safe on any address, with or without a
  proxy in front.
- **The model**: NVIDIA's Canary-1B-v2 (fp32 ONNX through
  [onnx-asr](https://github.com/istupakov/onnx-asr)), picked for Hungarian: on
  FLEURS it has 12.9% WER in Hungarian, against 14.1% for Whisper large-v3,
  among models that run in 16 GB. It knows 25 European languages: bg cs da de
  el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk.
- **The language**: Canary cannot tell which language it hears; told the wrong
  one, it translates. A clip without a `language` is written out in each of
  the service's languages (`--languages`, `LISTEN_LANGUAGES`) and the version
  the model is surest of (highest mean token log-probability) wins. Each
  stretch of speech is picked on its own, so a clip that switches language
  between sentences keeps both.
- **Stretches**: Silero VAD cuts a clip at pauses of 250 ms or more (a stretch
  is at most 20 s); silence gives `""`. Whole clips, no stream.
- **Speed** on an RTX 5060 Ti, with two languages: about 0.3 s for a 5 s sentence.
- **Memory**: about 6 GB of GPU memory while it runs.

## Install

```bash
git clone https://github.com/p4ulcristian/iris-listen && iris-listen/install.sh
```

It makes the token, sets up `.venv` here, downloads the models (4 GB, to
`~/.local/share/iris-listen/models`, or `LISTEN_MODELS`) and
starts the user service `iris-listen` on `127.0.0.1:7861` (`LISTEN_PORT`).
`LISTEN_HOSTS="127.0.0.1 <address>"` makes it listen on more addresses, a
VPN's for a proxy in front; an address that is not up yet is bound anyway and
answers once it is. `LISTEN_LANGUAGES="hu en"` sets the languages a clip
without one may be in (default `en`). `LISTEN_TOKEN_FILE` puts the token
somewhere else.

## API

`POST /v1/audio/transcriptions`, multipart form:

| field | |
|---|---|
| `file` | the clip: WAV, FLAC, OGG, MP3, MP4/M4A or WebM (anything FFmpeg reads), any sample rate, up to 25 MB |
| `language` | optional, e.g. `en`: forces that language (speech in another one comes out translated); without it the service picks among its languages |
| `prompt` | accepted and ignored: Canary takes no context |
| `response_format` | `json` (default): `{"text": "..."}`; `text`: the text alone |
| `model` | accepted and ignored, for OpenAI's clients |

One clip at a time; others wait. `GET /health` answers `ok`.

```bash
curl -s -H "Authorization: Bearer $(cat ~/.config/iris-listen/token)" \
  -F file=@hello.wav -F language=en http://127.0.0.1:7861/v1/audio/transcriptions
```
