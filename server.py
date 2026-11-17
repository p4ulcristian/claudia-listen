"""iris-listen: speech to text on your own GPU, Canary-1B-v2 behind OpenAI's transcription API.

    POST /v1/audio/transcriptions   multipart form → {"text": ...}
        file             the clip: WAV, FLAC, OGG, MP3, MP4/M4A or WebM (what browsers record;
                         anything FFmpeg reads), any rate, mono or not
        language         optional, e.g. "en": forces that language; without it the clip is
                         written out in each of --languages and the one the model is surest of wins, sentence by sentence
        prompt           accepted and ignored: Canary takes no context
        response_format  "json" (default) or "text"
        model            accepted and ignored (OpenAI's clients send it)
    GET  /health                    "ok"

Every request needs `Authorization: Bearer <token>`, the token in --token-file
(install.sh makes one); without it the answer is 401. So it is safe on any
address, behind a proxy or not.

Run it with this folder's venv:  .venv/bin/python server.py --token-file <file> ...
It listens on 127.0.0.1 by default. Each --host adds an address; one that is
not up yet (a VPN's) is bound anyway and starts answering when it comes up.

Canary has no language detection: told a language, it writes in that
language, and told the wrong one it translates. So a forced language must be
the one spoken. Silero VAD cuts each clip into stretches of speech at its
pauses; silence gives "". One clip at a time; others wait.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import secrets
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger("iris-listen")

RATE = 16000
MAX_BYTES = 25 * 1024 * 1024   # OpenAI's limit too
# The 25 languages Canary-1B-v2 is trained on (its model card). Its vocabulary
# has a token for every ISO code, but it only knows these.
LANGUAGES = frozenset("bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split())


# Where a clip is cut into stretches of speech (Silero VAD): at a pause of at
# least 250 ms, so a stretch is a sentence or less and its language is picked
# on its own (with 500 ms, a switch of language after a 0.3 s pause lost the
# second language's sentences); a stretch over 20 s is cut anyway (Canary is trained on
# clips up to 40 s). Each stretch keeps 0.1 s of the quiet around it, so no
# word loses its first or last sound.
CUTS = {"min_silence_duration_ms": 250, "max_speech_duration_s": 20, "speech_pad_ms": 100}

# How many stretches go through the model at once. The GPU is shared (OmniVoice, games), and a
# long clip's eight 20 s stretches at once once asked for more memory than was left. Two at a time;
# when even that finds no memory, one at a time. Never on the CPU (the person's rule): with no
# memory even for one stretch the clip is refused (503) and the caller keeps it to try again.
BATCH = 2


class GpuFull(Exception):
    """The GPU has no memory left even for one stretch: the clip is refused, to be sent again."""


def out_of_memory(e: Exception) -> bool:
    """Whether ONNX Runtime failed because the GPU had no memory left for it."""
    return "Failed to allocate memory" in str(e)


class Speech:
    """Canary-1B-v2 on the GPU, through onnx-asr (fp32 ONNX), behind Silero VAD.

    Canary has no language detection, so with several `languages` each
    stretch of speech is written out in every one of them and the version
    with the highest mean token log-probability wins: the language spoken
    scores about -0.01 to -0.06, a translation about -0.15 to -0.3. Picking
    per stretch keeps both languages of a clip that switches between them.
    Silence makes no stretch, so it comes back as "". Every model call runs
    on the one model thread (`worker`).
    """

    def __init__(self, path: Path, vad_path: Path, languages: list[str]) -> None:
        import onnxruntime as ort

        ort.preload_dlls()   # CUDA and cuDNN come from pip wheels; load them before the first session
        self.path, self.vad_path, self.languages = path, vad_path, languages
        self.load()

    def load(self) -> None:
        """The model's sessions, made new: their GPU memory pool starts empty."""
        import onnx_asr

        self.vad = onnx_asr.load_vad("silero", self.vad_path, providers=["CPUExecutionProvider"])
        self.plain = onnx_asr.load_model(
            "nemo-canary-1b-v2", path=self.path,
            # Grow the memory pool only by what is asked for, not in doubling steps.
            providers=[("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
                       "CPUExecutionProvider"],
        )
        self.model = self.plain.with_vad(self.vad, **CUTS, batch_size=BATCH).with_timestamps()
        self.single = self.plain.with_vad(self.vad, **CUTS, batch_size=1).with_timestamps()

    def warmup(self) -> None:
        """Allocate the GPU memory now, so the first clip doesn't wait for it.
        Past the VAD: it would find no speech in the noise and skip the model."""
        noise = np.random.default_rng(0).normal(0, 0.01, RATE * 5).astype(np.float32)
        self.plain.recognize(noise, sample_rate=RATE, language=self.languages[0])

    def recognize(self, samples: np.ndarray, language: str | None) -> str:
        """16 kHz audio, in `language`, or else each stretch in the one of
        `languages` the model is surest of. Out of GPU memory: again one
        stretch at a time (BATCH); still out: the model is loaded anew and
        tries once more (onnxruntime's memory pool only grows: every new
        largest buffer stays in it, so after hours it filled the card,
        11.2 GB on 2026-11-17; new sessions start with an empty pool);
        still out: GpuFull."""
        try:
            return self.written(self.model, samples, language)
        except Exception as e:   # onnxruntime's Fail carries no type of its own
            if not out_of_memory(e):
                raise
            log.warning("out of GPU memory: again one stretch at a time")
        try:
            return self.written(self.single, samples, language)
        except Exception as e:
            if not out_of_memory(e):
                raise
            log.warning("out of GPU memory one stretch at a time: loading the model anew")
        self.fresh()
        try:
            return self.written(self.model, samples, language)
        except Exception as e:
            if out_of_memory(e):
                raise GpuFull() from e
            raise

    def fresh(self) -> None:
        """Drops the sessions (and their memory pools), then loads them again."""
        import gc

        self.model = self.single = self.plain = self.vad = None
        gc.collect()
        self.load()
        self.warmup()

    def written(self, model, samples: np.ndarray, language: str | None) -> str:
        """The text of 16 kHz audio through one model (see recognize)."""
        langs = [language] if language else self.languages
        versions = [list(model.recognize(samples, sample_rate=RATE, language=lang)) for lang in langs]
        texts = []
        for stretch in zip(*versions, strict=True):
            scores = [float(np.mean(s.logprobs)) if s.logprobs else -np.inf for s in stretch]
            best = int(np.argmax(scores))
            log.debug("%.1f-%.1fs: %s", stretch[0].start, stretch[0].end,
                      {lang: round(sc, 3) for lang, sc in zip(langs, scores, strict=True)})
            texts.append(stretch[best].text.strip())
        return " ".join(t for t in texts if t)


def decode(data: bytes) -> np.ndarray:
    """The clip as 16 kHz mono float32, from any container and codec FFmpeg reads (PyAV)."""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    chunks = []
    with av.open(io.BytesIO(data)) as clip:
        if not clip.streams.audio:
            raise ValueError("it has no sound")
        for frame in clip.decode(clip.streams.audio[0]):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


token = b""                 # from --token-file, before the first request
speech: Speech | None = None
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
bearer = HTTPBearer(auto_error=False)


def authorized(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if cred is None or not secrets.compare_digest(cred.credentials.encode(), token):
        raise HTTPException(401, "a bearer token is needed", headers={"WWW-Authenticate": "Bearer"})


# No /docs or /openapi.json: the API is in this docstring and the README.
app = FastAPI(dependencies=[Depends(authorized)], docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health():
    return PlainTextResponse("ok")


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    language: str = Form(""),
    prompt: str = Form(""),         # accepted and ignored: Canary takes no context
    response_format: str = Form("json"),
    model: str = Form(""),          # OpenAI's; accepted and ignored
):
    if response_format not in ("json", "text"):
        raise HTTPException(400, 'response_format: "json" or "text"')
    language = language.strip().lower() or None
    if language and language not in LANGUAGES:
        raise HTTPException(400, f"language: one of {', '.join(sorted(LANGUAGES))}")
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, f"the clip is over {MAX_BYTES // (1024 * 1024)} MB")
    data = await file.read()
    try:
        samples = decode(data)
    except (av.FFmpegError, ValueError) as exc:
        raise HTTPException(400, f"cannot read the clip: {exc}") from exc

    def recognize() -> str:
        t0 = time.time()
        text = speech.recognize(samples, language)
        log.info("%.1fs audio -> %.2fs: %d chars", len(samples) / RATE, time.time() - t0, len(text))
        return text

    try:
        text = await asyncio.get_running_loop().run_in_executor(worker, recognize) if len(samples) else ""
    except GpuFull as exc:
        raise HTTPException(503, "the GPU has no memory left just now: send the clip again") from exc
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})


def main() -> None:
    global speech, token
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", action="append", help="address to listen on; repeat for more (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--model", type=Path, required=True, help="the folder holding Canary-1B-v2's ONNX files")
    ap.add_argument("--vad", type=Path, required=True, help="the folder holding Silero VAD's ONNX files")
    ap.add_argument("--languages", nargs="+", required=True, metavar="CODE",
                    help="the languages a clip without one may be in, e.g. hu en")
    ap.add_argument("--token-file", required=True, type=Path,
                    help="file holding the bearer token every request must carry")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = args.token_file.expanduser().read_text().strip().encode()
    if not token:
        ap.error(f"{args.token_file} is empty")
    if unknown := sorted(set(args.languages) - LANGUAGES):
        ap.error(f"Canary does not know {', '.join(unknown)}; it knows {', '.join(sorted(LANGUAGES))}")
    log.info("languages: %s", ", ".join(args.languages))
    t0 = time.time()
    speech = worker.submit(Speech, args.model.expanduser(), args.vad.expanduser(), args.languages).result()
    worker.submit(speech.warmup).result()
    log.info("model loaded and warmed up in %.1fs", time.time() - t0)
    hosts = args.host or ["127.0.0.1"]
    log.info("ready on %s, port %d", ", ".join(hosts), args.port)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[listen(h, args.port) for h in hosts])


def listen(host: str, port: int) -> socket.socket:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # A VPN address may come up after this service; bind it now and serve it once it does.
    sock.setsockopt(socket.IPPROTO_IP, getattr(socket, "IP_FREEBIND", 15), 1)   # 15 on Linux
    sock.bind((host, port))
    return sock


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the model is a local folder; never ask the Hub
    main()
