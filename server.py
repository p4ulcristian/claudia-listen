"""claudia-listen: speech to finished text on your own GPU. Musci-ASR-qwen-1019 hears, each
person's own word list steers it, a small local language model cleans the text up; behind OpenAI's
transcription API.

    POST /v1/audio/transcriptions   multipart form → {"text": cleaned, "raw": heard}
        file             the clip: WAV, FLAC, OGG, MP3, MP4/M4A or WebM (what browsers record;
                         anything FFmpeg reads), any rate, mono or not
        language         optional, e.g. "hu": the language the clip is in (Qwen3-ASR's 30); none
                         means "en", and speech in another language is still written in it
        speaker          optional: whose word list steers the model and the cleanup (an opaque
                         string, e.g. a sign-in email); none = no list
        cleanup          optional, "false" skips the cleanup: then text == raw
        prompt           optional: names and terms, comma separated, the model leans towards
                         (with the speaker's word list)
        context          optional: text around the dictation (a chat's last messages, a document);
                         its last 200 words steer the model and the cleanup towards the names and
                         terms in it, as background only, never as an instruction
        response_format  "json" (default) or "text" (the cleaned text alone)
        model            accepted and ignored (OpenAI's clients send it)
    GET  /v1/vocabulary/{speaker}   → {"words": [...]}  ([] for a speaker never seen)
    PUT  /v1/vocabulary/{speaker}   {"words": [...]} → {"words": [...]} as saved
    GET  /health                    → {"ok": true, "gpu": {...}, "models": {...}}

Every request needs `Authorization: Bearer <token>`, the token in --token-file; without it the
answer is 401. So it is safe on any address, behind a proxy or not.

Run it with this folder's venv:  .venv/bin/python server.py --token-file <file> ...
It listens on 127.0.0.1 by default. Each --host adds an address; one that is not up yet (a
VPN's) is bound anyway and starts answering when it comes up. A new copy started while the old
one runs loads its model first, then shares the port with it (SO_REUSEPORT), so stopping the old
one loses no clip.

Silero VAD finds the speech in each clip; silence gives "". One clip at a time; others wait.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import gc
import hashlib
import io
import json
import logging
import os
import secrets
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import uvicorn
from fastapi import Body, Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cleanup import Cleaner

log = logging.getLogger("claudia-listen")

RATE = 16000
MAX_BYTES = 25 * 1024 * 1024   # OpenAI's limit too
MAX_WORDS = 1000           # per speaker
MAX_WORD_CHARS = 100

# Where Silero VAD finds speech: a stretch ends at a pause of at least 250 ms and is cut anyway at
# 20 s; each keeps 0.1 s of the quiet around it, so no word loses its first or last sound.
# Silero's speech chance a frame needs to start a stretch. Its default 0.5 dropped quiet iPad speech:
# 9 of 87 clips came back empty on 2026-11-17 (the old onnx-asr server: 28 of 853).
VAD_THRESHOLD = 0.3
CUTS = {"min_silence_duration_ms": 250, "max_speech_duration_s": 20, "speech_pad_ms": 100}
# Stretches next to each other are heard together, up to this much audio from the first one's start
# to the last one's end: the model hears whole sentences with what came before, and no silence.
CHUNK_SECONDS = 30
# Room for the text: English speech is about 3 words, 4 tokens a second; twice that and a margin.
TOKENS_PER_SECOND = 8
WARMUP_CONTEXT_WORDS = 300   # the model's whole context (word list, prompt, context) is cut to this
CONTEXT_WORDS = 200          # a caller's context: its newest words


class GpuFull(Exception):
    """The GPU has no memory left even for one chunk: the clip is refused, to be sent again."""


def loudest_rms(samples: np.ndarray) -> float:
    """The loudest 0.1 s of the clip, as RMS (for the log when no speech is found)."""
    n = RATE // 10
    if len(samples) < n:
        return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    return float(np.sqrt((frames ** 2).mean(axis=1)).max())


def chunked(cuts: list[dict]) -> list[tuple[int, int]]:
    """Silero's stretches of speech joined into (start, end) chunks of at most CHUNK_SECONDS."""
    chunks = []
    for c in cuts:
        if chunks and c["end"] - chunks[-1][0] <= CHUNK_SECONDS * RATE:
            chunks[-1] = (chunks[-1][0], c["end"])
        else:
            chunks.append((c["start"], c["end"]))
    return chunks


class Speech:
    """Musci-ASR-qwen-1019 (a Qwen3-ASR-1.7B fine-tune for English; transformers, bf16) on the GPU,
    behind Silero VAD (on the GPU too).

    The language goes in the prompt (English unless the caller names another). Qwen3-ASR knows 30
    languages and the fine-tune kept them: told English, it still writes Hungarian speech as
    Hungarian (seen 2026-11-18; its Hungarian error rate is not measured). A clip's context
    (the speaker's word list and the caller's prompt) is the system prompt, which Qwen3-ASR leans
    towards for names and terms. Silero finds the speech; chunks of it go through the model in batches.

    After `idle` seconds without a clip the model is parked in RAM and its GPU memory given back;
    the next clip brings it back (a second or two). Every call runs on the one model thread.
    """

    def __init__(self, path: Path, batch: int, idle: float) -> None:
        self.path, self.batch, self.idle = path, batch, idle
        self.state = "loading"
        self.last = time.monotonic()
        self.load()

    def load(self) -> None:
        import torch
        from silero_vad import load_silero_vad
        from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

        # Its defaults pad a batch on the left, so every answer starts at the same place.
        self.processor = Qwen3ASRProcessor.from_pretrained(self.path)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.vad = load_silero_vad().to("cuda")
        self.state = "gpu"

    def wake(self) -> None:
        import torch

        if self.state == "ram":
            t0 = time.time()
            self.model.to("cuda")
            self.vad.to("cuda")
            self.state = "gpu"
            log.info("model back on the GPU in %.1fs", time.time() - t0)
            self.warmup()
        self.last = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def park(self) -> None:
        """Model to RAM, all its GPU memory given back, the reserved room too (the CUDA context
        stays); waking takes the room again."""
        import torch

        if self.state != "gpu" or time.monotonic() - self.last < self.idle:
            return
        self.model.to("cpu")
        self.vad.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        self.state = "ram"
        log.info("idle: model parked in RAM, GPU memory given back")

    def warmup(self) -> None:
        """Run the model once at the largest size a clip can ask for: `batch` chunks of CHUNK_SECONDS,
        a long context, every answer as long as allowed. The GPU memory that takes stays reserved by
        torch for this process (nothing calls empty_cache after a clip), so another program on the
        GPU can't take the room a clip needs. Stopping the server gives it all back."""
        import torch

        noise = np.random.default_rng(0).normal(0, 0.01, RATE * CHUNK_SECONDS).astype(np.float32)
        context = " ".join(["Claudia"] * WARMUP_CONTEXT_WORDS)
        self.batch_written([noise] * self.batch, "en", context, force_longest=True)
        self.batch_written([noise], "en", "", force_longest=True)
        log.info("GPU memory reserved for clips: %.2f GB (in use %.2f GB, peak %.2f GB)",
                 torch.cuda.memory_reserved() / 1e9, torch.cuda.memory_allocated() / 1e9,
                 torch.cuda.max_memory_allocated() / 1e9)

    def recognize(self, samples: np.ndarray, language: str, context: str) -> list[str]:
        """16 kHz audio → the text of each chunk of speech. Out of GPU memory (a clip larger than the
        warm-up): the chunks go again one at a time; still out: GpuFull."""
        import torch

        self.wake()
        chunks = self.chunks(samples)
        if not chunks:
            return []
        for batch in (self.batch, 1):
            try:
                return self.written(chunks, language, context, batch)
            except torch.OutOfMemoryError:
                log.warning("out of GPU memory with %d chunks at a time", batch)
                gc.collect()
        raise GpuFull()

    def chunks(self, samples: np.ndarray) -> list[np.ndarray]:
        import torch
        from silero_vad import get_speech_timestamps

        audio = torch.from_numpy(samples).to("cuda")
        cuts = get_speech_timestamps(audio, self.vad, sampling_rate=RATE, threshold=VAD_THRESHOLD, **CUTS)
        self.vad.reset_states()
        if not cuts:
            log.info("no speech found: loudest %.4f rms over 0.1 s", loudest_rms(samples))
        return [samples[start:end] for start, end in chunked(cuts)]

    def written(self, chunks: list[np.ndarray], language: str, context: str, batch: int) -> list[str]:
        texts = []
        for i in range(0, len(chunks), batch):
            texts += self.batch_written(chunks[i:i + batch], language, context)
        return [t for t in texts if t]

    def batch_written(self, chunks: list[np.ndarray], language: str, context: str,
                      force_longest: bool = False) -> list[str]:
        """Some chunks through the model at once; greedy decoding. `force_longest`: every answer runs
        to the token limit (the warm-up's largest size)."""
        import torch

        n = len(chunks)
        inputs = self.processor.apply_transcription_request(
            list(chunks), language=[language] * n, prompt=[context] * n if context else None,
        ).to("cuda", torch.bfloat16)
        longest = max(len(c) for c in chunks) / RATE
        limit = int(longest * TOKENS_PER_SECOND) + 32
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=limit,
                                      min_new_tokens=limit if force_longest else None)
        answers = out[:, inputs["input_ids"].shape[1]:]
        return [self.processor.decode(a, skip_special_tokens=True).strip() for a in answers]


class Vocabulary:
    """One word list per speaker, a JSON file each in `folder`. The file name is a hash of the
    speaker, so any string is safe as a speaker; the speaker itself is kept inside."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def file(self, speaker: str) -> Path:
        return self.folder / (hashlib.sha256(speaker.encode()).hexdigest()[:32] + ".json")

    def get(self, speaker: str) -> list[str]:
        try:
            return json.loads(self.file(speaker).read_text())["words"]
        except FileNotFoundError:
            return []

    def put(self, speaker: str, words: list[str]) -> list[str]:
        file = self.file(speaker)
        tmp = file.with_suffix(".tmp")
        with self.lock:
            tmp.write_text(json.dumps({"speaker": speaker, "words": words}, ensure_ascii=False, indent=1))
            tmp.replace(file)
        return words


def tidy(words) -> list[str]:
    """The words as saved: strings, trimmed, inner spaces single, no blanks, no repeats, in order."""
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        raise HTTPException(400, 'body: {"words": [string, ...]}')
    seen, out = set(), []
    for w in words:
        w = " ".join(w.split())
        if not w or w in seen:
            continue
        if len(w) > MAX_WORD_CHARS:
            raise HTTPException(400, f"a word or phrase is at most {MAX_WORD_CHARS} characters")
        seen.add(w)
        out.append(w)
    if len(out) > MAX_WORDS:
        raise HTTPException(400, f"at most {MAX_WORDS} words")
    return out


def speaker_of(raw: str) -> str:
    speaker = raw.strip()
    if len(speaker) > 320:
        raise HTTPException(400, "speaker: at most 320 characters")
    return speaker


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


def gpu() -> dict:
    """This process's GPU memory, in MB: what torch holds for tensors (allocated), what it keeps
    from the GPU including its cache (reserved), all of it as the driver counts it (process,
    with the CUDA context), and the card's free and total."""
    import pynvml
    import torch

    mb = 1024 * 1024
    free, total = torch.cuda.mem_get_info()
    used = None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        used = sum(p.usedGpuMemory or 0 for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                   if p.pid == os.getpid()) // mb
    except pynvml.NVMLError:
        pass
    return {"allocated_mb": torch.cuda.memory_allocated() // mb, "reserved_mb": torch.cuda.memory_reserved() // mb,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() // mb, "process_mb": used,
            "card_free_mb": free // mb, "card_total_mb": total // mb, "cap_fraction": cap_fraction}


token = b""                 # from --token-file, before the first request
cap_fraction = None
speech: Speech | None = None
vocabulary: Vocabulary | None = None
cleaner: Cleaner | None = None
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
bearer = HTTPBearer(auto_error=False)


def authorized(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if cred is None or not secrets.compare_digest(cred.credentials.encode(), token):
        raise HTTPException(401, "a bearer token is needed", headers={"WWW-Authenticate": "Bearer"})


@contextlib.asynccontextmanager
async def lifespan(_app):
    watch = asyncio.create_task(idle_watch())
    yield
    watch.cancel()


# No /docs or /openapi.json: the API is in this docstring and the README.
app = FastAPI(dependencies=[Depends(authorized)], docs_url=None, redoc_url=None, openapi_url=None,
              lifespan=lifespan)


async def on_model(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(worker, fn, *args)


@app.get("/health")
async def health():
    memory = await on_model(gpu)
    models = {"asr": {"name": speech.path.name, "state": speech.state},
              "cleanup": await cleaner.health() if cleaner else {"state": "off"}}
    return JSONResponse({"ok": True, "gpu": memory, "models": models})


@app.get("/v1/vocabulary/{speaker}")
def get_vocabulary(speaker: str):
    return JSONResponse({"words": vocabulary.get(speaker_of(speaker))})


@app.put("/v1/vocabulary/{speaker}")
async def put_vocabulary(speaker: str, body: dict = Body(...)):
    words = tidy(body.get("words"))
    vocabulary.put(speaker_of(speaker), words)
    return JSONResponse({"words": words})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    language: str = Form(""),
    speaker: str = Form(""),
    cleanup: str = Form("true"),
    prompt: str = Form(""),
    context: str = Form(""),
    response_format: str = Form("json"),
    model: str = Form(""),          # OpenAI's; accepted and ignored
):
    if response_format not in ("json", "text"):
        raise HTTPException(400, 'response_format: "json" or "text"')
    from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

    language = language.strip().lower() or "en"
    if language not in LANGUAGE_CODE_TO_NAME:
        raise HTTPException(400, f"language: one of {', '.join(sorted(LANGUAGE_CODE_TO_NAME))}")
    clean = cleanup.strip().lower() not in ("false", "0", "no", "off")
    speaker = speaker_of(speaker)
    words = vocabulary.get(speaker) if speaker else []
    terms = ", ".join(w for w in (*words, prompt.strip()) if w)
    around = newest_words(context, CONTEXT_WORDS)
    # The GPU room was reserved at start for WARMUP_CONTEXT_WORDS: the caller's context gives way.
    hint = "\n".join(t for t in (terms, newest_words(around, WARMUP_CONTEXT_WORDS - len(terms.split()))) if t)
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, f"the clip is over {MAX_BYTES // (1024 * 1024)} MB")
    data = await file.read()
    try:
        samples = decode(data)
    except (av.FFmpegError, ValueError) as exc:
        raise HTTPException(400, f"cannot read the clip: {exc}") from exc

    t0 = time.time()
    try:
        chunks = await on_model(speech.recognize, samples, language, hint) if len(samples) else []
    except GpuFull as exc:
        raise HTTPException(503, "the GPU has no memory left just now: send the clip again") from exc
    raw = " ".join(chunks)
    if not raw and loudest_rms(samples) > MISSED_RMS:
        keep_missed(data)
    t1 = time.time()
    listed = [*words, *(w.strip() for w in prompt.split(",") if w.strip())]
    text = await cleaner.clean(chunks, listed, around) if clean and cleaner and chunks else raw
    log.info("%.1fs audio, %d words listed, %d of context: asr %.2fs, cleanup %.2fs, %d -> %d chars",
             len(samples) / RATE, len(words), len(around.split()), t1 - t0, time.time() - t1,
             len(raw), len(text))
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text, "raw": raw})



def newest_words(text: str, n: int) -> str:
    """The last n words of text (all of it when shorter; "" for n <= 0)."""
    words = text.split()
    return " ".join(words[-n:]) if n > 0 else ""


MISSED_RMS = 0.01   # louder than room noise: a clip this loud that gave no text was probably speech
MISSED_KEEP = 20


def keep_missed(data: bytes) -> None:
    """A loud clip that gave no text, kept as sent (the newest MISSED_KEEP) in the data folder's
    missed/, to replay and see why."""
    folder = vocabulary.folder.parent / "missed"
    folder.mkdir(exist_ok=True)
    (folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}.audio").write_bytes(data)
    for old in sorted(folder.glob("*.audio"))[:-MISSED_KEEP]:
        old.unlink()
    log.info("kept the missed clip in %s", folder)


async def idle_watch() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            await on_model(speech.park)
        except Exception:
            log.exception("parking the idle model failed")


def main() -> None:
    global speech, token, vocabulary, cleaner, cap_fraction
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", action="append", help="address to listen on; repeat for more (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--model", type=Path, required=True, help="Musci-ASR-qwen-1019's folder (transformers)")
    ap.add_argument("--token-file", required=True, type=Path,
                    help="file holding the bearer token every request must carry")
    ap.add_argument("--data-dir", required=True, type=Path, help="where the word lists are kept")
    ap.add_argument("--gpu-fraction", type=float, required=True,
                    help="the most of the GPU's memory this process may take, e.g. 0.4; past it a clip "
                         "goes again one chunk at a time, then gets 503")
    ap.add_argument("--idle-minutes", type=float, required=True,
                    help="after this long without a clip the model is parked in RAM and its GPU memory freed")
    ap.add_argument("--batch", type=int, default=4, help="chunks of speech through the model at once")
    ap.add_argument("--cleanup-url", help="an OpenAI-style chat server (llama-server), e.g. http://127.0.0.1:7863; "
                                          "without it there is no cleanup and text == raw")
    ap.add_argument("--cleanup-key-file", type=Path, help="file holding that server's API key")
    ap.add_argument("--cleanup-timeout", type=float, default=10.0,
                    help="seconds to wait for the cleanup; past it the raw text goes out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    token = args.token_file.expanduser().read_text().strip().encode()
    if not token:
        ap.error(f"{args.token_file} is empty")
    if not 0 < args.gpu_fraction <= 1:
        ap.error("--gpu-fraction: above 0, at most 1")
    import torch

    if not torch.cuda.is_available():
        ap.error("no CUDA GPU: claudia-listen runs on the GPU only")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    cap_fraction = args.gpu_fraction
    vocabulary = Vocabulary(args.data_dir.expanduser() / "vocabulary")
    if args.cleanup_url:
        key = args.cleanup_key_file.expanduser().read_text().strip() if args.cleanup_key_file else ""
        cleaner = Cleaner(args.cleanup_url, key, args.cleanup_timeout)
    log.info("cleanup: %s", args.cleanup_url or "off")
    t0 = time.time()
    speech = worker.submit(Speech, args.model.expanduser(), args.batch, args.idle_minutes * 60).result()
    worker.submit(speech.warmup).result()
    log.info("model loaded and warmed up in %.1fs", time.time() - t0)
    hosts = args.host or ["127.0.0.1"]
    sockets = [listen(h, args.port) for h in hosts]
    log.info("ready on %s, port %d", ", ".join(hosts), args.port)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=sockets)


def listen(host: str, port: int) -> socket.socket:
    """The listening socket. SO_REUSEPORT lets a new copy of this service bind beside a running one
    (same user): both answer until the old one stops, so a restart loses no clip. A port held
    without it (another program) is waited for, after the model is loaded, then taken."""
    waited = False
    while True:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # A VPN address may come up after this service; bind it now and serve it once it does.
        sock.setsockopt(socket.IPPROTO_IP, getattr(socket, "IP_FREEBIND", 15), 1)   # 15 on Linux
        try:
            sock.bind((host, port))
            return sock
        except OSError as exc:
            sock.close()
            if exc.errno != errno.EADDRINUSE:
                raise
            if not waited:
                log.info("%s:%d is taken; waiting for it", host, port)
                waited = True
            time.sleep(0.02)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the model is a local file; never ask the Hub
    main()
