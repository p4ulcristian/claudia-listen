"""iris-listen: speech to finished text on your own GPU. Canary-1B-v2 hears, each person's own
word list steers it, a small local language model cleans the text up; behind OpenAI's
transcription API.

    POST /v1/audio/transcriptions   multipart form → {"text": cleaned, "raw": heard}
        file             the clip: WAV, FLAC, OGG, MP3, MP4/M4A or WebM (what browsers record;
                         anything FFmpeg reads), any rate, mono or not
        language         optional, e.g. "en": forces that language; without it each stretch of
                         speech is written out in each of --languages and the one the model is
                         surest of wins
        speaker          optional: whose word list steers the model and the cleanup (an opaque
                         string, e.g. a sign-in email); none = no list
        cleanup          optional, "false" skips the cleanup: then text == raw
        prompt           accepted and ignored: Canary takes no context
        response_format  "json" (default) or "text" (the cleaned text alone)
        model            accepted and ignored (OpenAI's clients send it)
    GET  /v1/vocabulary/{speaker}   → {"words": [...]}  ([] for a speaker never seen)
    PUT  /v1/vocabulary/{speaker}   {"words": [...]} → {"words": [...]} as saved
    GET  /health                    → {"ok": true, "gpu": {...}, "models": {...}}

Every request needs `Authorization: Bearer <token>`, the token in --token-file; without it the
answer is 401. So it is safe on any address, behind a proxy or not.

Run it with this folder's venv:  .venv/bin/python server.py --token-file <file> ...
It listens on 127.0.0.1 by default. Each --host adds an address; one that is not up yet (a
VPN's) is bound anyway and starts answering when it comes up.

Canary has no language detection: told a language, it writes in that language, and told the
wrong one it translates. So a forced language must be the one spoken. Silero VAD cuts each clip
into stretches of speech at its pauses; silence gives "". One clip at a time; others wait.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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

log = logging.getLogger("iris-listen")

RATE = 16000
MAX_BYTES = 25 * 1024 * 1024   # OpenAI's limit too
# The 25 languages Canary-1B-v2 is trained on (its model card).
LANGUAGES = frozenset("bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split())
MAX_WORDS = 1000           # per speaker
MAX_WORD_CHARS = 100

# Where a clip is cut into stretches of speech (Silero VAD): at a pause of at least 250 ms, so a
# stretch is a sentence or less and its language is picked on its own (with 500 ms, a switch of
# language after a 0.3 s pause lost the second language's sentences); a stretch over 20 s is cut
# anyway (Canary is trained on clips up to 40 s). Each stretch keeps 0.1 s of the quiet around it,
# so no word loses its first or last sound.
# Silero's speech chance a frame needs to start a stretch. Its default 0.5 dropped quiet iPad speech:
# 9 of 87 clips came back empty on 2026-11-17 (the old onnx-asr server: 28 of 853).
VAD_THRESHOLD = 0.3
CUTS = {"min_silence_duration_ms": 250, "max_speech_duration_s": 20, "speech_pad_ms": 100}

# Phrase boosting (NeMo GPU-PB) in Canary's beam search. NVIDIA's numbers for Canary: 1.0 a step
# along a phrase, no extra weight for depth.
BOOST = {"context_score": 1.0, "depth_scaling": 1.0}


class GpuFull(Exception):
    """The GPU has no memory left even for one stretch: the clip is refused, to be sent again."""


def loudest_rms(samples: np.ndarray) -> float:
    """The loudest 0.1 s of the clip, as RMS (for the log when no speech is found)."""
    n = RATE // 10
    if len(samples) < n:
        return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    return float(np.sqrt((frames ** 2).mean(axis=1)).max())


class Speech:
    """Canary-1B-v2 (NeMo, bf16) on the GPU, behind Silero VAD (on the GPU too).

    Decoding is beam search with NeMo's fusion models: a speaker's word list is a prepared
    boosting tree (GPUBoostingTreeModel), built once per list and cached, and each request puts
    its speaker's tree into the beam search's `fusion_models` (none for no speaker), so switching
    speakers costs nothing. Canary has no language detection, so with several `languages` each
    stretch is written out in every one and the version with the highest score per token wins:
    the language spoken scores far above a translation. Picking per stretch keeps both languages
    of a clip that switches between them.

    After `idle` seconds without a clip the model is parked in RAM and its GPU memory given back;
    the next clip brings it back (a second or two). Every call runs on the one model thread.
    """

    def __init__(self, path: Path, languages: list[str], beam: int, batch: int, alpha: float,
                 idle: float) -> None:
        self.path, self.languages, self.beam, self.batch, self.alpha = path, languages, beam, batch, alpha
        self.idle = idle
        self.trees: dict[tuple[str, ...], object] = {}   # word list → its boosting tree
        self.prompts: dict = {}                            # language → Canary's prompt tokens
        self.state = "loading"
        self.last = time.monotonic()
        self.load()

    def load(self) -> None:
        import torch
        from nemo.collections.asr.models import EncDecMultiTaskModel
        from omegaconf import open_dict
        from silero_vad import load_silero_vad

        cfg = EncDecMultiTaskModel.restore_from(str(self.path), return_config=True)
        with open_dict(cfg):
            cfg.restore_timestamps_model = False   # a CTC model for word times: not used here
        model = EncDecMultiTaskModel.restore_from(str(self.path), override_config_path=cfg, map_location="cpu")
        model.eval()
        for part in (model.encoder, model.transf_decoder, model.log_softmax):
            part.to(torch.bfloat16)
        self.model = model.to("cuda")
        decoding = model.cfg.decoding
        with open_dict(decoding):
            decoding.strategy = "beam"
            decoding.beam.beam_size = self.beam
            decoding.beam.return_best_hypothesis = False   # the scores pick the language
            # A placeholder list, so NeMo builds the beam search that takes fusion models;
            # each request then sets its own (or none).
            decoding.beam.boosting_tree = {"key_phrases_list": ["Iris"], **BOOST}
            decoding.beam.boosting_tree_alpha = self.alpha
        model.change_decoding_strategy(decoding)
        self.search = model.decoding.decoding.beam_search
        self.vad = load_silero_vad().to("cuda")
        self.state = "gpu"

    def tree(self, words: tuple[str, ...]):
        """The boosting tree for a word list, built on first use and kept."""
        from nemo.collections.asr.parts.context_biasing import BoostingTreeModelConfig, GPUBoostingTreeModel

        if words not in self.trees:
            t0 = time.time()
            cfg = BoostingTreeModelConfig(key_phrases_list=list(words), **BOOST)
            self.trees[words] = GPUBoostingTreeModel.from_config(cfg, tokenizer=self.model.tokenizer)
            log.info("boosting tree for %d words built in %.2fs", len(words), time.time() - t0)
        return self.trees[words]

    def forget(self, words: tuple[str, ...]) -> None:
        self.trees.pop(words, None)

    def wake(self) -> None:
        import torch

        if self.state == "ram":
            t0 = time.time()
            self.model.to("cuda")
            self.vad.to("cuda")
            self.prompts = {}
            self.state = "gpu"
            log.info("model back on the GPU in %.1fs", time.time() - t0)
        self.last = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def park(self) -> None:
        """Model to RAM, all its GPU memory given back (the CUDA context stays)."""
        import torch

        if self.state != "gpu" or time.monotonic() - self.last < self.idle:
            return
        self.model.to("cpu")
        self.vad.to("cpu")
        for tree in self.trees.values():
            tree.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        self.state = "ram"
        log.info("idle: model parked in RAM, GPU memory given back")

    def warmup(self) -> None:
        """Run the model once, so the first clip doesn't wait for kernels to load."""
        noise = np.random.default_rng(0).normal(0, 0.01, RATE * 3).astype(np.float32)
        self.written([noise], self.languages, self.tree(("Iris",)), 1)
        self.written([noise], self.languages, None, 1)
        self.done()

    def recognize(self, samples: np.ndarray, language: str | None, words: tuple[str, ...]) -> list[str]:
        """16 kHz audio → the text of each stretch of speech. Out of GPU memory: the cache is
        given back and the stretches go again one at a time; still out: GpuFull."""
        import torch

        self.wake()
        try:
            stretches = self.stretches(samples)
            if not stretches:
                return []
            tree = self.tree(words) if words else None
            langs = [language] if language else self.languages
            for batch in (self.batch, 1):
                try:
                    return self.written(stretches, langs, tree, batch)
                except torch.OutOfMemoryError:
                    log.warning("out of GPU memory with %d stretches at a time", batch)
                    gc.collect()
                    torch.cuda.empty_cache()
            raise GpuFull()
        finally:
            self.done()

    def done(self) -> None:
        """After every clip: the memory torch kept for reuse goes back to the GPU."""
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    def stretches(self, samples: np.ndarray) -> list[np.ndarray]:
        import torch
        from silero_vad import get_speech_timestamps

        audio = torch.from_numpy(samples).to("cuda")
        cuts = get_speech_timestamps(audio, self.vad, sampling_rate=RATE, threshold=VAD_THRESHOLD, **CUTS)
        self.vad.reset_states()
        if not cuts:
            log.info("no speech found: loudest %.4f rms over 0.1 s", loudest_rms(samples))
        return [samples[c["start"]:c["end"]] for c in cuts]

    def written(self, stretches: list[np.ndarray], langs: list[str], tree, batch: int) -> list[str]:
        texts = []
        for i in range(0, len(stretches), batch):
            texts += self.batch_written(stretches[i:i + batch], langs, tree)
        return [t for t in texts if t]

    def batch_written(self, stretches: list[np.ndarray], langs: list[str], tree) -> list[str]:
        """Some stretches through the encoder once, then through the decoder in each language;
        per stretch the version with the highest score per token wins. Straight through the
        model, not NeMo's transcribe(), which sets up a data loader and runs the encoder again
        for every call."""
        import torch

        self.search.fusion_models = [tree] if tree is not None else []
        self.search.fusion_models_alpha = [self.alpha] if tree is not None else []
        lens = torch.tensor([len(s) for s in stretches], device="cuda")
        audio = torch.zeros(len(stretches), int(lens.max()), device="cuda")
        for k, s in enumerate(stretches):
            audio[k, :len(s)] = torch.from_numpy(s)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, states, mask = self.model.forward(input_signal=audio, input_signal_length=lens)
            # All languages in one batch: the beam search's cost is its steps, not their width.
            n = len(stretches)
            found = self.model.decoding.decode_predictions_tensor(
                encoder_hidden_states=states.repeat(len(langs), 1, 1), encoder_input_mask=mask.repeat(len(langs), 1),
                decoder_input_ids=torch.cat([self.prompt(lang).repeat(n, 1) for lang in langs]),
                return_hypotheses=True)
            versions = [[self.best(h) for h in found[k * n:(k + 1) * n]] for k in range(len(langs))]
        texts = []
        for stretch in zip(*versions, strict=True):
            # An empty version scores per token as well as any (no tokens to lose on), so it
            # could beat the real text: a version with words wins over one without.
            best = max(range(len(langs)), key=lambda k: (bool(stretch[k][0]), stretch[k][1]))
            if not stretch[best][0]:
                log.info("stretch heard as nothing in %s", "/".join(langs))
            log.debug("stretch: %s", {lang: round(v[1], 3) for lang, v in zip(langs, stretch, strict=True)})
            texts.append(stretch[best][0])
        return texts

    @staticmethod
    def best(nbest) -> tuple[str, float]:
        """The best hypothesis of a beam: its text and its score per token."""
        hyps = nbest if isinstance(nbest, list) else getattr(nbest, "n_best_hypotheses", [nbest])
        h = max(hyps, key=lambda h: float(h.score))
        return h.text.strip(), float(h.score) / max(1, len(h.y_sequence))

    def prompt(self, lang: str):
        """Canary's prompt for writing out speech in `lang` as it is spoken, with punctuation
        and capitals (its other slots at their defaults); made once per language."""
        if lang not in self.prompts:
            turns = self.model.prompt.get_default_dialog_slots()
            for turn in turns:
                if turn["role"] == "user":
                    turn["slots"] = {**turn["slots"], "source_lang": lang, "target_lang": lang, "pnc": "yes"}
            self.prompts[lang] = self.model.prompt.encode_dialog(turns=turns)["context_ids"].unsqueeze(0).to("cuda")
        return self.prompts[lang]


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
    models = {"asr": {"name": speech.path.stem, "state": speech.state, "boosting_trees": len(speech.trees)},
              "cleanup": await cleaner.health() if cleaner else {"state": "off"}}
    return JSONResponse({"ok": True, "gpu": memory, "models": models})


@app.get("/v1/vocabulary/{speaker}")
def get_vocabulary(speaker: str):
    return JSONResponse({"words": vocabulary.get(speaker_of(speaker))})


@app.put("/v1/vocabulary/{speaker}")
async def put_vocabulary(speaker: str, body: dict = Body(...)):
    speaker = speaker_of(speaker)
    words = tidy(body.get("words"))
    old = tuple(vocabulary.get(speaker))
    vocabulary.put(speaker, words)
    if words:   # built now, so the speaker's next clip doesn't wait for it
        await on_model(speech.tree, tuple(words))
    if old and old != tuple(words):
        await on_model(speech.forget, old)
    return JSONResponse({"words": words})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    language: str = Form(""),
    speaker: str = Form(""),
    cleanup: str = Form("true"),
    prompt: str = Form(""),         # accepted and ignored: Canary takes no context
    response_format: str = Form("json"),
    model: str = Form(""),          # OpenAI's; accepted and ignored
):
    if response_format not in ("json", "text"):
        raise HTTPException(400, 'response_format: "json" or "text"')
    language = language.strip().lower() or None
    if language and language not in LANGUAGES:
        raise HTTPException(400, f"language: one of {', '.join(sorted(LANGUAGES))}")
    clean = cleanup.strip().lower() not in ("false", "0", "no", "off")
    speaker = speaker_of(speaker)
    words = tuple(vocabulary.get(speaker)) if speaker else ()
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, f"the clip is over {MAX_BYTES // (1024 * 1024)} MB")
    data = await file.read()
    try:
        samples = decode(data)
    except (av.FFmpegError, ValueError) as exc:
        raise HTTPException(400, f"cannot read the clip: {exc}") from exc

    t0 = time.time()
    try:
        stretches = await on_model(speech.recognize, samples, language, words) if len(samples) else []
    except GpuFull as exc:
        raise HTTPException(503, "the GPU has no memory left just now: send the clip again") from exc
    raw = " ".join(stretches)
    if not raw and loudest_rms(samples) > MISSED_RMS:
        keep_missed(data)
    t1 = time.time()
    text = await cleaner.clean(stretches, list(words)) if clean and cleaner and stretches else raw
    log.info("%.1fs audio, %d words listed: asr %.2fs, cleanup %.2fs, %d -> %d chars",
             len(samples) / RATE, len(words), t1 - t0, time.time() - t1, len(raw), len(text))
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text, "raw": raw})


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
    ap.add_argument("--model", type=Path, required=True, help="Canary-1B-v2's .nemo file")
    ap.add_argument("--languages", nargs="+", required=True, metavar="CODE",
                    help="the languages a clip without one may be in, e.g. hu en")
    ap.add_argument("--token-file", required=True, type=Path,
                    help="file holding the bearer token every request must carry")
    ap.add_argument("--data-dir", required=True, type=Path, help="where the word lists are kept")
    ap.add_argument("--gpu-fraction", type=float, required=True,
                    help="the most of the GPU's memory this process may take, e.g. 0.4; past it a clip "
                         "goes again one stretch at a time, then gets 503")
    ap.add_argument("--idle-minutes", type=float, required=True,
                    help="after this long without a clip the model is parked in RAM and its GPU memory freed")
    ap.add_argument("--beam", type=int, default=4, help="beam size (phrase boosting works in the beam)")
    ap.add_argument("--batch", type=int, default=4, help="stretches of speech through the model at once")
    ap.add_argument("--boost", type=float, default=1.0, help="weight of a speaker's word list in the beam search")
    ap.add_argument("--cleanup-url", help="an OpenAI-style chat server (llama-server), e.g. http://127.0.0.1:7863; "
                                          "without it there is no cleanup and text == raw")
    ap.add_argument("--cleanup-key-file", type=Path, help="file holding that server's API key")
    ap.add_argument("--cleanup-timeout", type=float, default=10.0,
                    help="seconds to wait for the cleanup; past it the raw text goes out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("nemo_logger", "nemo", "lightning", "pytorch_lightning", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    token = args.token_file.expanduser().read_text().strip().encode()
    if not token:
        ap.error(f"{args.token_file} is empty")
    if unknown := sorted(set(args.languages) - LANGUAGES):
        ap.error(f"Canary does not know {', '.join(unknown)}; it knows {', '.join(sorted(LANGUAGES))}")
    if not 0 < args.gpu_fraction <= 1:
        ap.error("--gpu-fraction: above 0, at most 1")
    import torch

    if not torch.cuda.is_available():
        ap.error("no CUDA GPU: iris-listen runs on the GPU only")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    cap_fraction = args.gpu_fraction
    vocabulary = Vocabulary(args.data_dir.expanduser() / "vocabulary")
    if args.cleanup_url:
        key = args.cleanup_key_file.expanduser().read_text().strip() if args.cleanup_key_file else ""
        cleaner = Cleaner(args.cleanup_url, key, args.cleanup_timeout)
    log.info("languages: %s; cleanup: %s", ", ".join(args.languages), args.cleanup_url or "off")
    t0 = time.time()
    speech = worker.submit(Speech, args.model.expanduser(), args.languages, args.beam, args.batch, args.boost,
                           args.idle_minutes * 60).result()
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
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the model is a local file; never ask the Hub
    main()
