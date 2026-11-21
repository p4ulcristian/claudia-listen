# The cleanup's LoRA adapter

`train.py` fine-tunes Gemma 4 E4B (instruction-tuned) with a LoRA adapter on the pairs in `data/`
(JSON Lines: `in` the transcript as heard, `out` what the speaker meant, optional `words` and
`context`), with the exact prompt `cleanup.py` sends. Loss is on the reply only; the base loads in
4 bit, so it trains beside the live service on a 32 GB card (~12 GB, ~35 min for 1,700 pairs × 2).

The pairs were written by Claude from the rules in `cleanup.SYSTEM`: `a-*` self-corrections, false
starts, repeats and fillers; `b-*` what must stay, grammar, never answering; `c-*` names, word lists,
context, codes and spoken code. Every pair passes `cleanup.plausible`.

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 \
      "torch==2.11.0+cu128" transformers peft trl bitsandbytes accelerate datasets httpx wordfreq
    git clone --depth 1 --branch b12451 https://github.com/ggml-org/llama.cpp
    uv pip install --python .venv/bin/python -e llama.cpp/gguf-py
    .venv/bin/python train.py .. data run          # adapter in run/adapter
    .venv/bin/python llama.cpp/convert_lora_to_gguf.py --base <gemma-4-E4B-it snapshot> \
      --outtype f16 --outfile run/adapter.gguf run/adapter

llama-server loads it on top of the Q4_K_M model with `--lora` (claudia-listen-cleanup.service).
Before shipping a new adapter, compare it with the one running on held-out cases and on real
transcripts.

`eval/hand.json` holds 49 hand-written cases (`in`, accepted outputs in `ok`, optional `words`)
kept out of training: the run of 2026-11-21 got 49/49, the prompt alone 44/49. A new adapter
should not score lower.
