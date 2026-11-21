"""LoRA fine-tune of the cleanup model (Gemma 4 E4B it) on messy -> clean pairs, with the exact
prompt claudia-listen sends (cleanup.SYSTEM + vocabulary + context, the transcript in tags).

    python train.py <src dir with cleanup.py> <data dir with *.jsonl> <out dir>

Loss is on the reply only (TRL prompt/completion). The base is loaded in 4 bit (QLoRA), so it fits
beside the live dictation on the GPU; the adapter goes on the language layers only.
"""
import glob
import json
import os
import random
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

SRC, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)
import cleanup  # noqa: E402

BASE = "google/gemma-4-E4B-it"


def system_for(words, context):
    s = cleanup.SYSTEM + "\n\nVocabulary:\n" + ("\n".join(f"- {w}" for w in words) or "(none)")
    if context:
        s += f"\n\nContext:\n<context>{context}</context>"
    return s


def example(row):
    return {"prompt": [{"role": "system", "content": system_for(row.get("words") or [], row.get("context") or "")},
                       {"role": "user", "content": f"<transcript>{row['in']}</transcript>"}],
            "completion": [{"role": "assistant", "content": row["out"]}]}


def rows():
    out = []
    for f in sorted(glob.glob(f"{DATA}/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                out.append(example(json.loads(line)))
    random.Random(7).shuffle(out)
    return out


def main():
    data = rows()
    print("examples:", len(data))
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      # the language layers only (Gemma 4 E4B also has vision and audio towers), under
                      # the names llama.cpp's convert_lora_to_gguf.py maps
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    cfg = SFTConfig(output_dir=OUT, num_train_epochs=2, per_device_train_batch_size=1,
                    gradient_accumulation_steps=16, learning_rate=1e-4, lr_scheduler_type="cosine",
                    warmup_steps=10, completion_only_loss=True, logging_steps=10, save_strategy="no", bf16=True,
                    gradient_checkpointing=True, max_length=3072, report_to=[],
                    max_steps=int(os.environ.get("MAX_STEPS", "-1")))  # MAX_STEPS=2: a smoke run
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(data),
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.model.save_pretrained(f"{OUT}/adapter")
    tok.save_pretrained(f"{OUT}/adapter")
    print("saved", f"{OUT}/adapter")


if __name__ == "__main__":
    main()
