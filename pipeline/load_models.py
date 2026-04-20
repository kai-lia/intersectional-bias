import logging
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

MODEL_IDS = {
    "granite": "ibm-granite/granite-3.0-8b-instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

# yes/no appears first in the output — 20 tokens is enough for all models
_MAX_NEW_TOKENS = {
    "granite": 20,
    "llama":   20,
    "mistral": 20,
}


# ── Device detection ──────────────────────────────────────────────────────────

def detect_device() -> tuple[str, object, torch.dtype]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"CUDA device: {name}, VRAM: {vram:.1f} GB")
        return "cuda", "auto", torch.float16

    if torch.backends.mps.is_available():
        log.info("MPS device available (Apple Silicon).")
        return "mps", {"": "mps"}, torch.float16

    if torch.version.cuda is None:
        log.error("PyTorch installed without CUDA. Reinstall with CUDA wheels.")
    else:
        log.error("CUDA build present but no GPU visible — check SLURM flags or nvidia-smi.")
    log.error("Exiting — running this workload on CPU is not practical.")
    sys.exit(1)


def mem_used(device: str) -> str:
    if device == "cuda":
        return f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
    if device == "mps":
        return f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB"
    return "N/A"


# ── Loader / unloader ─────────────────────────────────────────────────────────

def load_model(name: str, device_map, dtype):
    model_id = MODEL_IDS[name]
    log.info(f"[{name}] Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # left-pad so batches of different lengths generate correctly
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
    log.info(f"[{name}] Loaded on {next(model.parameters()).device}")
    return model, tokenizer


def unload_model(name: str, model, tokenizer, device: str) -> None:
    log.info(f"[{name}] Unloading model")
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ── Batch runner functions ────────────────────────────────────────────────────
# Each takes a list of prompt strings and returns a list of answers.

def _batch_generate(texts: list[str], model, tokenizer, name: str) -> list[str]:
    messages = [[{"role": "user", "content": t}] for t in texts]
    chats = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    inputs = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=_MAX_NEW_TOKENS[name],
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


def run_granite_batch(texts: list[str], model, tokenizer) -> list[str]:
    decoded_list = _batch_generate(texts, model, tokenizer, "granite")
    results = []
    for decoded in decoded_list:
        m = re.search(r'<\|end_of_role\|>\s*(yes|no)\b', decoded, re.IGNORECASE)
        results.append(m.group(1).lower() if m else "improper output")
    return results


def run_llama_batch(texts: list[str], model, tokenizer) -> list[str]:
    decoded_list = _batch_generate(texts, model, tokenizer, "llama")
    results = []
    for decoded in decoded_list:
        split = re.search(
            r'(?<=<\|start_header_id\|>assistant<\|end_header_id\|>\n\n)(.*)',
            decoded, re.IGNORECASE,
        )
        if split:
            m = re.search(r'\b(yes|no)\b', split.group(0), re.IGNORECASE)
            if m:
                results.append(m.group(0).lower())
                continue
        results.append("improper output")
    return results


def run_mistral_batch(texts: list[str], model, tokenizer) -> list[str]:
    decoded_list = _batch_generate(texts, model, tokenizer, "mistral")
    results = []
    for decoded in decoded_list:
        m = re.search(r'\[/INST](.*?)(\b(yes|no)\b)', decoded, re.IGNORECASE)
        results.append(m.group(3).lower() if m else "improper output")
    return results


RUNNERS = {
    "granite": run_granite_batch,
    "llama":   run_llama_batch,
    "mistral": run_mistral_batch,
}
