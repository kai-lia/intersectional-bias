import gc
import logging
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow ops not yet implemented on MPS to fall back to CPU silently.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

log = logging.getLogger(__name__)

MODEL_IDS = {
    "granite": "ibm-granite/granite-3.0-8b-instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

# yes/no is always the first generated token — 5 is a safe ceiling
_MAX_NEW_TOKENS = {
    "granite": 5,
    "llama":   5,
    "mistral": 5,
}


# ── Device detection ──────────────────────────────────────────────────────────

def _cuda_batch_size(vram_gb: float) -> int:
    """Recommend batch size based on available VRAM for ~7-8B models."""
    if vram_gb >= 80:
        return 64
    if vram_gb >= 40:
        return 32
    if vram_gb >= 24:
        return 16
    return 8


def detect_device() -> tuple[str, object, torch.dtype, int]:
    """Return (device_str, device_map, dtype, recommended_batch_size)."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        name  = props.name
        vram  = props.total_memory / 1e9
        # bfloat16 is preferred on Ampere (sm_80) and newer; fall back to float16
        dtype = torch.bfloat16 if props.major >= 8 else torch.float16
        batch = _cuda_batch_size(vram)
        log.info(f"CUDA device: {name}, VRAM: {vram:.1f} GB, dtype: {dtype}, batch: {batch}")
        return "cuda", "auto", dtype, batch

    if torch.backends.mps.is_available():
        # Apple Silicon unified memory — bfloat16 is well-supported on M-series
        # 64 is safe for 8B models; ~50 GB free after model load on a 64 GB system
        batch = 64
        log.info(f"MPS device (Apple Silicon), dtype: bfloat16, batch: {batch}")
        return "mps", {"": "mps"}, torch.bfloat16, batch

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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map,
        attn_implementation="sdpa",
    )
    log.info(f"[{name}] Loaded on {next(model.parameters()).device}")
    return model, tokenizer


def unload_model(name: str, model, tokenizer, device: str) -> None:
    log.info(f"[{name}] Unloading model")
    del model, tokenizer
    gc.collect()
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
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=_MAX_NEW_TOKENS[name],
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the newly generated tokens — avoids re-processing the full input.
    new_tokens = outputs[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def _parse_yes_no(decoded_list: list[str]) -> list[str]:
    """Extract yes/no from decoded new-token strings."""
    results = []
    for decoded in decoded_list:
        m = re.search(r'\b(yes|no)\b', decoded, re.IGNORECASE)
        results.append(m.group(1).lower() if m else "improper output")
    return results


def run_granite_batch(texts: list[str], model, tokenizer) -> list[str]:
    return _parse_yes_no(_batch_generate(texts, model, tokenizer, "granite"))


def run_llama_batch(texts: list[str], model, tokenizer) -> list[str]:
    return _parse_yes_no(_batch_generate(texts, model, tokenizer, "llama"))


def run_mistral_batch(texts: list[str], model, tokenizer) -> list[str]:
    return _parse_yes_no(_batch_generate(texts, model, tokenizer, "mistral"))


RUNNERS = {
    "granite": run_granite_batch,
    "llama":   run_llama_batch,
    "mistral": run_mistral_batch,
}
