"""
Modified runners that generate full responses and return (answer, reasoning) tuples.
Drop-in replacement for pipeline.load_models.RUNNERS in pt2_test.
"""
import logging
import re

import torch

log = logging.getLogger(__name__)

# Generate enough tokens to capture a full reasoning paragraph
_MAX_NEW_TOKENS = {
    "granite": 300,
    "llama":   300,
    "mistral": 300,
}


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
            do_sample=False,
            temperature=0,
            max_new_tokens=_MAX_NEW_TOKENS[name],
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def _parse_yes_no_reasoning(decoded_list: list[str]) -> list[tuple[str, str]]:
    """Return (answer, reasoning) for each decoded response."""
    results = []
    for decoded in decoded_list:
        decoded = decoded.strip()
        m = re.search(r'\b(yes|no)\b', decoded, re.IGNORECASE)
        if m:
            answer = m.group(1).lower()
            # everything after the first yes/no, stripped of leading punctuation/spaces
            reasoning = decoded[m.end():].strip().lstrip(".,;:- ").strip()
        else:
            answer = "improper output"
            reasoning = decoded
        results.append((answer, reasoning))
    return results


def run_granite_batch(texts: list[str], model, tokenizer) -> list[tuple[str, str]]:
    return _parse_yes_no_reasoning(_batch_generate(texts, model, tokenizer, "granite"))


def run_llama_batch(texts: list[str], model, tokenizer) -> list[tuple[str, str]]:
    return _parse_yes_no_reasoning(_batch_generate(texts, model, tokenizer, "llama"))


def run_mistral_batch(texts: list[str], model, tokenizer) -> list[tuple[str, str]]:
    return _parse_yes_no_reasoning(_batch_generate(texts, model, tokenizer, "mistral"))


RUNNERS = {
    "granite": run_granite_batch,
    "llama":   run_llama_batch,
    "mistral": run_mistral_batch,
}
