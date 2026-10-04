"""
One-time extraction of just enough of the model's output head to do logit-lens
on the yes/no decision: the final-layer-norm parameters plus the lm_head rows
for a curated set of yes/no token-string variants (not the full vocab, which
would be hundreds of MB -- we only need ~10-20 rows).

Why this is possible without a new generation run: extract_activations.py
(reused by random_sample_activations.py) already saves the residual stream at
the *final prompt token* -- exactly the position logit-lens needs -- so the
existing activations_random/*.npz files can be projected through this head
locally, no new forward passes required.

Introspects the model's actual final-norm module rather than assuming
RMSNorm vs LayerNorm, so this works whether Granite turns out to use one or
the other -- saves a `norm_type` flag the analysis script branches on.

Output: pt2_test/data/lm_head_yesno.npz
    token_strs        (n,)   object array of the candidate strings tried
    token_ids         (n,)   int array, -1 for strings that didn't tokenize
                              to exactly one token (skipped downstream)
    is_yes            (n,)   bool, True for yes-like variants, False for no-like
    lm_head_rows      (n, d) float32, the lm_head weight row for each valid token
    norm_weight       (d,)   float32
    norm_bias         (d,) or None
    norm_type         str, e.g. "RMSNorm" or "LayerNorm" -- read by the
                      downstream logit_lens.py to pick the right formula
    norm_eps          float
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from pipeline.load_models import detect_device, load_model, unload_model, mem_used

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# per-model: the head differs by checkpoint, and a shared filename would
# silently overwrite one model's head with another's
# base and instruct checkpoints have DIFFERENT unembeddings, so comparing
# base activations against an instruct head would confound tuning with readout.
CHECKPOINTS = {
    ("granite", "base"):     "ibm-granite/granite-3.0-8b-base",
    ("granite", "instruct"): "ibm-granite/granite-3.0-8b-instruct",
    ("llama",   "base"):     "meta-llama/Llama-3.1-8B",
    ("llama",   "instruct"): "meta-llama/Llama-3.1-8B-Instruct",
    ("mistral", "base"):     "mistralai/Mistral-7B-v0.1",
    ("mistral", "instruct"): "mistralai/Mistral-7B-Instruct-v0.1",
}


def out_path(model: str, variant: str = "instruct"):
    suffix = "" if variant == "instruct" else f"_{variant}"
    return ROOT / "data" / f"lm_head_yesno_{model}{suffix}.npz"

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES", "Yes,", "Yes.", "▁Yes", "▁yes"]
NO_VARIANTS  = ["No", " No", "no", " no", "NO", " NO", "No,", "No.", "▁No", "▁no"]


def find_final_norm(model):
    """Generic lookup across common HF causal-LM architectures (Llama-style,
    which Granite follows): model.model.norm holds the final pre-lm_head norm."""
    for path in ["model.norm", "model.model.norm", "transformer.ln_f"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError(
        "Could not find final norm module -- inspect `model` structure manually "
        "(print(model)) and add its attribute path to find_final_norm()."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite", choices=["granite", "llama", "mistral"])
    parser.add_argument("--variant", default="instruct", choices=["base", "instruct"])
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    device, device_map, dtype, _ = detect_device()
    if args.variant == "instruct":
        model, tokenizer = load_model(args.model, device_map, dtype)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = CHECKPOINTS[(args.model, args.variant)]
        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, device_map=device_map, dtype=dtype).eval()
    log.info(f"[{args.model}] loaded (mem: {mem_used(device)})")

    norm = find_final_norm(model)
    norm_type = type(norm).__name__
    norm_weight = norm.weight.detach().float().cpu().numpy()
    norm_bias = norm.bias.detach().float().cpu().numpy() if getattr(norm, "bias", None) is not None else None
    norm_eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    log.info(f"final norm: type={norm_type}, eps={norm_eps}, has_bias={norm_bias is not None}")

    lm_head = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    log.info(f"lm_head shape: {lm_head.shape}")

    token_strs, token_ids, is_yes, rows = [], [], [], []
    for variants, yes_flag in [(YES_VARIANTS, True), (NO_VARIANTS, False)]:
        for s in variants:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                log.info(f"  skip '{s}': tokenizes to {len(ids)} tokens {ids}")
                continue
            token_strs.append(s)
            token_ids.append(ids[0])
            is_yes.append(yes_flag)
            rows.append(lm_head[ids[0]])

    log.info(f"kept {len(token_strs)} single-token variants: "
             f"{[s for s, y in zip(token_strs, is_yes) if y]} (yes) / "
             f"{[s for s, y in zip(token_strs, is_yes) if not y]} (no)")

    out_path(args.model, args.variant).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path(args.model, args.variant),
        token_strs=np.array(token_strs, dtype=object),
        token_ids=np.array(token_ids, dtype=np.int64),
        is_yes=np.array(is_yes, dtype=bool),
        lm_head_rows=np.stack(rows).astype(np.float32),
        norm_weight=norm_weight.astype(np.float32),
        norm_bias=norm_bias.astype(np.float32) if norm_bias is not None else np.array([]),
        norm_type=norm_type,
        norm_eps=norm_eps,
    )
    log.info(f"saved -> {out_path(args.model, args.variant)}")

    unload_model(args.model, model, tokenizer, device)


if __name__ == "__main__":
    main()