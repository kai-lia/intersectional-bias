"""
Base vs instruct comparison -- the only design that can attribute anything in
this project to instruction/safety tuning.

Every "safety training does X" claim made from the instruct-only data is
inference, not evidence.  This probes the matched base checkpoints on the same
prompts and measures what tuning actually changed.

Design note (matters for interpretation)
----------------------------------------
The instruct dataset elsewhere in this repo was built with
`apply_chat_template(..., add_generation_prompt=True)`.  Base checkpoints have
no chat template, so reusing that data would confound "tuning" with "prompt
format".  This script therefore re-probes BOTH variants with an identical
plain-text prompt, so the model weights are the only thing that differs.
Consequence: the instruct numbers here will NOT match the chat-template numbers
elsewhere, and should not be compared across the two.

Measures, at the final prompt token, from the raw next-token logits:
    delta      = logsumexp(yes-variant logits) - logsumexp(no-variant logits)
                 (a genuine conditional log odds-ratio; the softmax normaliser
                 cancels in the difference)
    yn_mass    = P(yes-variants) + P(no-variants) over the FULL vocabulary
                 -- validity diagnostic.  A base model may not treat yes/no as
                 a plausible continuation at all; if yn_mass is tiny the delta
                 is a comparison between two irrelevant tokens and the whole
                 comparison for that model is weak.  Always read this first.

Tier 1 (this script): base condition + all individual identities.
    37 + 112*37 = 4181 forward passes per variant.  Enough to test the two
    load-bearing claims -- whether tuning moves the decision margin, and
    whether it suppresses identity sensitivity.
Tier 2 (not run here): combo conditions, for flip rates.  Add --combo-pairs N
    to sample N pairs per pattern.

Output: pt2_test/data/eval/base_vs_instruct{tag}.csv
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))
load_dotenv(REPO / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.load_models import detect_device
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from random_sample_activations import load_identities, single_phrase

OUT_DIR = ROOT / "data" / "eval"

PAIRS = {
    "granite": ("ibm-granite/granite-3.0-8b-base",  "ibm-granite/granite-3.0-8b-instruct"),
    "llama":   ("meta-llama/Llama-3.1-8B",          "meta-llama/Llama-3.1-8B-Instruct"),
    "mistral": ("mistralai/Mistral-7B-v0.1",        "mistralai/Mistral-7B-Instruct-v0.1"),
}

YES = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO  = ["No", " No", "no", " no", "NO", " NO"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def token_ids(tokenizer, strings):
    ids = []
    for s in strings:
        e = tokenizer.encode(s, add_special_tokens=False)
        if len(e) == 1:
            ids.append(e[0])
    return sorted(set(ids))


def build_prompts(identities, combined, patterns):
    """(pattern_id, condition, identity, prompt_text) for base + individuals."""
    rows = []
    for pid, prow in patterns.iterrows():
        rows.append((pid, "base", None, _apply_swap(str(prow["Base Case"]))))
        tmpl = str(prow["Pattern"])
        for t in identities:
            rows.append((pid, "individual", t,
                         _apply_swap(tmpl.replace("{stigma}", single_phrase(combined, t)))))
    return rows


@torch.inference_mode()
def probe(model_id: str, prompts, device_map, dtype, batch: int, suffix: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"          # final real token must be last
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, dtype=dtype)
    model.eval()

    yes_ids, no_ids = token_ids(tokenizer, YES), token_ids(tokenizer, NO)
    log.info(f"  {model_id}: {len(yes_ids)} yes-tokens, {len(no_ids)} no-tokens")

    deltas, masses = [], []
    for i in range(0, len(prompts), batch):
        chunk = [p + suffix for p in prompts[i:i + batch]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        ly = torch.logsumexp(logprobs[:, yes_ids], dim=-1)
        ln = torch.logsumexp(logprobs[:, no_ids], dim=-1)
        deltas.append((ly - ln).cpu().numpy())
        masses.append((ly.exp() + ln.exp()).cpu().numpy())
        if (i // batch) % 20 == 0:
            log.info(f"    {i + len(chunk)}/{len(prompts)}")

    del model
    gc = __import__("gc"); gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return np.concatenate(deltas), np.concatenate(masses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--variants", nargs="+", default=["base", "instruct"])
    ap.add_argument("--identities", default="full", choices=["fixed15", "full"])
    ap.add_argument("--suffix", default="\nAnswer:",
                    help="appended to every prompt so yes/no is the natural continuation; "
                         "identical for base and instruct so it cannot confound the comparison")
    ap.add_argument("--tag", default="_full")
    args = ap.parse_args()

    jobs = [(m, v) for m in args.models for v in args.variants]

    # Loading several checkpoints into one MPS process silently corrupts every
    # model after the first: the same prompts that give a yes/no probability
    # mass of 0.334 in a fresh process give 0.004 when the model is loaded
    # fifth.  Freeing with del + gc.collect() + torch.mps.empty_cache() does not
    # fix it.  So each (model, variant) is run in its own subprocess and the
    # per-job outputs are merged here.
    if len(jobs) > 1:
        import subprocess
        parts = []
        for m, v in jobs:
            sub_tag = f"{args.tag}__{m}_{v}"
            log.info(f"=== subprocess: {m}/{v} ===")
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--models", m, "--variants", v,
                            "--identities", args.identities,
                            "--suffix", args.suffix, "--tag", sub_tag],
                           cwd=REPO, check=True)
            parts.append(pd.read_csv(OUT_DIR / f"base_vs_instruct{sub_tag}.csv"))
        merged = pd.concat(parts, ignore_index=True)
        p = OUT_DIR / f"base_vs_instruct{args.tag}.csv"
        merged.to_csv(p, index=False)
        log.info(f"merged {len(jobs)} jobs -> {p}")
        chk = merged.groupby(["model", "variant"]).yn_mass.median()
        log.info("median yes/no mass per variant (validity gate, want >> 0.01):\n"
                 + chk.round(4).to_string())
        return

    if (tok := os.getenv("HF_TOKEN")):
        login(tok)

    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities(args.identities, combined)
    patterns = load_patterns(PATTERNS_YES_NO)
    spec = build_prompts(identities, combined, patterns)
    prompts = [s[3] for s in spec]
    log.info(f"{len(identities)} identities x {len(patterns)} patterns -> {len(prompts)} prompts/variant")

    device, device_map, dtype, batch = detect_device()
    out = []
    for m in args.models:
        base_id, inst_id = PAIRS[m]
        for variant, mid in [("base", base_id), ("instruct", inst_id)]:
            if variant not in args.variants:
                continue
            log.info(f"[{m}/{variant}] probing {mid}")
            d, mass = probe(mid, prompts, device_map, dtype, batch, args.suffix)
            df = pd.DataFrame(spec, columns=["pattern_id", "condition", "identity", "prompt"])
            df = df.drop(columns=["prompt"])
            df["model"], df["variant"] = m, variant
            df["delta_yes"], df["yn_mass"] = d, mass
            out.append(df)
            p = OUT_DIR / f"base_vs_instruct{args.tag}.csv"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.concat(out, ignore_index=True).to_csv(p, index=False)
            log.info(f"[{m}/{variant}] done  mean|delta|={np.abs(d).mean():.2f}  "
                     f"median yes/no mass={np.median(mass):.4f}  -> {p}")


if __name__ == "__main__":
    main()
