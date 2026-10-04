"""
Regenerate race × criminal-legal responses for annotation.  INSTRUCT-only.

Why instruct-only
-----------------
The annotation exists to characterize what real users see.  Users interact
with instruct checkpoints via a chat interface; base checkpoints are not
deployed as user-facing products.  A pilot on granite showed base models loop
badly on some prompts (one healthcare-refuse item generated 9418 chars of
verbatim repetition until hitting the token cap, taking ~5 min alone) because
base checkpoints lack question-answering EOS training.  Annotating that mix
of clean and looped base outputs would introduce more noise than the base-vs-
instruct comparison would recover -- and the paper's quantitative base-vs-
instruct comparison already lives in the readout analyses.

The deployed prompt format is used: each model's chat template is applied via
tokenizer.apply_chat_template with the SocialStigmaQA prompt as the user
message.  Generations are deterministic (do_sample=False), so outputs are
reproducible.

Length ceiling
--------------
Previous behavioural runs used max_new_tokens near 300, truncating ~50% of
llama outputs mid-sentence.  This run uses max_new_tokens=2048, safely below
granite's 4096 context limit and above any observed complete-response length,
so all generations terminate on EOS rather than at the token cap.

Cost
----
~11 pairs x 8 templates x 2 orderings x 3 models ~ 528 items.  At ~8s per
generation for instruct on MPS, ~1 hour per model sequentially, ~3 hours
total.  Subprocess-per-model handling mirrors generate_base_vs_instruct to
avoid the silent MPS-model-corruption bug documented in base_vs_instruct.py.
"""
import argparse
import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT))
load_dotenv(REPO / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.load_models import detect_device
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from random_sample_activations import single_phrase
from extract_pairs_base_vs_instruct import PAIRS

OUT_DIR = ROOT / "data" / "annotation_gen"
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# The 10 race x criminal pairs with the highest mean adjusted residual across
# the three models, from data/eval/{model}_deviation_screen.csv.  Sex Offender
# dominates by construction of the data; two Gang Member and one Drug Dealing
# pair are added so the annotation isn't a single-criminal-identity finding.
TOP_PAIRS = [
    ("Asian American",           "Sex Offender"),
    ("Latina/Latino",            "Sex Offender"),
    ("South Asian",              "Sex Offender"),
    ("Middle Eastern",           "Sex Offender"),
    ("Multiracial",              "Sex Offender"),
    ("Black/African American",   "Sex Offender"),
    ("African American",         "Sex Offender"),
    ("Latino",                   "Sex Offender"),
    ("South Asian",              "Gang Member Currently"),
    ("Middle Eastern",           "Gang Member Currently"),
    ("Black/African American",   "Drug Dealing"),
]

# 8 templates: 2 yes-harmful risk-relevant, 2 yes-harmful not-relevant,
# 2 no-harmful risk-relevant, 2 no-harmful not-relevant.  Choosing this way
# so annotations can be split by both polarity and by scenario type.
# Template index -> (polarity, risk-relevant) as classified in the scenario-
# split analysis.
TEMPLATES = {
    0:  ("yes", True),   # kids playing at household  (child safety, YH)
    1:  ("yes", True),   # kids in danger              (child safety, YH)
    2:  ("no",  True),   # hire candidate              (employment, NH)
    3:  ("no",  True),   # renew lease                 (housing, NH)
    15: ("no",  True),   # hire babysitter             (child safety, NH)
    25: ("yes", False),  # stop listening to doctor    (healthcare, YH, NR)
    27: ("yes", False),  # give poor tip               (social, YH, NR)
    32: ("yes", False),  # refuse healthcare           (healthcare, YH, NR)
}


def build_prompts():
    """One row per (pair, template, ordering).  Deterministic order.

    Combo phrases are read from combined_neostigmas.csv rather than assembled
    inline.  The table has separate rows for (A,B) and (B,A) with correctly
    conjugated 'With Stigma' strings ("who is X and is a Y" vs "who is a Y
    and is X"), which we cannot reproduce reliably from single phrases alone
    (verb agreement, article insertion, etc.)."""
    combined = pd.read_csv(COMBINED_PATH)
    allp = load_patterns(PATTERNS_YES_NO)
    def combo(a, b):
        r = combined[(combined.stigma1 == a) & (combined.stigma2 == b)]
        if r.empty:
            raise ValueError(f"No combined row for ({a}, {b})")
        return r.iloc[0]["With Stigma"]
    rows = []
    for pid in TEMPLATES:
        pattern = allp.loc[pid, "Pattern"]
        base_case = allp.loc[pid, "Base Case"]
        polarity, rr = TEMPLATES[pid]
        for a, b in TOP_PAIRS:
            phrase_a = single_phrase(combined, a)
            phrase_b = single_phrase(combined, b)
            for ord_tag, s1, s2 in [("combo12", a, b), ("combo21", b, a)]:
                stigma = combo(s1, s2)
                rows.append({
                    "pair_id":     f"{a}||{b}",
                    "identity_a":  a,  "identity_b":  b,
                    "pattern_id":  pid, "ordering":   ord_tag,
                    "polarity":    polarity,   "risk_relevant": rr,
                    "prompt_plain": _apply_swap(str(pattern).replace("{stigma}", stigma)),
                    "base_case":    _apply_swap(str(base_case)),
                    "phrase_a":    phrase_a,   "phrase_b":     phrase_b,
                })
    return pd.DataFrame(rows)


def prompt_for_variant(row, tokenizer, variant):
    """Chat-formatted for instruct, plain-with-answer-suffix for base."""
    plain = row["prompt_plain"]
    if variant == "instruct" and tokenizer.chat_template is not None:
        # user message = the SocialStigmaQA prompt
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": plain}],
            tokenize=False, add_generation_prompt=True,
        )
    # base -- no chat template -- append answer suffix so the model continues
    return plain + "\nAnswer:"


@torch.inference_mode()
def generate(model_id, prompts, device_map, dtype, batch, max_new_tokens):
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    mo = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, dtype=dtype).eval()
    out, t0 = [], time.time()
    # tokenize with the SAME (already-templated) strings so the split point is unambiguous
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tk(chunk, return_tensors="pt", padding=True,
                 truncation=True, max_length=3900).to(mo.device)
        gen = mo.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tk.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        out.extend(tk.batch_decode(new, skip_special_tokens=True))
        done = i + len(chunk)
        rate = done / max(time.time() - t0, 1e-9)
        eta_min = (len(prompts) - done) / max(rate, 1e-9) / 60
        log.info(f"  {done}/{len(prompts)}  {rate:.2f}/s  eta {eta_min:.1f} min")
    del mo; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out, tk


def one_run(model, variant, args):
    """One model, one variant, one process (mandatory for MPS safety)."""
    if (tok := os.getenv("HF_TOKEN")):
        login(tok)
    model_id = PAIRS[model][0 if variant == "base" else 1]
    df = build_prompts()
    tk = AutoTokenizer.from_pretrained(model_id)
    df["prompt"] = df.apply(lambda r: prompt_for_variant(r, tk, variant), axis=1)
    log.info(f"[{model}/{variant}] {len(df)} prompts -> {model_id}")
    device, device_map, dtype, batch = detect_device()
    texts, _ = generate(model_id, df.prompt.tolist(), device_map, dtype, batch,
                        args.max_new_tokens)
    df["response"] = texts
    df["model"] = model; df["variant"] = variant
    # truncation diagnostic
    def truncated(t):
        return not str(t).rstrip().endswith((".", "!", "?", "\"", ")", "*", "}"))
    df["maybe_truncated"] = df.response.map(truncated)
    trunc_rate = df.maybe_truncated.mean()
    log.info(f"[{model}/{variant}] apparent truncation rate: {trunc_rate:.1%}"
             + ("  <- unexpectedly high" if trunc_rate > 0.10 else ""))
    p = OUT_DIR / f"racecrim_{model}_{variant}.csv"
    df.to_csv(p, index=False)
    log.info(f"[{model}/{variant}] saved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--variants", nargs="+", default=["instruct"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    jobs = [(m, v) for m in args.models for v in args.variants]
    if len(jobs) > 1:
        # subprocess per (model, variant) -- MPS silently corrupts a second
        # loaded checkpoint in the same process
        for m, v in jobs:
            log.info(f"=== subprocess: {m}/{v} ===")
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--models", m, "--variants", v,
                            "--max-new-tokens", str(args.max_new_tokens)],
                           cwd=REPO, check=True)
        return
    one_run(args.models[0], args.variants[0], args)


if __name__ == "__main__":
    main()
