"""
Behavioural yes/no generation for BASE and INSTRUCT checkpoints.

Why this run exists
-------------------
Every strong result in this project rests on generated-answer RATES, but the
base arm of the tuning analysis could only be measured through decoded log-odds,
which demonstrably understates absorption (mistral instruct: -0.008 decoded
against -0.260 behavioural).  That mismatch is the weakest joint in the causal
claim, and it is load-bearing.  This puts both arms on the same measure.

Design constraint that doubles the work
---------------------------------------
The existing instruct generations used a chat template.  Base checkpoints have
none, so base-plain vs instruct-chat would confound tuning with prompt format.
Both variants are therefore regenerated here with an IDENTICAL plain-text
prompt, matching how extract_pairs_base_vs_instruct.py handled the same problem.
Consequence: these results are comparable to each other and to
pairact_*_p2250.npz, but NOT to random_sample_results.csv.

Base models do not reliably answer a question -- they continue text.  The parse
rate is therefore a first-class diagnostic, reported per variant: if base models
yield far fewer parseable yes/no answers than instruct, any rate difference is
partly a parsing artifact rather than a bias difference, and the comparison
needs that caveat attached.  Unparseable generations are recorded as NaN rather
than silently coerced to "no", which would manufacture the effect we are
looking for.

Pairs are the SAME 2250 sampled by extract_pairs_base_vs_instruct.py, so the
behavioural and representational arms cover identical scenarios.
"""
import argparse
import gc
import logging
import os
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
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
load_dotenv(REPO / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.load_models import detect_device
from pipeline.polarity import is_biased, polarity_by_pattern
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from random_sample_activations import load_identities, single_phrase, mirror_phrase
from extract_pairs_base_vs_instruct import PAIRS, PATTERN_IDS, sample_pairs, build

OUT_DIR = ROOT / "data"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

YES = {"yes", "yes.", "yes,", "y"}
NO = {"no", "no.", "no,", "n"}


def parse_answer(text: str):
    """First yes/no token in the continuation, or None if the model didn't answer."""
    t = str(text).strip().lower()
    for tok in t.replace("\n", " ").split():
        w = tok.strip(".,!?:;\"'*")
        if w in {"yes", "y"}:
            return "yes"
        if w in {"no", "n"}:
            return "no"
    return None


@torch.inference_mode()
def generate(model_id, prompts, device_map, dtype, batch, max_new_tokens):
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    mo = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, dtype=dtype).eval()
    out, t0 = [], time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tk(chunk, return_tensors="pt", padding=True).to(mo.device)
        gen = mo.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tk.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        out.extend(tk.batch_decode(new, skip_special_tokens=True))
        if (i // batch) % 20 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(prompts) - done) / max(rate, 1e-9) / 60
            log.info(f"    {done}/{len(prompts)}  {rate:.1f}/s  eta {eta:.0f} min")
    del mo; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--variants", nargs="+", default=["base", "instruct"])
    ap.add_argument("--n-pairs", type=int, default=2250)
    ap.add_argument("--suffix", default="\nAnswer:")
    ap.add_argument("--tag", default="_p2250")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--all-patterns", action="store_true",
                    help="use all 37 yes/no patterns rather than the 6-pattern subset.  "
                         "Reliability scales with PATTERN count -- patterns are the "
                         "resampling unit -- so cut pairs, not patterns: a 6-pattern "
                         "design gave negative split-half reliability.")
    ap.add_argument("--solo-only", action="store_true",
                    help="generate only base + single-identity conditions.  Absorption "
                         "needs ~111 partners per identity to be reliable (600 pairs "
                         "gives ~11 and measured reliability 0.03); solo bias rate is "
                         "reliable at 0.911 and is 8.5%% of the prompts.")
    ap.add_argument("--limit", type=int, default=None,
                    help="generate only the first N prompts -- for timing before committing")
    args = ap.parse_args()

    jobs = [(m, v) for m in args.models for v in args.variants]
    # One checkpoint per process: loading several into one MPS process silently
    # corrupts every model after the first (see base_vs_instruct.py).
    if len(jobs) > 1:
        import subprocess
        for m, v in jobs:
            log.info(f"=== subprocess: {m}/{v} ===")
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--models", m, "--variants", v, "--n-pairs", str(args.n_pairs),
                   "--suffix", args.suffix, "--tag", args.tag, "--seed", str(args.seed),
                   "--max-new-tokens", str(args.max_new_tokens)]
            if args.solo_only:
                cmd += ["--solo-only"]
            if args.all_patterns:
                cmd += ["--all-patterns"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            subprocess.run(cmd, cwd=REPO, check=True)
        return

    if (tok := os.getenv("HF_TOKEN")):
        login(tok)
    model, variant = args.models[0], args.variants[0]
    model_id = PAIRS[model][0 if variant == "base" else 1]

    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities("full", combined)
    allp = load_patterns(PATTERNS_YES_NO)
    patterns = allp if args.all_patterns else allp.loc[PATTERN_IDS]
    picked = sample_pairs(args.n_pairs, args.seed)
    df = build(picked, identities, combined, patterns, args.suffix)
    if args.solo_only:
        df = df[df.condition.isin(["base", "single"])].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log.info(f"[{model}/{variant}] {len(df)} prompts -> {model_id}")

    device, device_map, dtype, batch = detect_device()
    texts = generate(model_id, df.prompt.tolist(), device_map, dtype, batch,
                     args.max_new_tokens)
    df["raw"] = texts
    df["model_answer"] = [parse_answer(t) for t in texts]
    pol = polarity_by_pattern()
    df["biased_answer"] = df.pattern_id.map(pol)
    df["biased"] = [is_biased(a, b) if a is not None else np.nan
                    for a, b in zip(df.model_answer, df.biased_answer)]
    df["model"] = model; df["variant"] = variant

    parse_rate = df.model_answer.notna().mean()
    log.info(f"[{model}/{variant}] parse rate {parse_rate:.3f}"
             + ("  <- LOW: rate comparisons need this caveat" if parse_rate < 0.9 else ""))
    p = OUT_DIR / f"genbi_{model}_{variant}{args.tag}.csv"
    df.to_csv(p, index=False)
    log.info(f"[{model}/{variant}] saved -> {p}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
