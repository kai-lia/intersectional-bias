"""
Generate SINGLES-only behavioural responses under Positive and Doubt framings.

SocialStigmaQA provides four prompt columns for every pattern: Pattern (Original),
Positively Biasing Version, Doubt Biasing Version, and Base Case.  Every prior
run in this project used only the Original variant, so any claim about how bias
responds to framing rests on nothing.  This script fills the gap: for each of
three instruction-tuned models, it generates responses to every (identity,
pattern) cell under Positive and Doubt framings, matching the format of
`base_vs_instruct_full.csv` (columns pattern_id, condition, identity, model,
variant, delta_yes, yn_mass, plus a new `prompt_style` column).

Design constraints copied from generate_base_vs_instruct.py:
  * Deterministic decoding (do_sample=False), max_new_tokens=6 (yes/no is one
    token; extra headroom for punctuation and casing).
  * Chat-template applied for instruct so outputs match deployed behaviour.
  * Subprocess-per-(model, variant, style) because MPS silently corrupts a
    second model loaded into the same process (see base_vs_instruct.py).
  * Parse rate reported per (model, variant, style) — a low rate flags a
    framing that broke the model's answer format.
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
from pipeline.polarity import is_biased, polarity_by_pattern
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from random_sample_activations import load_identities, single_phrase
from extract_pairs_base_vs_instruct import PAIRS

OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# The four styles the benchmark provides; we skip "original" because it's
# already covered by base_vs_instruct_full.csv, and "base" is separately in
# every run.  Passing --styles adds one at a time.
STYLE_COL = {
    "original": "Pattern",
    "positive": "Positively Biasing Version",
    "doubt":    "Doubt Biasing Version",
    "base":     "Base Case",
}


def parse_answer(text: str):
    t = str(text).strip().lower()
    for tok in t.replace("\n", " ").split():
        w = tok.strip(".,!?:;\"'*")
        if w in {"yes", "y"}: return "yes"
        if w in {"no", "n"}:  return "no"
    return None


def build_prompts(style: str, suffix: str):
    """One row per (pattern, identity).  For 'base' the identity is None
    (template-only prompt) — the column has no {stigma} placeholder."""
    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities("full", combined)
    patterns = load_patterns(PATTERNS_YES_NO)
    col = STYLE_COL[style]
    rows = []
    for pid, pat in patterns[col].items():
        s = _apply_swap(str(pat))
        if style == "base":
            # base has no {stigma} placeholder; one row per pattern
            rows.append({"pattern_id": int(pid), "identity": None,
                         "prompt": s + suffix})
        else:
            for iden in identities:
                phrase = single_phrase(combined, iden)
                filled = _apply_swap(str(pat).replace("{stigma}", phrase))
                rows.append({"pattern_id": int(pid), "identity": iden,
                             "prompt": filled + suffix})
    return pd.DataFrame(rows)


@torch.inference_mode()
def generate(model_id, prompts, device_map, dtype, batch, max_new_tokens):
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    mo = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, dtype=dtype).eval()
    out, t0 = [], time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tk(chunk, return_tensors="pt", padding=True,
                 truncation=True, max_length=3900).to(mo.device)
        gen = mo.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tk.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        out.extend(tk.batch_decode(new, skip_special_tokens=True))
        if (i // batch) % 10 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(prompts) - done) / max(rate, 1e-9) / 60
            log.info(f"    {done}/{len(prompts)}  {rate:.1f}/s  eta {eta:.1f} min")
    del mo; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def apply_chat(tok, prompt):
    if tok.chat_template is None:
        return prompt + "\nAnswer:"
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def one_run(model, variant, style, args):
    if (tok_env := os.getenv("HF_TOKEN")):
        login(tok_env)
    model_id = PAIRS[model][0 if variant == "base" else 1]
    df = build_prompts(style, suffix="")
    tk = AutoTokenizer.from_pretrained(model_id)
    if variant == "instruct":
        df["prompt_final"] = df.prompt.apply(lambda p: apply_chat(tk, p))
    else:
        df["prompt_final"] = df.prompt + "\nAnswer:"

    log.info(f"[{model}/{variant}/{style}] {len(df)} prompts -> {model_id}")
    device, device_map, dtype, batch = detect_device()
    texts = generate(model_id, df.prompt_final.tolist(), device_map, dtype, batch,
                     args.max_new_tokens)
    df["raw"] = texts
    df["model_answer"] = [parse_answer(t) for t in texts]
    df["biased_answer"] = df.pattern_id.map(polarity_by_pattern())
    df["biased"] = [is_biased(a, b) if a is not None else np.nan
                    for a, b in zip(df.model_answer, df.biased_answer)]
    df["model"] = model; df["variant"] = variant; df["prompt_style"] = style
    parse_rate = df.model_answer.notna().mean()
    log.info(f"[{model}/{variant}/{style}] parse rate {parse_rate:.1%}"
             + ("  <- LOW" if parse_rate < 0.8 else ""))
    p = OUT_DIR / f"framings_{model}_{variant}_{style}.csv"
    df.to_csv(p, index=False)
    log.info(f"[{model}/{variant}/{style}] saved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--variants", nargs="+", default=["instruct"],
                    help="instruct is the priority; base saturates on many patterns")
    ap.add_argument("--styles", nargs="+", default=["positive", "doubt"],
                    help="styles to generate (positive, doubt); original + base "
                         "are already elsewhere")
    ap.add_argument("--max-new-tokens", type=int, default=6)
    args = ap.parse_args()

    jobs = [(m, v, s) for m in args.models for v in args.variants for s in args.styles]
    if len(jobs) > 1:
        # subprocess-per-(model, variant, style) — MPS safety
        for m, v, s in jobs:
            log.info(f"=== subprocess: {m}/{v}/{s} ===")
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--models", m, "--variants", v, "--styles", s,
                            "--max-new-tokens", str(args.max_new_tokens)],
                           cwd=REPO, check=True)
        return
    one_run(args.models[0], args.variants[0], args.styles[0], args)


if __name__ == "__main__":
    main()
