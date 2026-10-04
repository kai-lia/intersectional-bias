"""
Pilot: Granite only, all layers, 15 stratified identities, all 210 ordered
pairs among them, 37 templates x 4 wordings (original + 3 paraphrases).

Purpose: diagnostic — does every-layer resolution tell us anything the
every-5th-layer sweep doesn't? Decide the full-scale wordings/layers/scope
after seeing this pilot's outputs.

Output: pt2_test/data/activations_paraphrase_pilot/{model}_pattern{P}_wording{W}_layer{L}.npz
One npz per (pattern, wording, layer) so analyses that compare across wordings
within a layer don't need to read the full 40-layer blob.

Scope:
  model:       granite only
  identities:  15 stratified (5 dangerous + 5 in-between + 5 protected)
  pairs:       C(15,2) = 105 unordered, 210 ordered
  templates:   all 37
  wordings:    4 (original + 3 paraphrases from gn_paraphrases_review.csv)
  layers:      all 40

Per (pattern, wording): 1 base + 15 singles + 210 combos = 226 forward passes
Total: 37 x 4 x 226 = 33,448 forward passes
Est compute (~1-2 pass/sec on MPS): 5-9 hours
Est storage: ~15 GB
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT))
load_dotenv(REPO / ".env")

from pipeline.load_models import detect_device, load_model, unload_model, mem_used
from pipeline.prompt import _apply_swap
from extract_activations import extract_activations

# Clean combined CSV (drops Mental Retardation + fixes Wheelchair spelling)
CLEAN_COMBINED = REPO / "data" / "templates" / "final_clean" / "clean_combined_neostigmas.csv"
# Reviewed paraphrases (37 templates × 4 wordings)
PARAPHRASES = REPO / "data" / "templates" / "final_clean" / "gn_paraphrases_review.csv"

OUT_DIR = ROOT / "data" / "activations_paraphrase_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 15 stratified identities for the pilot — 5 per tier
PILOT_IDENTITIES = [
    # dangerous (5)
    "Sex Offender",
    "Crystal Meth. Use Recreationally",
    "Gang Member Currently",
    "Transgender",
    "Lesbian/Gay/Bisexual/Non-Heterosexual",
    # in-between (5)
    "Muslim",
    "Fundamentalist Christian",
    "Depression Remitted",
    "Breast Cancer Current Avg. Symptoms",
    "Overweight Current Avg. Severity",
    # protected (5)
    "Black",
    "Asian",
    "Latina",
    "Old Age",
    "Homeless",
]


def single_phrase(df: pd.DataFrame, stigma: str) -> str:
    m = df[(df.stigma1 == stigma) & (df.stigma2.isna())]
    if m.empty:
        raise ValueError(f"No single-stigma row for '{stigma}' in {CLEAN_COMBINED}")
    return m.iloc[0]["With Stigma"]


def combo_phrase(df: pd.DataFrame, s1: str, s2: str) -> str:
    m = df[(df.stigma1 == s1) & (df.stigma2 == s2)]
    if m.empty:
        raise ValueError(f"No combo row for ({s1}, {s2}) in {CLEAN_COMBINED}")
    return m.iloc[0]["With Stigma"]


def _extract(prompt: str, model, tokenizer, dtype) -> dict:
    return {layer: vec.astype(dtype) for layer, vec in extract_activations(prompt, model, tokenizer).items()}


def extract_pattern_wording(template_text: str, base_text: str, identities, pairs,
                            combined, model, tokenizer, dtype):
    """One (pattern, wording) sweep: base + 15 singles + 210 combos."""
    base_prompt = _apply_swap(base_text)
    base_by_layer = _extract(base_prompt, model, tokenizer, dtype)

    ind_cache = {}
    for trait in identities:
        phrase = single_phrase(combined, trait)
        prompt = _apply_swap(template_text.replace("{stigma}", phrase))
        ind_cache[trait] = _extract(prompt, model, tokenizer, dtype)

    combo_s1, combo_s2 = [], []
    combo_rows = []
    for i, s1 in enumerate(identities):
        for s2 in identities[i+1:]:
            # ordering (s1, s2)
            prompt12 = _apply_swap(template_text.replace("{stigma}", combo_phrase(combined, s1, s2)))
            combo_rows.append(_extract(prompt12, model, tokenizer, dtype))
            combo_s1.append(s1); combo_s2.append(s2)
            # ordering (s2, s1)
            prompt21 = _apply_swap(template_text.replace("{stigma}", combo_phrase(combined, s2, s1)))
            combo_rows.append(_extract(prompt21, model, tokenizer, dtype))
            combo_s1.append(s2); combo_s2.append(s1)

    return base_by_layer, ind_cache, combo_rows, combo_s1, combo_s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="granite", choices=["granite", "llama", "mistral"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()
    dtype = np.float16 if args.dtype == "fp16" else np.float32

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN."); sys.exit(1)
    login(token)

    combined = pd.read_csv(CLEAN_COMBINED)
    paras = pd.read_csv(PARAPHRASES).rename(columns={"itpattern_id": "pattern_id"})
    paras["pattern_id"] = paras.pattern_id.astype(int)

    # Verify all pilot identities exist in the clean combined
    singles = set(combined[combined.stigma2.isna()].stigma1)
    missing = [i for i in PILOT_IDENTITIES if i not in singles]
    if missing:
        log.error(f"Pilot identities missing from clean_combined_neostigmas.csv: {missing}")
        sys.exit(1)

    identities = list(PILOT_IDENTITIES)
    n_pairs = len(identities) * (len(identities) - 1)  # ordered
    log.info(f"[{args.model}] pilot scope: {len(identities)} identities, {n_pairs} ordered pairs, "
             f"{len(paras)} templates x 4 wordings")
    passes_per = 1 + len(identities) + n_pairs
    total = len(paras) * 4 * passes_per
    log.info(f"[{args.model}] forward passes/pattern/wording = {passes_per}  total = {total:,}")

    device, device_map, model_dtype, _ = detect_device()
    model, tokenizer = load_model(args.model, device_map, model_dtype)
    n_layers = model.config.num_hidden_layers
    layers = list(range(1, n_layers + 1))
    log.info(f"[{args.model}] loaded, {n_layers} layers, mem={mem_used(device)}")

    def shard_path(pat_id, wording_id, layer):
        return OUT_DIR / f"{args.model}_pattern{pat_id}_wording{wording_id}_layer{layer}.npz"

    def done_marker(pat_id, wording_id):
        return OUT_DIR / f"{args.model}_pattern{pat_id}_wording{wording_id}.done"

    paras_sorted = paras.sort_values("pattern_id").reset_index(drop=True)
    wording_cols = ["original", "paraphrase_1", "paraphrase_2", "paraphrase_3"]

    jobs_done = 0
    jobs_total = len(paras_sorted) * len(wording_cols)
    for _, row in paras_sorted.iterrows():
        pat_id = int(row.pattern_id)
        for w_id, col in enumerate(wording_cols):
            jobs_done += 1
            if done_marker(pat_id, w_id).exists():
                log.info(f"[{args.model}] pat{pat_id} wording{w_id} ({col}) already done — skip  [{jobs_done}/{jobs_total}]")
                continue
            tpl = str(row[col])
            # base case = template with {stigma} removed — use "a person" as neutral fill
            # (matches the original pipeline's base = pattern with no identity)
            # But we don't have the Base Case for paraphrases, so use the paraphrase itself
            # with {stigma} removed (just "someone")
            base_text = tpl.replace("someone {stigma}", "someone").replace("{stigma}", "")
            # clean up double spaces / stray words
            base_text = " ".join(base_text.split())
            log.info(f"[{args.model}] pat{pat_id} wording{w_id} ({col}) starting  [{jobs_done}/{jobs_total}]  mem={mem_used(device)}")
            base_by_layer, ind_cache, combo_rows, combo_s1, combo_s2 = \
                extract_pattern_wording(tpl, base_text, identities, None, combined, model, tokenizer, dtype)
            singles_arr = np.array(identities, dtype=object)
            s1_arr = np.array(combo_s1, dtype=object)
            s2_arr = np.array(combo_s2, dtype=object)
            for layer in layers:
                np.savez(
                    shard_path(pat_id, w_id, layer),
                    pattern_id=pat_id, wording_id=w_id,
                    singles_vec=np.stack([ind_cache[t][layer] for t in identities]),
                    singles_stigma=singles_arr,
                    base_vec=base_by_layer[layer][None, :],
                    combo_vec=np.stack([r[layer] for r in combo_rows]),
                    combo_stigma1=s1_arr, combo_stigma2=s2_arr,
                )
            done_marker(pat_id, w_id).touch()
            log.info(f"[{args.model}] pat{pat_id} wording{w_id} ({col}) DONE  mem={mem_used(device)}")

    log.info(f"[{args.model}] pilot complete -> {OUT_DIR}")
    unload_model(args.model, model, tokenizer, device)


if __name__ == "__main__":
    main()
