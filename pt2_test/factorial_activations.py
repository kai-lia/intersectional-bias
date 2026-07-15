"""
Residual-stream activations for the same crossed factorial design as
factorial_sample_generation.py (5 race levels x 4 orientation levels, both
phrasing orders) -- single forward passes, no generation, for the
vector-arithmetic additivity/CKA analysis (as opposed to that script's real
generation, which is for the MAIHDA/behavioral analysis).

Maps onto the same ind1/ind2/combo12/combo21/base schema as
random_sample_activations.py (race -> ind1, orientation -> ind2), so every
existing analysis script (additivity.py, additivity_random.py,
additivity_random_scenarios.py, cka_sweep.py, cka_delta_sweep.py) works on
this data unchanged -- just point ACT_DIR at pt2_test/data/activations_factorial.

race_only and orientation_only vectors are cached per (trait, pattern), since
each of the 5 race levels is reused across 4 orientation partners and vice
versa -- avoids recomputing the same forward pass 4-5x.
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

load_dotenv()

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from pipeline.load_models import detect_device, load_model, unload_model, mem_used
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from extract_activations import extract_activations
from factorial_sample_generation import RACE_LEVELS, ORIENTATION_LEVELS, single_phrase, combo_phrase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "activations_factorial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite", choices=["granite", "llama", "mistral"])
    parser.add_argument("--n-patterns", type=int, default=None, help="default: all available patterns")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    combined = pd.read_csv(COMBINED_PATH)
    patterns = load_patterns(PATTERNS_YES_NO)
    if args.n_patterns:
        patterns = patterns.sample(n=args.n_patterns, random_state=args.seed)
    log.info(f"{len(RACE_LEVELS)} race x {len(ORIENTATION_LEVELS)} orientation "
             f"= {len(RACE_LEVELS) * len(ORIENTATION_LEVELS)} pairs x {len(patterns)} patterns")

    device, device_map, dtype, _ = detect_device()
    model, tokenizer = load_model(args.model, device_map, dtype)
    n_layers = model.config.num_hidden_layers
    log.info(f"[{args.model}] {n_layers} layers  (mem after load: {mem_used(device)})")

    activations = {layer: {c: [] for c in ["ind1", "ind2", "combo12", "combo21", "base"]}
                   for layer in range(1, n_layers + 1)}
    scenario_ids = []

    total_pairs = len(RACE_LEVELS) * len(ORIENTATION_LEVELS) * len(patterns)
    done = 0
    for pat_idx, pat_row in patterns.iterrows():
        base_prompt = _apply_swap(str(pat_row["Base Case"]))
        base_vecs = extract_activations(base_prompt, model, tokenizer)

        race_cache: dict = {}
        orientation_cache: dict = {}
        for r in RACE_LEVELS:
            prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", single_phrase(combined, r)))
            race_cache[r] = extract_activations(prompt, model, tokenizer)
        for o in ORIENTATION_LEVELS:
            prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", single_phrase(combined, o)))
            orientation_cache[o] = extract_activations(prompt, model, tokenizer)

        for r in RACE_LEVELS:
            for o in ORIENTATION_LEVELS:
                combo12_prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", combo_phrase(combined, r, o)))
                combo21_prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", combo_phrase(combined, o, r)))
                layer_vecs = {
                    "ind1": race_cache[r],
                    "ind2": orientation_cache[o],
                    "combo12": extract_activations(combo12_prompt, model, tokenizer),
                    "combo21": extract_activations(combo21_prompt, model, tokenizer),
                    "base": base_vecs,
                }
                for layer in range(1, n_layers + 1):
                    for c in ["ind1", "ind2", "combo12", "combo21", "base"]:
                        activations[layer][c].append(layer_vecs[c][layer])
                scenario_ids.append((r, o, pat_idx))

                done += 1
                if done % 50 == 0:
                    log.info(f"Progress: {done}/{total_pairs}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario_ids_arr = np.array(scenario_ids, dtype=object)
    for layer, by_cond in activations.items():
        payload = {c: np.stack(v) for c, v in by_cond.items()}
        payload["scenario_ids"] = scenario_ids_arr
        np.savez(OUT_DIR / f"{args.model}_layer{layer}.npz", **payload)

    log.info(f"[{args.model}] saved {n_layers} layer files -> {OUT_DIR}  ({done} scenarios)")
    unload_model(args.model, model, tokenizer, device)


if __name__ == "__main__":
    main()