"""
Extract residual-stream activations for a random sample of stigma-pair
combinations (not just Black/Lesbian), to test whether the additive /
non-additive pattern found for that one pair generalizes across many
different identity/trait combinations.

Design: sample N_STIGMAS pairs (stigma1, stigma2) from
data/templates/combined_neostigmas.csv, and N_PATTERNS patterns from
patterns_list_yes_no.csv -- the SAME patterns are reused across every stigma
pair, so scenario content is held fixed and only the identity varies.

Five conditions per (pair, pattern):
    ind1     = "who is <stigma1>"
    ind2     = "who is <stigma2>"
    combo12  = "who is <stigma1> and is <stigma2>"
    combo21  = "who is <stigma2> and is <stigma1>"
    base     = the pattern's no-stigma control (identical text regardless of
               stigma pair -- computed once per pattern, reused for all pairs)

combo12/combo21 phrasing is pulled directly from combined_neostigmas.csv
(both orderings already exist as separate rows, correctly grammar-normalized
by pipeline/combined_stigmas.py) rather than re-implemented here.

Output: pt2_test/data/activations_random/{model}_layer{N}.npz, same shape
convention as extract_activations.py (condition -> (n_scenarios, d) array +
scenario_ids), with generic condition names so pt2_test/eval/additivity.py's
logic applies with a key rename, not a rewrite.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "activations_random"

N_STIGMAS  = 100
N_PATTERNS = 3
SEED       = 0

CONDITIONS = ["ind1", "ind2", "combo12", "combo21", "base"]


def sample_stigma_pairs(n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(COMBINED_PATH)
    paired = df.dropna(subset=["stigma1", "stigma2"]).reset_index(drop=True)
    return paired.sample(n=n, random_state=seed).reset_index(drop=True)


def mirror_phrase(df: pd.DataFrame, s1: str, s2: str) -> str:
    """combo phrase for (s2, s1) order -- already exists as its own row."""
    match = df[(df.stigma1 == s2) & (df.stigma2 == s1)]
    if match.empty:
        raise ValueError(f"No mirror row for ({s2}, {s1}) in {COMBINED_PATH}")
    return match.iloc[0]["With Stigma"]


def single_phrase(df: pd.DataFrame, stigma: str) -> str:
    match = df[(df.stigma1 == stigma) & (df.stigma2.isna())]
    if match.empty:
        raise ValueError(f"No single-stigma row for '{stigma}' in {COMBINED_PATH}")
    return match.iloc[0]["With Stigma"]


def build_prompts(pattern_row: pd.Series, ind1_phrase: str, ind2_phrase: str,
                   combo12_phrase: str, combo21_phrase: str) -> dict:
    """'original'-style prompts: bare scenario + stigma phrase inserted,
    matching pipeline/prompt.py's make_prompts 'original' key."""
    template = str(pattern_row["Pattern"])
    return {
        "ind1":    _apply_swap(template.replace("{stigma}", ind1_phrase)),
        "ind2":    _apply_swap(template.replace("{stigma}", ind2_phrase)),
        "combo12": _apply_swap(template.replace("{stigma}", combo12_phrase)),
        "combo21": _apply_swap(template.replace("{stigma}", combo21_phrase)),
        "base":    _apply_swap(str(pattern_row["Base Case"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite", choices=["granite", "llama", "mistral"])
    parser.add_argument("--n-stigmas", type=int, default=N_STIGMAS)
    parser.add_argument("--n-patterns", type=int, default=N_PATTERNS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    combined = pd.read_csv(COMBINED_PATH)
    stigma_pairs = sample_stigma_pairs(args.n_stigmas, args.seed)
    patterns = load_patterns(PATTERNS_YES_NO).sample(n=args.n_patterns, random_state=args.seed)
    log.info(f"Sampled {len(stigma_pairs)} stigma pairs x {len(patterns)} patterns "
             f"= {len(stigma_pairs) * len(patterns)} scenarios")

    device, device_map, dtype, _ = detect_device()
    model, tokenizer = load_model(args.model, device_map, dtype)
    n_layers = model.config.num_hidden_layers
    log.info(f"[{args.model}] {n_layers} layers  (mem after load: {mem_used(device)})")

    activations = {layer: {c: [] for c in CONDITIONS} for layer in range(1, n_layers + 1)}
    scenario_ids = []

    # base only depends on the pattern, not the stigma pair -- compute once
    # per pattern and reuse across all sampled stigma pairs.
    base_layer_vecs_by_pattern = {}
    for pat_idx, pat_row in patterns.iterrows():
        base_prompt = _apply_swap(str(pat_row["Base Case"]))
        base_layer_vecs_by_pattern[pat_idx] = extract_activations(base_prompt, model, tokenizer)

    total = len(stigma_pairs) * len(patterns)
    done, errors = 0, 0
    for _, srow in stigma_pairs.iterrows():
        s1, s2 = srow["stigma1"], srow["stigma2"]
        try:
            ind1_phrase = single_phrase(combined, s1)
            ind2_phrase = single_phrase(combined, s2)
            combo12_phrase = srow["With Stigma"]
            combo21_phrase = mirror_phrase(combined, s1, s2)
        except ValueError as exc:
            log.warning(f"Skipping pair ({s1}, {s2}): {exc}")
            errors += len(patterns)
            continue

        for pat_idx, pat_row in patterns.iterrows():
            prompts = build_prompts(pat_row, ind1_phrase, ind2_phrase, combo12_phrase, combo21_phrase)
            layer_vecs = {
                c: (base_layer_vecs_by_pattern[pat_idx] if c == "base"
                    else extract_activations(prompts[c], model, tokenizer))
                for c in CONDITIONS
            }
            for layer in range(1, n_layers + 1):
                for c in CONDITIONS:
                    activations[layer][c].append(layer_vecs[c][layer])
            scenario_ids.append((s1, s2, pat_idx))

            done += 1
            if done % 50 == 0:
                log.info(f"Progress: {done}/{total}  errors={errors}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario_ids_arr = np.array(scenario_ids, dtype=object)
    for layer, by_cond in activations.items():
        payload = {c: np.stack(v) for c, v in by_cond.items()}
        payload["scenario_ids"] = scenario_ids_arr
        np.savez(OUT_DIR / f"{args.model}_layer{layer}.npz", **payload)

    log.info(f"[{args.model}] saved {n_layers} layer files -> {OUT_DIR}  "
             f"({done} scenarios, {errors} skipped)")

    unload_model(args.model, model, tokenizer, device)


if __name__ == "__main__":
    main()