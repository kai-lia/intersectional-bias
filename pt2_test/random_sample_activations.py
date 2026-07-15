"""
Extract residual-stream activations for ALL pairwise combinations of a fixed,
deliberately chosen set of 15 identities (not a random sample) -- so the
additivity generalization test runs on a specific, interpretable set of
traits instead of whatever happens to land in a random draw from the full
112-trait template list.

FIXED_IDENTITIES (15): Black, Asian, Muslim, Fundamentalist Christian, Autism
Or Autism Spectrum Disorder, Using A Wheel Chair All The Time, Multiple Body
Piercings, Criminal Record, Lesbian, Illiteracy, Was Raped Previously, Teen
Parent Previously, Infertile, Short, Unattractive.

Note: there is no incarceration-status trait in the template set -- "prison"
is approximated by "Criminal Record", the closest available proxy. "Christian"
maps to "Fundamentalist Christian" (only Christian-specific option available).

All C(15,2) = 105 unordered pairs are used, both phrasing orders (210
ordered pairs), across every available pattern by default -- the same
patterns reused across every pair, so scenario content is held fixed and
only the identity varies.

Five conditions per (pair, pattern):
    ind1     = "who is <stigma1>"
    ind2     = "who is <stigma2>"
    combo12  = "who is <stigma1> and is <stigma2>"
    combo21  = "who is <stigma2> and is <stigma1>"
    base     = the pattern's no-stigma control (identical text regardless of
               stigma pair -- computed once per pattern, reused for all pairs)

combo12/combo21 phrasing is pulled directly from combined_neostigmas.csv
(both orderings already exist as separate rows, correctly grammar-normalized
by pipeline/combined_stigmas.py) rather than re-implemented here. ind1/ind2
vectors are cached per (trait, pattern), since each of the 15 traits is
reused across 14 different pairs.

Output: pt2_test/data/activations_random/{model}_layer{N}.npz, same shape
convention as extract_activations.py (condition -> (n_scenarios, d) array +
scenario_ids), with generic condition names so pt2_test/eval/additivity.py's
logic applies with a key rename, not a rewrite.
"""
import argparse
import itertools
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

FIXED_IDENTITIES = [
    "Black", "Asian", "Muslim", "Fundamentalist Christian",
    "Autism Or Autism Spectrum Disorder", "Using A Wheel Chair All The Time",
    "Multiple Body Piercings", "Criminal Record", "Lesbian", "Illiteracy",
    "Was Raped Previously", "Teen Parent Previously", "Infertile", "Short", "Unattractive",
]
SEED = 0

CONDITIONS = ["ind1", "ind2", "combo12", "combo21", "base"]


def all_stigma_pairs() -> list[tuple[str, str]]:
    return list(itertools.combinations(FIXED_IDENTITIES, 2))


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
    parser.add_argument("--n-patterns", type=int, default=None, help="default: all available patterns")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    combined = pd.read_csv(COMBINED_PATH)
    stigma_pairs = all_stigma_pairs()
    patterns = load_patterns(PATTERNS_YES_NO)
    if args.n_patterns:
        patterns = patterns.sample(n=args.n_patterns, random_state=args.seed)
    log.info(f"{len(FIXED_IDENTITIES)} fixed identities -> {len(stigma_pairs)} pairs x {len(patterns)} patterns "
             f"= {len(stigma_pairs) * len(patterns)} scenarios")

    device, device_map, dtype, _ = detect_device()
    model, tokenizer = load_model(args.model, device_map, dtype)
    n_layers = model.config.num_hidden_layers
    log.info(f"[{args.model}] {n_layers} layers  (mem after load: {mem_used(device)})")

    activations = {layer: {c: [] for c in CONDITIONS} for layer in range(1, n_layers + 1)}
    scenario_ids = []

    total = len(stigma_pairs) * len(patterns)
    done = 0
    for pat_idx, pat_row in patterns.iterrows():
        # base and individual-trait vectors only depend on (trait, pattern),
        # not on which pair is being evaluated -- cache per pattern, since
        # each of the 15 traits is reused across 14 different pairs.
        base_prompt = _apply_swap(str(pat_row["Base Case"]))
        base_vecs = extract_activations(base_prompt, model, tokenizer)

        ind_cache: dict = {}
        for trait in FIXED_IDENTITIES:
            phrase = single_phrase(combined, trait)
            prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", phrase))
            ind_cache[trait] = extract_activations(prompt, model, tokenizer)

        for s1, s2 in stigma_pairs:
            combo12_phrase = combined[(combined.stigma1 == s1) & (combined.stigma2 == s2)].iloc[0]["With Stigma"]
            combo21_phrase = mirror_phrase(combined, s1, s2)
            combo12_prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", combo12_phrase))
            combo21_prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", combo21_phrase))

            layer_vecs = {
                "ind1": ind_cache[s1], "ind2": ind_cache[s2],
                "combo12": extract_activations(combo12_prompt, model, tokenizer),
                "combo21": extract_activations(combo21_prompt, model, tokenizer),
                "base": base_vecs,
            }
            for layer in range(1, n_layers + 1):
                for c in CONDITIONS:
                    activations[layer][c].append(layer_vecs[c][layer])
            scenario_ids.append((s1, s2, pat_idx))

            done += 1
            if done % 200 == 0:
                log.info(f"Progress: {done}/{total}")

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