"""
Extract residual-stream activations for pairwise combinations of stigma
identities, so the additivity generalization test can run either on a
specific, interpretable subset of traits (identities="fixed15") or on the
full C(112,2) pairwise sweep across every trait in the template set
(identities="full", the default).

FIXED_IDENTITIES (15): Black, Asian, Muslim, Fundamentalist Christian, Autism
Or Autism Spectrum Disorder, Using A Wheel Chair All The Time, Multiple Body
Piercings, Criminal Record, Lesbian, Illiteracy, Was Raped Previously, Teen
Parent Previously, Infertile, Short, Unattractive.

Note: there is no incarceration-status trait in the template set -- "prison"
is approximated by "Criminal Record", the closest available proxy. "Christian"
maps to "Fundamentalist Christian" (only Christian-specific option available).

For identities="full" (112 traits), C(112,2) = 6,216 unordered pairs (12,432
ordered, both phrasing orders) x 37 patterns = 464,165 forward passes per
model -- 55x the fixed-15 scale (8,362). Two things that don't matter at
fixed-15 scale become load-bearing here:

Storage: naively storing ind1/ind2/base duplicated into every pair-row they
appear in (the fixed-15 script's original scheme) would cost ~1.2-1.5 TB per
model at full scale. Instead, each pattern's contribution is written as a
*deduplicated* shard -- solo-trait vectors stored once per pattern (not once
per pair), base stored once per pattern, and only the combo (pair) vectors
genuinely need one row per ordered pair. pt2_test/eval/activation_io.py
reconstructs the legacy ind1/ind2/combo12/combo21/base wide-row view that the
analysis scripts (additivity_random.py etc.) expect, on read, so none of
their statistical logic needs to change.

Checkpointing: a pattern (not a whole run) is the unit of work. Each
pattern's shard is flushed to disk as soon as that pattern's ~12.5k forward
passes finish, and a shard that already exists on disk is skipped on the next
invocation -- so a killed/preempted run resumes from the last completed
pattern instead of losing everything (the old fixed-15 script accumulated
every scenario for every layer in RAM and wrote once at the very end, which
is fine for 3,885 scenarios but would hold ~150GB in RAM with zero
checkpointing at full scale).

Five conditions per (pair, pattern), same as before:
    ind1     = "who is <stigma1>"
    ind2     = "who is <stigma2>"
    combo12  = "who is <stigma1> and is <stigma2>"
    combo21  = "who is <stigma2> and is <stigma1>"
    base     = the pattern's no-stigma control (identical text regardless of
               stigma pair -- computed once per pattern, reused for all pairs)

combo12/combo21 phrasing is pulled directly from combined_neostigmas.csv
(both orderings already exist as separate rows, correctly grammar-normalized
by pipeline/combined_stigmas.py) rather than re-implemented here. ind1/ind2
vectors are cached per (trait, pattern), since each trait is reused across
every other pair sharing that pattern.

Output: pt2_test/data/activations_random/{model}_pattern{pattern_id}_layer{N}.npz,
one shard per (pattern, layer) -- not one file per pattern holding every
layer -- because a stacked multi-layer array can't be partially read off
disk: an analysis script that processes one layer at a time (all of them do,
to bound RAM) would otherwise re-read every shard's *entire* multi-layer
blob once per layer it looks at, an n_layers-x I/O blowup. Splitting by
layer means a full analysis sweep reads the whole dataset exactly once.
Each file holds:
    singles_vec    (n_identities, d)   solo-trait vectors
    singles_stigma (n_identities,)     trait name per row
    base_vec       (1, d)              no-stigma control vector
    combo_vec      (n_ordered_pairs, d) combo vectors
    combo_stigma1, combo_stigma2 (n_ordered_pairs,)
    pattern_id     scalar
A companion pt2_test/data/activations_random/{model}_pattern{pattern_id}.done
marker file, written only after all of that pattern's layer files are
flushed, is what resume/read-completeness checks key off -- so a run killed
mid-pattern redoes that whole pattern rather than reading a partial one.
Read via pt2_test/eval/activation_io.py, not directly.
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


def load_identities(scope: str, combined: pd.DataFrame) -> list[str]:
    """scope='fixed15' -> the curated 15-trait subset; scope='full' -> every
    single-stigma trait in the template set (112, as of writing)."""
    if scope == "fixed15":
        return list(FIXED_IDENTITIES)
    if scope == "full":
        singles = combined[combined["stigma2"].isna()]
        return singles["stigma1"].tolist()
    raise ValueError(f"Unknown identities scope: {scope!r}")


def all_stigma_pairs(identities: list[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(identities, 2))


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


def _extract(prompt: str, model, tokenizer, dtype: np.dtype) -> dict[int, np.ndarray]:
    """extract_activations() wrapper that casts to the target dtype
    immediately, so a pattern's transient in-RAM footprint is halved (fp16)
    instead of holding a full fp32 copy alongside the cast-down one."""
    return {layer: vec.astype(dtype) for layer, vec in extract_activations(prompt, model, tokenizer).items()}


def extract_pattern(pat_row: pd.Series, identities: list[str], stigma_pairs: list[tuple[str, str]],
                     combined: pd.DataFrame, model, tokenizer, layers: list[int], dtype: np.dtype) -> dict[int, dict]:
    """Run every forward pass needed for one pattern and return
    {layer: deduplicated arrays}, one shard-payload dict per layer -- kept
    per-layer (not stacked across layers) so each on-disk file holds exactly
    one layer's data. That matters for read efficiency: a stacked
    (n_layers, n, d) array can't be partially read off disk, so an analysis
    script that processes one layer at a time (all of them do, to bound RAM)
    would otherwise re-read every shard's full multi-layer blob once per
    layer -- an n_layers-x I/O blowup at full scale (152GB -> ~6TB of reads).
    One file per (pattern, layer) means each layer-pass reads only its own
    slice of the data, exactly once, across the whole analysis run."""
    base_prompt = _apply_swap(str(pat_row["Base Case"]))
    base_by_layer = _extract(base_prompt, model, tokenizer, dtype)

    ind_cache: dict[str, dict[int, np.ndarray]] = {}
    for trait in identities:
        phrase = single_phrase(combined, trait)
        prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", phrase))
        ind_cache[trait] = _extract(prompt, model, tokenizer, dtype)

    combo_stigma1, combo_stigma2 = [], []
    combo_rows: list[dict[int, np.ndarray]] = []
    for s1, s2 in stigma_pairs:
        combo12_phrase = combined[(combined.stigma1 == s1) & (combined.stigma2 == s2)].iloc[0]["With Stigma"]
        combo21_phrase = mirror_phrase(combined, s1, s2)
        combo12_prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", combo12_phrase))
        combo21_prompt = _apply_swap(str(pat_row["Pattern"]).replace("{stigma}", combo21_phrase))

        combo_rows.append(_extract(combo12_prompt, model, tokenizer, dtype))
        combo_stigma1.append(s1); combo_stigma2.append(s2)
        combo_rows.append(_extract(combo21_prompt, model, tokenizer, dtype))
        combo_stigma1.append(s2); combo_stigma2.append(s1)

    combo_stigma1_arr = np.array(combo_stigma1, dtype=object)
    combo_stigma2_arr = np.array(combo_stigma2, dtype=object)
    singles_stigma_arr = np.array(identities, dtype=object)

    by_layer: dict[int, dict] = {}
    for layer in layers:
        by_layer[layer] = {
            "singles_vec": np.stack([ind_cache[t][layer] for t in identities]),
            "singles_stigma": singles_stigma_arr,
            "base_vec": base_by_layer[layer][None, :],
            "combo_vec": np.stack([row[layer] for row in combo_rows]),
            "combo_stigma1": combo_stigma1_arr,
            "combo_stigma2": combo_stigma2_arr,
        }
    return by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite", choices=["granite", "llama", "mistral"])
    parser.add_argument("--identities", default="full", choices=["fixed15", "full"],
                         help="fixed15: curated 15-trait subset. full: all traits in the template set (default).")
    parser.add_argument("--n-patterns", type=int, default=None, help="default: all available patterns")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    args = parser.parse_args()
    dtype = np.float16 if args.dtype == "fp16" else np.float32

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities(args.identities, combined)
    stigma_pairs = all_stigma_pairs(identities)
    patterns = load_patterns(PATTERNS_YES_NO)
    if args.n_patterns:
        patterns = patterns.sample(n=args.n_patterns, random_state=args.seed)

    fwd_per_pattern = 1 + len(identities) + 2 * len(stigma_pairs)
    log.info(f"{len(identities)} identities ({args.identities}) -> {len(stigma_pairs)} pairs x {len(patterns)} patterns "
              f"= {fwd_per_pattern * len(patterns)} forward passes total "
              f"({fwd_per_pattern}/pattern)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _done_marker(pat_idx) -> Path:
        # written last, after every one of that pattern's per-layer files is
        # flushed -- so its presence means the pattern is safe to read, and
        # its absence (even if some per-layer files exist from a killed run)
        # means the whole pattern gets redone, avoiding partial-layer reads.
        return OUT_DIR / f"{args.model}_pattern{pat_idx}.done"

    already_done = [pat_idx for pat_idx in patterns.index if _done_marker(pat_idx).exists()]
    if already_done:
        log.info(f"[{args.model}] resuming -- {len(already_done)}/{len(patterns)} pattern shards already on disk")

    pending = patterns.drop(index=already_done)
    if pending.empty:
        log.info(f"[{args.model}] all pattern shards already done -- nothing to do.")
        return

    device, device_map, model_dtype, _ = detect_device()
    model, tokenizer = load_model(args.model, device_map, model_dtype)
    n_layers = model.config.num_hidden_layers
    layers = list(range(1, n_layers + 1))
    log.info(f"[{args.model}] {n_layers} layers  (mem after load: {mem_used(device)})")

    for i, (pat_idx, pat_row) in enumerate(pending.iterrows()):
        by_layer = extract_pattern(pat_row, identities, stigma_pairs, combined, model, tokenizer, layers, dtype)
        for layer, payload in by_layer.items():
            shard_path = OUT_DIR / f"{args.model}_pattern{pat_idx}_layer{layer}.npz"
            np.savez(shard_path, pattern_id=pat_idx, **payload)
        _done_marker(pat_idx).touch()
        log.info(f"[{args.model}] pattern {pat_idx} done ({i + 1}/{len(pending)} this run, "
                  f"{len(already_done) + i + 1}/{len(patterns)} total)  mem={mem_used(device)}")

    log.info(f"[{args.model}] saved {len(pending)} pattern shards -> {OUT_DIR}")

    unload_model(args.model, model, tokenizer, device)


if __name__ == "__main__":
    main()
