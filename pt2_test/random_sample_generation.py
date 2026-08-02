"""
Real generation (yes/no + reasoning) for the same identity design as
random_sample_activations.py (--identities fixed15 or full, matching that
script's flag) -- connects the representational non-additivity findings
(additivity_random.py / additivity_random_scenarios.py) to actual model
behavior: does the model's yes/no decision on a combo scenario deviate from
what you'd predict by additively combining the two individual traits' bias
rates, and does that behavioral deviation track the representational
non-additive fraction?

Conditions per pattern (fixed15: 226, full: 12,545):
    individual (N)      "who is <trait>"
    combo12    (C(N,2)) "who is <stigma1> and is <stigma2>"
    combo21    (C(N,2)) "who is <stigma2> and is <stigma1>"
    base       (1)      no-stigma control

Already resumable/incremental (appends to CSV, skips completed_keys on
restart), so this scales to the full-112 design (~464k rows/model) with no
architectural change -- unlike random_sample_activations.py's raw-activation
output, per-row generation results are cheap enough to just append to CSV.

Output: pt2_test/data/random_sample_results.csv, same column convention as
factorial_results.csv (model_answer, Reasoning, biased) so filter_reasoning.py's
degeneracy classify() logic applies unchanged.
"""
import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

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
from load_models_reasoning import RUNNERS
from random_sample_activations import load_identities, all_stigma_pairs, mirror_phrase, single_phrase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUTPUT_CSV = ROOT / "data" / "random_sample_results.csv"


def build_conditions(combined: pd.DataFrame, identities: list[str]) -> list[dict]:
    """Returns a list of {condition, stigma1, stigma2, phrase} dicts --
    fixed across all patterns, since these only depend on the trait pairing."""
    conditions = []
    for t in identities:
        conditions.append({"condition": "individual", "stigma1": t, "stigma2": None,
                            "phrase": single_phrase(combined, t)})
    for s1, s2 in all_stigma_pairs(identities):
        combo12_phrase = combined[(combined.stigma1 == s1) & (combined.stigma2 == s2)].iloc[0]["With Stigma"]
        combo21_phrase = mirror_phrase(combined, s1, s2)
        conditions.append({"condition": "combo12", "stigma1": s1, "stigma2": s2, "phrase": combo12_phrase})
        conditions.append({"condition": "combo21", "stigma1": s1, "stigma2": s2, "phrase": combo21_phrase})
    conditions.append({"condition": "base", "stigma1": None, "stigma2": None, "phrase": None})
    return conditions


def build_prompt(pattern_row: pd.Series, condition: dict) -> str:
    if condition["condition"] == "base":
        return _apply_swap(str(pattern_row["Base Case"]))
    template = str(pattern_row["Pattern"])
    return _apply_swap(template.replace("{stigma}", condition["phrase"]))


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite", choices=["granite", "llama", "mistral"])
    parser.add_argument("--identities", default="full", choices=["fixed15", "full"],
                         help="fixed15: curated 15-trait subset. full: all traits in the template set (default).")
    parser.add_argument("--n-patterns", type=int, default=None, help="default: all available patterns")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities(args.identities, combined)
    conditions = build_conditions(combined, identities)

    patterns = load_patterns(PATTERNS_YES_NO)
    if args.n_patterns:
        patterns = patterns.sample(n=args.n_patterns, random_state=args.seed)
    log.info(f"{len(conditions)} conditions x {len(patterns)} patterns "
             f"= {len(conditions) * len(patterns)} total prompts")

    write_header = not OUTPUT_CSV.exists()
    completed_keys: set = set()
    if not write_header:
        existing = pd.read_csv(OUTPUT_CSV)
        required_cols = {"pattern_id", "condition", "stigma1", "stigma2", "model"}
        missing = required_cols - set(existing.columns)
        if missing:
            log.error(
                f"{OUTPUT_CSV} exists but is missing columns {missing} -- "
                f"looks like a stale/incompatible file. Rename or delete it to start fresh."
            )
            sys.exit(1)
        for _, r in existing.iterrows():
            completed_keys.add((r["pattern_id"], r["condition"], r["stigma1"], r["stigma2"], r["model"]))
        log.info(f"Resuming -- {len(completed_keys)} rows already done.")

    work = []
    for pat_idx, pat_row in patterns.iterrows():
        for cond in conditions:
            key = (pat_idx, cond["condition"], cond["stigma1"], cond["stigma2"], args.model)
            if key in completed_keys:
                continue
            work.append((pat_idx, pat_row, cond))

    if not work:
        log.info("All prompts already done -- nothing to do.")
        return

    device, device_map, dtype, batch_size = detect_device()
    model, tokenizer = load_model(args.model, device_map, dtype)
    runner = RUNNERS[args.model]
    log.info(f"[{args.model}] {len(work)} prompts to run (mem after load: {mem_used(device)})")

    work.sort(key=lambda item: len(build_prompt(item[1], item[2])))

    csv_buffer: list[dict] = []
    done, errors = 0, 0

    def flush_buffer(f):
        nonlocal write_header
        if not csv_buffer:
            return
        pd.DataFrame(csv_buffer).to_csv(f, header=write_header, index=False)
        write_header = False
        f.flush()
        csv_buffer.clear()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "a", newline="") as f:
        for batch in _chunks(work, batch_size):
            texts = [build_prompt(pat_row, cond) for _, pat_row, cond in batch]

            try:
                answers = runner(texts, model, tokenizer)
            except Exception as exc:
                errors += len(batch)
                log.error(f"Batch error ({len(batch)} prompts): {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
                answers = [("error", "")] * len(batch)

            for (pat_idx, pat_row, cond), (answer, reasoning) in zip(batch, answers):
                csv_buffer.append({
                    "pattern_id": pat_idx, "condition": cond["condition"],
                    "stigma1": cond["stigma1"], "stigma2": cond["stigma2"],
                    "stigma_phrase": cond["phrase"], "prompt": build_prompt(pat_row, cond),
                    "model": args.model, "model_answer": answer, "Reasoning": reasoning,
                    "biased": 1 if answer == "yes" else 0,
                })
                done += 1

            if len(csv_buffer) >= 50:
                flush_buffer(f)
            if done % 200 == 0:
                log.info(f"Progress: {done}/{len(work)}  errors={errors}")

        flush_buffer(f)

    log.info(f"Done. {done} rows written, {errors} errors -> {OUTPUT_CSV}")
    unload_model(args.model, model, tokenizer, device)


if __name__ == "__main__":
    main()
