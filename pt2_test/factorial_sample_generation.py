"""
Real generation (yes/no + reasoning) for a small, deliberately crossed
factorial design -- built for a MAIHDA-style analysis (Option A from the
project discussion), not the vector-arithmetic additivity test.

Why crossed, not random pairs: MAIHDA's additive-main-effects model needs a
dummy variable per trait level. random_sample_activations.py's 100 arbitrary
pairs drew from 91 distinct traits -- ~90 parameters for ~100 strata, badly
underpowered. Here, two small axes (5 race levels x 4 orientation levels,
20 strata) share just 9 dummy variables across many well-populated strata --
the same structure as the original MAIHDA paper's small demographic axes.

Conditions per pattern (50 total):
    race_only        (5)   "who is <race>"
    orientation_only  (4)   "who is <orientation>"
    combo_r_first    (20)   "who is <race> and is <orientation>"
    combo_o_first    (20)   "who is <orientation> and is <race>"
    base              (1)   no-stigma control

Output: pt2_test/data/factorial_results.csv, same columns as pt2_test/data/
results_pt2.csv (model_answer, Reasoning, biased) so filter_reasoning.py's
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RACE_LEVELS        = ["Black", "Latina/Latino", "Asian", "Middle Eastern", "Native American"]
ORIENTATION_LEVELS = ["Lesbian", "Gay", "Bisexual", "Asexual"]

OUTPUT_CSV = ROOT / "data" / "factorial_results.csv"


def single_phrase(df: pd.DataFrame, stigma: str) -> str:
    match = df[(df.stigma1 == stigma) & (df.stigma2.isna())]
    if match.empty:
        raise ValueError(f"No single-stigma row for '{stigma}' in {COMBINED_PATH}")
    return match.iloc[0]["With Stigma"]


def combo_phrase(df: pd.DataFrame, first: str, second: str) -> str:
    """phrase for 'first and is second' -- already exists as its own row."""
    match = df[(df.stigma1 == first) & (df.stigma2 == second)]
    if match.empty:
        raise ValueError(f"No combo row for ({first}, {second}) in {COMBINED_PATH}")
    return match.iloc[0]["With Stigma"]


def build_conditions(combined: pd.DataFrame) -> list[dict]:
    """Returns a list of {condition, race, orientation, phrase} dicts --
    fixed across all patterns, since these only depend on the trait levels."""
    conditions = []
    for r in RACE_LEVELS:
        conditions.append({"condition": "race_only", "race": r, "orientation": None,
                            "phrase": single_phrase(combined, r)})
    for o in ORIENTATION_LEVELS:
        conditions.append({"condition": "orientation_only", "race": None, "orientation": o,
                            "phrase": single_phrase(combined, o)})
    for r in RACE_LEVELS:
        for o in ORIENTATION_LEVELS:
            conditions.append({"condition": "combo_r_first", "race": r, "orientation": o,
                                "phrase": combo_phrase(combined, r, o)})
            conditions.append({"condition": "combo_o_first", "race": r, "orientation": o,
                                "phrase": combo_phrase(combined, o, r)})
    conditions.append({"condition": "base", "race": None, "orientation": None, "phrase": None})
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
    parser.add_argument("--n-patterns", type=int, default=None, help="default: all available patterns")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    combined = pd.read_csv(COMBINED_PATH)
    conditions = build_conditions(combined)

    patterns = load_patterns(PATTERNS_YES_NO)
    if args.n_patterns:
        patterns = patterns.sample(n=args.n_patterns, random_state=args.seed)
    log.info(f"{len(conditions)} conditions x {len(patterns)} patterns "
             f"= {len(conditions) * len(patterns)} total prompts")

    write_header = not OUTPUT_CSV.exists()
    completed_keys: set = set()
    if not write_header:
        existing = pd.read_csv(OUTPUT_CSV)
        required_cols = {"pattern_id", "condition", "race", "orientation", "model"}
        missing = required_cols - set(existing.columns)
        if missing:
            log.error(
                f"{OUTPUT_CSV} exists but is missing columns {missing} -- "
                f"looks like a stale/incompatible file. Rename or delete it to start fresh."
            )
            sys.exit(1)
        for _, r in existing.iterrows():
            completed_keys.add((r["pattern_id"], r["condition"], r["race"], r["orientation"], r["model"]))
        log.info(f"Resuming -- {len(completed_keys)} rows already done.")

    work = []
    for pat_idx, pat_row in patterns.iterrows():
        for cond in conditions:
            key = (pat_idx, cond["condition"], cond["race"], cond["orientation"], args.model)
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
                    "race": cond["race"], "orientation": cond["orientation"],
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