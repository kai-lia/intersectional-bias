"""
Behavioral (outcome-side) mirror of additivity_random.py's representational
non-additive-fraction metric: for each stigma pair, does the model's real
yes/no bias rate on the combo condition decompose additively as
bias_ind1 + bias_ind2 - bias_base, or is there a residual an additive model
can't explain?

Reads pt2_test/random_sample_generation.py's output (random_sample_results.csv:
one row per (pattern_id, condition, stigma1, stigma2, model), condition in
{individual, combo12, combo21, base}) and, per model, per stigma pair,
computes:
    predicted   = bias_ind1 + bias_ind2 - bias_base
    residual12  = bias_combo12 - predicted
    residual21  = bias_combo21 - predicted
    non_additive_frac_combo{12,21} = |residual| / |bias_combo{12,21} - bias_base|

Kept split by phrasing order (combo12 vs combo21) rather than pooled into one
number, matching additivity_random.py's convention on the representational
side -- so the two are directly comparable per pair, per direction.

This replaces the one-off granite_representational_vs_behavioral.csv, which
had no generating script anywhere in the repo's git history (an orphaned,
unreproducible file) and was computed from whatever behavioral data existed
before random_sample_generation.py's fixed15/full identity-scope generalization
gave the representational and behavioral sides a matched scenario set.

Output: pt2_test/data/eval/{model}_behavioral_additivity{tag}.csv
Consumed by pt2_test/eval/logit_lens.py's --behavior-csv (default path).
"""
import argparse
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT.parent / "data" / "random_sample_results.csv"
OUT_DIR = ROOT.parent / "data" / "eval"


def compute_behavioral_additivity(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """NOTE: `biased` must be polarity-corrected -- 23 of 37 patterns have "no"
    as the biased answer, so a `1 if answer=="yes"` coding inverts them and
    pooling across patterns cancels real signal.  main() repairs the column on
    load if the source file predates the fix."""
    sub = df[df["model"] == model_name]
    if sub.empty:
        raise ValueError(f"No rows for model '{model_name}' in the results CSV.")

    base_vals = sub.loc[sub.condition == "base", "biased"]
    if base_vals.empty:
        raise ValueError(f"No 'base' condition rows for model '{model_name}' -- can't compute a baseline.")
    bias_base = base_vals.mean()

    ind = sub[sub.condition == "individual"].groupby("stigma1")["biased"].mean()
    combo12 = sub[sub.condition == "combo12"].groupby(["stigma1", "stigma2"])["biased"].mean()
    combo21 = sub[sub.condition == "combo21"].groupby(["stigma1", "stigma2"])["biased"].mean()

    rows = []
    for (s1, s2), bias_combo12 in combo12.items():
        if (s1, s2) not in combo21.index or s1 not in ind.index or s2 not in ind.index:
            continue
        bias_combo21 = combo21.loc[(s1, s2)]
        bias_ind1, bias_ind2 = ind.loc[s1], ind.loc[s2]
        predicted = bias_ind1 + bias_ind2 - bias_base

        resid12 = bias_combo12 - predicted
        resid21 = bias_combo21 - predicted
        shift12 = bias_combo12 - bias_base
        shift21 = bias_combo21 - bias_base

        rows.append({
            "stigma1": s1, "stigma2": s2,
            "bias_ind1": bias_ind1, "bias_ind2": bias_ind2, "bias_base": bias_base,
            "bias_combo12": bias_combo12, "bias_combo21": bias_combo21,
            "predicted": predicted,
            "behavioral_residual12": resid12, "behavioral_residual21": resid21,
            "non_additive_frac_combo12": abs(resid12) / abs(shift12) if shift12 else np.nan,
            "non_additive_frac_combo21": abs(resid21) / abs(shift21) if shift21 else np.nan,
            "abs_behavioral_residual": (abs(resid12) + abs(resid21)) / 2,
        })

    return pd.DataFrame(rows).sort_values("abs_behavioral_residual")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    parser.add_argument("--results-csv", default=str(RESULTS_CSV))
    parser.add_argument("--tag", default="", help="output filename suffix, e.g. _full112 -- "
                         "keeps a full-scale rerun from overwriting existing fixed-15 results")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)

    # Repair legacy files written before polarity was carried through.
    if "biased_answer" not in df.columns:
        from pipeline.polarity import polarity_by_pattern, is_biased
        pol = polarity_by_pattern()
        df["biased_answer"] = df.pattern_id.map(pol)
        naive = df["biased"].copy()
        df["biased"] = [is_biased(a, b) for a, b in zip(df.model_answer, df.biased_answer)]
        print(f"  polarity repaired on load: {(naive != df['biased']).mean():.1%} of rows changed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        try:
            out = compute_behavioral_additivity(df, model_name)
        except ValueError as exc:
            print(f"[{model_name}] skipped: {exc}")
            continue
        out_path = OUT_DIR / f"{model_name}_behavioral_additivity{args.tag}.csv"
        out.to_csv(out_path, index=False)
        print(f"[{model_name}] {len(out)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
