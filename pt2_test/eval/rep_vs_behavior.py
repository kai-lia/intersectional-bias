"""
Representation <-> behaviour link, recomputed on the polarity-corrected
log-odds data produced by pyes_pipeline.py.

The original version of this comparison used the `biased` column from
random_sample_generation.py, which had two defects:

  1. polarity was dropped -- 23 of 37 patterns have "no" as the biased answer,
     but the column was coded `1 if answer == "yes" else 0` unconditionally,
     so pooling across patterns averaged +signal against -signal;
  2. it was a 1-bit threshold of a continuous quantity, averaged over 37
     Bernoulli draws, leaving a ~8pp noise floor larger than the effects.

Both are fixed upstream in pyes_pipeline.py.  This script reports the headline
correlation on the corrected data alongside the old number, so the size of the
correction is visible rather than asserted.

Note on circularity: `resid_add` is a 1-D projection (the model's own yes/no
readout direction) at the final layer, whereas `rep_frac` is a full-dimensional
geometric distance averaged over all layers.  Correlating them is therefore not
circular -- it asks whether generic residual-stream geometry carries the
behaviourally-consequential component.  `resid_pred` IS decoded from a
representation-space construction, so its agreement with `resid_add` is
reported separately as a compositionality check, not as a link result.

Outputs: printed report + rep_vs_behavior_summary{tag}.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "data" / "eval"


def load(model: str, tag: str):
    sc = pd.read_csv(OUT_DIR / f"{model}_pyes_scenarios{tag}.csv")
    # Normalised residual, dimensionally matched to the representational
    # non-additive fraction: |residual| / |shift from this pattern's baseline|.
    # The raw log-odds residual is on the scale of the logit gaps themselves
    # (tens), so it is not comparable to rep_frac without this.
    shift12 = (sc.delta_combo12 - sc.delta_base).abs()
    sc["frac_add_12"] = sc.resid_add_12.abs() / shift12.where(shift12 > 1e-9)
    sc["frac_pred_12"] = sc.resid_pred_12.abs() / shift12.where(shift12 > 1e-9)

    # per-pair behavioural non-additivity, computed within pattern then aggregated
    beh = sc.groupby(["stigma1", "stigma2"]).agg(
        abs_resid_add=("resid_add_12", lambda s: s.abs().mean()),
        abs_resid_pred=("resid_pred_12", lambda s: s.abs().mean()),
        resid_add=("resid_add_12", "mean"),
        frac_add=("frac_add_12", "mean"),
        frac_pred=("frac_pred_12", "mean"),
        delta_combo12=("delta_combo12", "mean"),
        delta_add=("delta_add", "mean"),
        delta_pred=("delta_pred", "mean"),
    ).reset_index()

    rep = pd.read_csv(OUT_DIR / f"{model}_additivity_random_pairs{tag}.csv")
    rep["rep_frac"] = rep[["non_additive_frac_combo12", "non_additive_frac_combo21"]].mean(axis=1)
    rep["lean_abs"] = rep[["lean_combo12", "lean_combo21"]].abs().mean(axis=1)

    df = beh.merge(rep[["stigma1", "stigma2", "rep_frac", "lean_abs"]], on=["stigma1", "stigma2"])

    # old (buggy) behavioural measure, for the before/after comparison
    old_path = OUT_DIR / f"{model}_behavioral_additivity{tag}.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)[["stigma1", "stigma2", "abs_behavioral_residual"]]
        df = df.merge(old, on=["stigma1", "stigma2"], how="left")
    return df, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_full")
    args = ap.parse_args()

    rows = []
    for m in args.models:
        df, sc = load(m, args.tag)
        print("=" * 84)
        print(f"{m.upper()}   n = {len(df)} pairs")
        print("=" * 84)

        r_new = df.rep_frac.corr(df.abs_resid_add)
        r_frac = df.rep_frac.corr(df.frac_add)          # dimensionally matched
        r_old = (df.rep_frac.corr(df.abs_behavioral_residual)
                 if "abs_behavioral_residual" in df else np.nan)
        r_lean = df.lean_abs.corr(df.abs_resid_add)
        r_lean_frac = df.lean_abs.corr(df.frac_add)

        print("\n  HEADLINE -- representational geometry vs behaviour")
        print(f"    OLD  r(rep_frac, |behavioural residual|)  binary+miscoded  = {r_old:+.3f}")
        print(f"    NEW  r(rep_frac, |resid_add|)  raw log-odds residual       = {r_new:+.3f}")
        print(f"    NEW  r(rep_frac, frac_add)     NORMALISED, dim-matched     = {r_frac:+.3f}   <- like-for-like")
        print(f"    NEW  r(|lean|,   |resid_add|)                              = {r_lean:+.3f}")
        print(f"    NEW  r(|lean|,   frac_add)                                 = {r_lean_frac:+.3f}")

        # how much did the behavioural measure itself change?
        if "abs_behavioral_residual" in df:
            r_oldnew = df.abs_behavioral_residual.corr(df.abs_resid_add)
            print(f"\n  old vs new behavioural measure agree at r = {r_oldnew:+.3f}"
                  f"   (low => the fixes changed the measurement substantially)")

        # compositionality check: does representation-space additivity agree
        # with behaviour-space additivity?
        r_pp = df.abs_resid_add.corr(df.abs_resid_pred)
        print(f"\n  COMPOSITIONALITY  r(|resid_add|, |resid_pred|) = {r_pp:+.3f}")
        print(f"    mean |resid_add|  (behaviour-space additivity miss) = {df.abs_resid_add.mean():.3f} log-odds")
        print(f"    mean |resid_pred| (representation-space, decoded)   = {df.abs_resid_pred.mean():.3f} log-odds")

        # the interpretable statistic: would the additive account give a
        # different answer than the model actually gives?
        flip_add = (np.sign(sc.delta_combo12) != np.sign(sc.delta_add)).mean()
        flip_pred = (np.sign(sc.delta_combo12) != np.sign(sc.delta_pred)).mean()
        print(f"\n  ANSWER FLIPS (fraction of pattern-pair scenarios where the additive")
        print(f"  account decodes to the OPPOSITE answer from the real combo)")
        print(f"    behaviour-space additive prediction : {flip_add:.3f}")
        print(f"    representation-space additive pred. : {flip_pred:.3f}")

        rows.append({"model": m, "n_pairs": len(df),
                     "r_old_binary": r_old, "r_new_logodds": r_new,
                     "r_new_normalised": r_frac,
                     "r_lean_behaviour": r_lean, "r_lean_normalised": r_lean_frac,
                     "r_add_vs_pred": r_pp,
                     "mean_abs_resid_add": df.abs_resid_add.mean(),
                     "mean_abs_resid_pred": df.abs_resid_pred.mean(),
                     "flip_frac_add": flip_add, "flip_frac_pred": flip_pred})

        # layer-resolved: when does the behaviourally-relevant signal appear?
        bl_path = OUT_DIR / f"{m}_pyes_by_layer{args.tag}.csv"
        if bl_path.exists():
            bl = pd.read_csv(bl_path)
            final = bl.layer.max()
            fin = bl[bl.layer == final].set_index(["stigma1", "stigma2"]).abs_resid_add
            print(f"\n  LAYER-RESOLVED -- r(|resid_add| at layer L, |resid_add| at final layer {final})")
            traj = []
            for L, g in bl.groupby("layer"):
                s = g.set_index(["stigma1", "stigma2"]).abs_resid_add
                idx = s.index.intersection(fin.index)
                traj.append((L, s.loc[idx].corr(fin.loc[idx])))
            t = pd.DataFrame(traj, columns=["layer", "r"]).sort_values("layer")
            bins = np.array_split(np.arange(len(t)), 5)
            print("    " + "  ".join(
                f"L{int(t.layer.iloc[b[0]])}-{int(t.layer.iloc[b[-1]])}: {t.r.iloc[b].mean():+.2f}"
                for b in bins if len(b)))
        print()

    summary = pd.DataFrame(rows)
    p = OUT_DIR / f"rep_vs_behavior_summary{args.tag}.csv"
    summary.to_csv(p, index=False)
    print("=" * 84)
    print(summary.to_string(index=False))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
