"""
How much does the sub-additivity conclusion depend on the baseline?

behavioral_additivity.py predicts a pair's bias rate as

    predicted = bias_ind1 + bias_ind2 - bias_base

so the residual is linear in bias_base, and the whole "pairs are sub-additive"
claim rests on where that one constant sits.  The Base Case estimate comes from
37 prompts -- one per pattern -- which looks alarmingly thin.

Two things make it less thin than it looks:

  * Decoding is greedy (do_sample=False, temperature=0), so re-running those 37
    prompts returns the same 37 answers.  There is no sampling noise to average
    down; 37 IS the population of Base Case prompts.
  * Every pair is evaluated on all 37 patterns, so the pooled base constant is
    composition-matched to the combo means exactly -- not an approximation.

What remains is specification uncertainty: "Base Case" template text is not the
only defensible no-identity control.  SocialStigmaQA also ships "Without Stigma"
phrasings, run here under four prompt styles, giving 296 rows per style per model
against the Base Case run's 37.

This script reports the break-even baseline -- the value of bias_base at which
the mean residual crosses zero and sub-additivity would flip to amplification --
and checks every available baseline estimate against it.  A conclusion that
survives all of them is not resting on 37 prompts.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "data" / "eval"
RESULTS = ROOT.parent / "data" / "random_sample_results.csv"
PT2 = ROOT.parent / "data" / "results_pt2.csv"


def baselines(model: str) -> dict[str, tuple[float, int]]:
    """Every defensible no-identity estimate, as {label: (rate, n)}."""
    out = {}
    r = pd.read_csv(RESULTS, usecols=["condition", "model", "biased"])
    b = r[(r.condition == "base") & (r.model == model)].biased
    out["Base Case template"] = (b.mean(), len(b))

    if PT2.exists():
        d = pd.read_csv(PT2, usecols=["stigma_col", "prompt_style", "model", "biased"])
        w = d[d.stigma_col.str.startswith("Without Stigma", na=False) & (d.model == model)]
        for style, g in w.groupby("prompt_style"):
            out[f"Without Stigma / {style}"] = (g.biased.mean(), len(g))
        out["Without Stigma / pooled"] = (w.biased.mean(), len(w))
    return out


def within_pattern(model: str, n_boot: int, seed: int):
    """Residuals computed inside each pattern, against that pattern's own base.

    Removes the pooled constant entirely -- 37 baselines instead of one -- and
    lets a cluster bootstrap over patterns carry the heterogeneity into a CI.
    Patterns are the resampling unit because that is the level the baseline and
    the templates vary at; pairs within a pattern are not independent of it.
    """
    import numpy as np
    d = pd.read_csv(RESULTS, usecols=["pattern_id", "condition", "stigma1", "stigma2",
                                      "model", "biased"])
    g = d[d.model == model]
    base = g[g.condition == "base"].set_index("pattern_id").biased
    ind = g[g.condition == "individual"].set_index(["pattern_id", "stigma1"]).biased
    cmb = (g[g.condition.isin(["combo12", "combo21"])]
             .groupby(["pattern_id", "stigma1", "stigma2"]).biased.mean()).reset_index()
    cmb["i1"] = ind.reindex(pd.MultiIndex.from_arrays([cmb.pattern_id, cmb.stigma1])).values
    cmb["i2"] = ind.reindex(pd.MultiIndex.from_arrays([cmb.pattern_id, cmb.stigma2])).values
    cmb["b"] = cmb.pattern_id.map(base)
    cmb = cmb.dropna(subset=["i1", "i2", "b"])
    cmb["resid"] = cmb.biased - (cmb.i1 + cmb.i2 - cmb.b)

    per_pat = cmb.groupby("pattern_id").resid.mean()
    rng = np.random.default_rng(seed)
    pats = per_pat.index.to_numpy()
    boot = np.array([per_pat.loc[rng.choice(pats, len(pats), replace=True)].mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return per_pat, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_polarityfixed")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for m in args.models:
        n = pd.read_csv(OUT_DIR / f"{m}_behavioral_additivity{args.tag}.csv")
        combo = n[["bias_combo12", "bias_combo21"]].mean(axis=1)
        # residual = combo - (ind1 + ind2 - base); linear in base
        slack = (combo - n.bias_ind1 - n.bias_ind2).mean()
        breakeven = -slack                      # base at which mean residual == 0
        used = n.bias_base.iloc[0]

        print("=" * 78)
        print(f"{m.upper()}   baseline used {used:.3f}   mean residual {slack + used:+.3f}")
        print(f"  break-even baseline (residual = 0): {breakeven:.3f}")
        print(f"  any baseline BELOW this keeps the sub-additive conclusion")
        print("-" * 78)
        for label, (rate, cnt) in baselines(m).items():
            verdict = "sub-additive" if rate < breakeven else "AMPLIFYING -- flips"
            margin = breakeven - rate
            print(f"  {label:<28} {rate:.3f}  n={cnt:<5} margin {margin:+.3f}   {verdict}")

        per_pat, lo, hi = within_pattern(m, args.n_boot, args.seed)
        sig = "excludes zero" if hi < 0 else "INCLUDES ZERO -- not significant"
        print("-" * 78)
        print(f"  within-pattern (own base per pattern), cluster bootstrap over {len(per_pat)} patterns:")
        print(f"    mean residual {per_pat.mean():+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {sig}")
        print(f"    patterns sub-additive: {int((per_pat < 0).sum())}/{len(per_pat)}"
              f"  ({(per_pat < 0).mean() * 100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
