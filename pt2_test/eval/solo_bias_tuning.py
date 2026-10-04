"""
Does alignment create the protected-category bias gap, or widen a pretrained one?

Scope, and why it is narrow
---------------------------
This measures SOLO bias only -- each identity presented alone.  The composition
question (which identity is absorbed when two are combined) needs ~111 partners
per identity to be reliable; at 600 pairs the absorption estimate measured 0.03
split-half, and the earlier log-odds version measured NEGATIVE.  Solo bias, by
contrast, measures 0.905-0.927.  So this file answers the half of the mediation
question the data can actually support, and says nothing about the other half.

Design
------
Both arms use an IDENTICAL plain-text prompt.  The instruct models therefore run
WITHOUT their chat template, i.e. outside their trained distribution.  That is
deliberate -- giving instruct a chat template and base nothing would confound
tuning with prompt format -- but it means ABSOLUTE bias levels for the instruct
arm are not trustworthy.  Only the BETWEEN-TIER differential is, since that is a
within-model comparison and a shared format effect cancels.

Patterns are filtered per arm to those with usable variance (single-identity
biased-rate strictly between 0.02 and 0.98), then intersected.  Saturated
patterns contribute an identical constant to every identity, so they dilute
between-identity variance without biasing it; dropping them concentrates signal.

Reliability is checked BEFORE any effect is reported.  Three separate instruments
in this project produced confident-looking absorption numbers that turned out to
have zero or negative split-half reliability, so the gate runs first and the
effect table is only printed if it clears.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
sys.path.insert(0, str(ROOT))
from taxonomy import category, protection

TIERS = ["protected", "conditional", "unprotected"]


def load(model, variant, tag):
    p = DATA / f"genbi_{model}_{variant}{tag}.csv"
    return pd.read_csv(p) if p.exists() else None


def usable_patterns(d, min_parse=0.8):
    """Patterns that both DISCRIMINATE and are actually ANSWERED.

    Two separate failure modes.  A saturated pattern (every identity gets the
    same answer) carries no information.  A pattern the model answers in prose
    rather than yes/no -- mistral/base does this on 16 of 37 -- yields a biased
    rate computed from whatever minority did parse, which is not the same
    quantity.  `biased.mean()` silently skips NaN, so the variance filter alone
    would pass a pattern with a 5% parse rate.
    """
    s = d[d.condition == "single"]
    var = s.groupby("pattern_id").biased.mean()
    parse = s.groupby("pattern_id").model_answer.apply(lambda x: x.notna().mean())
    return set(var[(var > 0.02) & (var < 0.98)].index) & set(parse[parse >= min_parse].index)


def solo(d, pats):
    s = d[(d.condition == "single") & (d.pattern_id.isin(pats))]
    return s.groupby("s1").biased.mean()


def reliability(d, pats, n_splits, rng):
    rs = []
    pats = np.array(sorted(pats))
    for _ in range(n_splits):
        p = rng.permutation(pats)
        a, b = solo(d, p[:len(p) // 2]), solo(d, p[len(p) // 2:])
        j = pd.concat([a, b], axis=1).dropna()
        rs.append(j.iloc[:, 0].corr(j.iloc[:, 1]))
    r = float(np.mean(rs))
    return r, 2 * r / (1 + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_solo")
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--min-reliability", type=float, default=0.6)
    ap.add_argument("--min-parse", type=float, default=0.8,
                    help="drop patterns the model answers in prose rather than yes/no")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for m in args.models:
        B, I = load(m, "base", args.tag), load(m, "instruct", args.tag)
        if B is None or I is None:
            print(f"[{m}] skipped -- missing {'base' if B is None else 'instruct'}")
            continue
        common = sorted(usable_patterns(B, args.min_parse) & usable_patterns(I, args.min_parse))
        if len(common) < 5:
            print(f"[{m}] only {len(common)} usable patterns after parse+variance "
                  f"filtering -- too few to report")
            continue
        rb, relb = reliability(B, common, args.n_splits, rng)
        ri, reli = reliability(I, common, args.n_splits, rng)

        print("=" * 76)
        print(f"{m.upper()}   {len(common)} common usable patterns")
        print(f"  parse rate      base {B.model_answer.notna().mean():.3f}"
              f"   instruct {I.model_answer.notna().mean():.3f}")
        print(f"  solo-bias reliability  base {relb:.3f}   instruct {reli:.3f}"
              f"   {'OK' if min(relb, reli) >= args.min_reliability else 'BELOW GATE -- not reported'}")
        if min(relb, reli) < args.min_reliability:
            print()
            continue

        t = pd.DataFrame({"base": solo(B, common), "instruct": solo(I, common)}).dropna()
        t["delta"] = t.instruct - t.base
        t["cat"] = [category(i) for i in t.index]
        t["prot"] = [protection(i) for i in t.index]

        print(f"\n  {'tier':14s}{'base':>8s}{'instruct':>10s}{'Δ':>9s}{'n':>5s}")
        for g in TIERS:
            k = t[t.prot == g]
            print(f"  {g:14s}{k.base.mean():8.3f}{k.instruct.mean():10.3f}"
                  f"{k.delta.mean():+9.3f}{len(k):5d}")
        r = t[t.cat == "race"]
        print(f"  {'race only':14s}{r.base.mean():8.3f}{r.instruct.mean():10.3f}"
              f"{r.delta.mean():+9.3f}{len(r):5d}")
        print(f"  {'ALL':14s}{t.base.mean():8.3f}{t.instruct.mean():10.3f}"
              f"{t.delta.mean():+9.3f}{len(t):5d}")

        pr, un = t[t.prot == "protected"], t[t.prot == "unprotected"]
        gap_b = un.base.mean() - pr.base.mean()
        gap_i = un.instruct.mean() - pr.instruct.mean()
        print(f"\n  protected/unprotected GAP   base {gap_b:+.3f}   instruct {gap_i:+.3f}"
              f"   widened by {gap_i - gap_b:+.3f}")
        print(f"  race ratio to overall       base {r.base.mean()/t.base.mean():.2f}"
              f"   instruct {r.instruct.mean()/t.instruct.mean():.2f}")
        print(f"  race mean rank (1=least)    base {t.base.rank().loc[r.index].mean():.0f}/{len(t)}"
              f"   instruct {t.instruct.rank().loc[r.index].mean():.0f}/{len(t)}")
        print()
        rows.append({"model": m, "gap_base": gap_b, "gap_instruct": gap_i,
                     "gap_widening": gap_i - gap_b,
                     "d_protected": pr.delta.mean(), "d_unprotected": un.delta.mean(),
                     "race_base": r.base.mean(), "race_instruct": r.instruct.mean(),
                     "overall_base": t.base.mean(), "overall_instruct": t.instruct.mean()})

    if rows:
        s = pd.DataFrame(rows).set_index("model")
        print("=" * 76)
        print("CROSS-MODEL SUMMARY")
        print(s.round(3).to_string())
        print(f"\n  pretrained gap present in all models: "
              f"{bool((s.gap_base > 0).all())}")
        print(f"  alignment widens the gap in all models: "
              f"{bool((s.gap_widening > 0).all())}")
        s.to_csv(ROOT.parent / "data" / "eval" / f"solo_bias_tuning{args.tag}.csv")


if __name__ == "__main__":
    main()
