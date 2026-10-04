"""
One construct, three independent operationalisations.

The construct: when a model meets a compound identity, does it treat that
identity as the sum of its parts?  Each measurement below asks that question on
a different surface, and none of them shares an estimator with the others.

  BEHAVIOUR       does the biased-answer rate decompose as ind1 + ind2 - base?
  REPRESENTATION  does v_combo decompose as v_A + v_B - v_base?
  EXPLANATION     does the model name both identities, relative to each one's
                  own single-identity ceiling?

The explanation measure is built to the same functional form as the other two,
so the comparison is like-for-like rather than merely thematic.  Under
independence, the chance of naming both is ceil_a * ceil_b -- each identity's
solo detectability.  Naming both LESS often than that product is the mention-
level analogue of a negative additivity residual:

    mention_ratio = P(both named) / (ceil_a * ceil_b)      < 1 means flattening

Ceiling-normalising matters here because identity labels differ enormously in
how findable they are in free text; a long paraphrasable label would otherwise
manufacture the result.

The branch point
----------------
If all three agree pair-for-pair, composition failure is one phenomenon visible
on three surfaces.  If behaviour and representation DISSOCIATE, that is the
stronger methodological result: output-level audits would be measuring a surface
that does not track what the model internally does.

Dominance vs order asymmetry
---------------------------
These are different claims and are reported separately, because both orderings
were generated:

  ORDER     the winner is whichever identity was mentioned FIRST
            -> a claim about the architecture / position
  DOMINANCE the winner is the same identity regardless of position
            -> a claim about the identity

Conflating them is the easiest thing for a reviewer to catch, so the script
reports the position effect and the order-invariant dominance rate side by side.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT_DIR = ROOT.parent / "data" / "eval"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT))

from score_mentions import terms, mentioned


def key(a, b):
    return tuple(sorted([str(a), str(b)]))


def load_behaviour(model, tag):
    d = pd.read_csv(OUT_DIR / f"{model}_behavioral_additivity{tag}.csv")
    d["pair"] = [key(a, b) for a, b in zip(d.stigma1, d.stigma2)]
    d["beh_resid"] = d[["behavioral_residual12", "behavioral_residual21"]].mean(axis=1)
    d["beh_nonadd"] = d.beh_resid.abs()
    d["beh_order"] = d.bias_combo12 - d.bias_combo21          # signed position effect
    return d


def load_representation(model):
    d = pd.read_csv(OUT_DIR / f"{model}_additivity_random_pairs_full.csv")
    d["pair"] = [key(a, b) for a, b in zip(d.stigma1, d.stigma2)]
    d["rep_nonadd"] = d[["non_additive_frac_combo12", "non_additive_frac_combo21"]].mean(axis=1)
    d["rep_order"] = d.lean_combo12 - d.lean_combo21
    return d[["pair", "rep_nonadd", "rep_order"]]


def load_mentions(model):
    """Per-pair P(both named) against the product of the two solo ceilings."""
    from pipeline.prompt import COMBINED_PATH
    from random_sample_activations import single_phrase
    combined = pd.read_csv(COMBINED_PATH)

    d = pd.read_csv(OUT_DIR / f"{model}_mentions_full.csv")
    ids = sorted(set(d.s1.dropna()) | set(d.s2.dropna()))
    tm = {}
    for t in ids:
        try:
            tm[t] = terms(single_phrase(combined, t))
        except Exception:
            tm[t] = (t, [w for w in str(t).lower().split() if len(w) > 2])

    sing = d[d.condition == "single"]
    ceil = {t: float(np.mean([mentioned(r, *tm[t])[1] for r in g.response]))
            for t, g in sing.groupby("s1")}

    pr = d[d.condition != "single"].copy()
    rows = []
    for r in pr.itertuples():
        m1 = mentioned(r.response, *tm[r.s1])[1]
        m2 = mentioned(r.response, *tm[r.s2])[1]
        # NB: s1 is the SAME identity in combo12 and combo21 -- only the phrase
        # order differs.  So only_s1/only_s2 track identities, not positions.
        rows.append({"pair": key(r.s1, r.s2), "condition": r.condition,
                     "first": r.s1, "second": r.s2,
                     "both": float(m1 and m2), "only_first": float(m1 and not m2),
                     "only_second": float(m2 and not m1), "neither": float(not m1 and not m2)})
    m = pd.DataFrame(rows)

    g = m.groupby("pair").agg(both=("both", "mean"), only_first=("only_first", "mean"),
                              only_second=("only_second", "mean"), n=("both", "size")).reset_index()
    g["expected_both"] = [ceil.get(p[0], np.nan) * ceil.get(p[1], np.nan) for p in g.pair]
    g["men_ratio"] = g.both / g.expected_both.replace(0, np.nan)
    g["men_nonadd"] = 1 - g.men_ratio.clip(upper=1)      # higher = more flattening
    return g, m, ceil


def dominance_vs_order(model, tag, mention_long):
    """Separate 'the same identity wins' from 'the first-mentioned wins'."""
    b = load_behaviour(model, tag)
    print("  BEHAVIOUR")
    print(f"    position effect  mean(combo12 - combo21) = {b.beh_order.mean():+.4f}"
          f"   mean|.| = {b.beh_order.abs().mean():.4f}")
    # which single-identity rate does the combo track?  winner = closer one.
    for cond, col in (("combo12", "bias_combo12"), ("combo21", "bias_combo21")):
        w1 = (b[col] - b.bias_ind1).abs() < (b[col] - b.bias_ind2).abs()
        b[f"win1_{cond}"] = w1
    agree = (b.win1_combo12 == b.win1_combo21).mean()
    print(f"    same identity wins in BOTH orderings: {agree:.3f} of pairs"
          f"   (0.5 = position decides, 1.0 = identity decides)")
    first_wins = ((b.win1_combo12).mean() + (~b.win1_combo21).mean()) / 2
    print(f"    first-mentioned identity wins:        {first_wins:.3f}"
          f"   (0.5 = no position effect)")

    if mention_long is not None and len(mention_long):
        m = mention_long
        print("  EXPLANATION")
        p12, p21 = m[m.condition == "combo12"], m[m.condition == "combo21"]
        print(f"    names only identity s1 (1st in combo12, 2nd in combo21): "
              f"{p12.only_first.mean():.3f} / {p21.only_first.mean():.3f}")
        print(f"    names only identity s2 (2nd in combo12, 1st in combo21): "
              f"{p12.only_second.mean():.3f} / {p21.only_second.mean():.3f}")
        # order-invariant dominance: per pair, is the same identity named across orderings?
        sel = m.assign(named_first=m.only_first - m.only_second)
        piv = sel.pivot_table(index="pair", columns="condition", values="named_first")
        if {"combo12", "combo21"} <= set(piv.columns):
            # s1 is the same identity in both conditions but changes POSITION,
            # so a position-driven effect flips the asymmetry sign between them
            # (negative corr) while an identity-driven effect preserves it.
            r = piv["combo12"].corr(piv["combo21"])
            print(f"    corr(s1-vs-s2 naming asymmetry, combo12 vs combo21) = {r:+.3f}"
                  f"   (POSITIVE = identity decides, negative = position decides)")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_polarityfixed")
    args = ap.parse_args()

    for model in args.models:
        print("=" * 84)
        print(f"{model.upper()}")
        print("=" * 84)
        beh = load_behaviour(model, args.tag)
        rep = load_representation(model)
        try:
            men, men_long, ceil = load_mentions(model)
        except FileNotFoundError:
            men, men_long = None, None

        df = beh[["pair", "beh_nonadd", "beh_resid"]].merge(rep, on="pair")
        if men is not None:
            df = df.merge(men[["pair", "men_nonadd", "both", "expected_both", "n"]],
                          on="pair", how="left")

        print(f"\n  CONVERGENCE  (n = {len(df)} pairs; mentions on "
              f"{int(df.men_nonadd.notna().sum()) if men is not None else 0})")
        print(f"    r(behaviour, representation) = {df.beh_nonadd.corr(df.rep_nonadd):+.3f}"
              f"   spearman {df.beh_nonadd.corr(df.rep_nonadd, method='spearman'):+.3f}")
        if men is not None:
            w = df.dropna(subset=["men_nonadd"])
            print(f"    r(behaviour, explanation)    = {w.beh_nonadd.corr(w.men_nonadd):+.3f}"
                  f"   spearman {w.beh_nonadd.corr(w.men_nonadd, method='spearman'):+.3f}   n={len(w)}")
            print(f"    r(representation, explanation) = {w.rep_nonadd.corr(w.men_nonadd):+.3f}"
                  f"   spearman {w.rep_nonadd.corr(w.men_nonadd, method='spearman'):+.3f}")
            print(f"\n    mention flattening: P(both named) {w.both.mean():.3f} vs "
                  f"expected {w.expected_both.mean():.3f}"
                  f"   ratio {(w.both.mean()/w.expected_both.mean()):.3f}")

        print()
        dominance_vs_order(model, args.tag, men_long)
        print()


if __name__ == "__main__":
    main()
