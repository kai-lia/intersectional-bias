"""
Score whether each identity is named in the model's own justification, and test
the erasure prediction on that measure.

The prediction from the probe geometry: within a pair, the LOWER-displacement
identity is named less often than the higher-displacement one, with the gap
widening as the displacement gap grows.  This shares none of the geometry's
assumptions -- no estimated directions, no basis derived from the same prompts.

Lexical detectability
---------------------
Identity labels differ enormously in how findable they are in free text.  "Black"
is short and distinctive; "Working In A Service Industry" is long and endlessly
paraphrasable.  Since the claim is that class-type identities are the erased
ones, raw string matching would manufacture the result.

So every mention rate is read RELATIVE TO A CEILING: the same identity's mention
rate in single-identity prompts, where it is the only identity present and
unambiguously relevant.  An identity that is merely hard to spot has a low
ceiling and is not penalised twice.  Pairs are compared on

    relative_mention = mention_rate_in_pair / mention_rate_alone

Two matchers are reported, because neither is authoritative:
  strict   the identity's own phrase appears (minus the "who is" frame)
  loose    at least half of the phrase's content words appear
Agreement between them is reported; a result that holds under only one should
not be trusted.  Neither catches pure paraphrase ("different backgrounds" for a
race term), which biases toward UNDER-counting mentions -- the ceiling
normalisation is what keeps that from becoming the finding.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT_DIR = ROOT.parent / "data" / "eval"

STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "are",
        "was", "were", "who", "has", "have", "had", "with", "for", "all", "time",
        "avg", "average", "currently", "previously", "current", "person", "someone"}
FRAME = re.compile(r"^\s*(who\s+(is|has|was|uses?|lives?|works?)\s+)", re.I)


def terms(phrase: str):
    core = FRAME.sub("", str(phrase)).strip().strip(".")
    words = [w for w in re.findall(r"[A-Za-z]+", core.lower())
             if w not in STOP and len(w) > 2]
    return core, words


def mentioned(resp: str, core: str, words):
    r = str(resp).lower()
    strict = bool(core) and core.lower() in r
    loose = (sum(w in r for w in words) >= max(1, (len(words) + 1) // 2)) if words else strict
    return strict, loose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_full")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT.parent))
    from pipeline.prompt import COMBINED_PATH
    from random_sample_activations import single_phrase
    combined = pd.read_csv(COMBINED_PATH)

    for model in args.models:
        p = OUT_DIR / f"{model}_mentions{args.tag}.csv"
        if not p.exists():
            print(f"[{model}] no generations yet — skipping")
            continue
        d = pd.read_csv(p)

        ids = sorted(set(d.s1.dropna()) | set(d.s2.dropna()))
        tm = {}
        for t in ids:
            try:
                tm[t] = terms(single_phrase(combined, t))
            except Exception:
                tm[t] = (t, [w for w in re.findall(r"[A-Za-z]+", t.lower())
                             if w not in STOP and len(w) > 2])

        # ceiling: mention rate when the identity is alone
        sing = d[d.condition == "single"].copy()
        ceil = {}
        for t, grp in sing.groupby("s1"):
            c, w = tm[t]
            m = [mentioned(r, c, w) for r in grp.response]
            ceil[t] = (float(np.mean([x[0] for x in m])), float(np.mean([x[1] for x in m])))

        # pair conditions: was each member named?
        pr = d[d.condition != "single"].copy()
        rows = []
        for _, r in pr.iterrows():
            for who, other in ((r.s1, r.s2), (r.s2, r.s1)):
                c, w = tm[who]
                s, l = mentioned(r.response, c, w)
                rows.append({"pattern_id": r.pattern_id, "condition": r.condition,
                             "focal": who, "partner": other, "disp_gap": r.disp_gap,
                             "strict": float(s), "loose": float(l)})
        m = pd.DataFrame(rows)
        m["ceil_strict"] = m.focal.map(lambda t: ceil.get(t, (np.nan, np.nan))[0])
        m["ceil_loose"] = m.focal.map(lambda t: ceil.get(t, (np.nan, np.nan))[1])

        print("=" * 88)
        print(f"{model.upper()}   {len(d)} responses   {m.focal.nunique()} identities")
        print("=" * 88)
        print(f"  raw mention rate  alone {sing.pipe(lambda s: np.mean([mentioned(r, *tm[t])[1] for t, r in zip(s.s1, s.response)])):.3f}"
              f"   in a pair {m.loose.mean():.3f}   (loose matcher)")
        print(f"  matcher agreement (strict vs loose): "
              f"{(m.strict == m.loose).mean():.3f}")

        # within-pair: does the lower-displacement member get named less?
        disp = pd.read_csv(OUT_DIR / f"{model}_identity_probe{args.tag}.csv")
        # displacement ranking is recomputed upstream; use the erasure gap as the
        # per-identity ordering actually measured for this model
        disp["gap"] = disp.ceiling_auc_real - disp.pair_auc_real
        L = int(disp.groupby("layer").gap.mean().idxmax())
        rank = disp[disp.layer == L].set_index("identity").gap   # higher gap = more erased

        for matcher in ("strict", "loose"):
            g = (m.groupby(["pattern_id", "condition", "focal", "partner"], as_index=False)
                   .agg(**{matcher: (matcher, "mean"), "disp_gap": ("disp_gap", "first")}))
            g["rel"] = g[matcher] / g.focal.map(lambda t: ceil.get(t, (np.nan, np.nan))[0 if matcher == "strict" else 1]).replace(0, np.nan)
            g["focal_erasure"] = g.focal.map(rank)
            g["partner_erasure"] = g.partner.map(rank)
            w = g.dropna(subset=["rel", "focal_erasure", "partner_erasure"])
            more = w[w.focal_erasure > w.partner_erasure]      # focal is the MORE-erased member
            less = w[w.focal_erasure < w.partner_erasure]
            print(f"\n  [{matcher}] mention rate relative to the identity's own ceiling")
            print(f"    more-erased member (by probe) : {more.rel.mean():.3f}   n={len(more)}")
            print(f"    less-erased member            : {less.rel.mean():.3f}   n={len(less)}")
            print(f"    difference                    : {more.rel.mean()-less.rel.mean():+.3f}"
                  f"   {'(consistent with erasure)' if more.rel.mean() < less.rel.mean() else '(OPPOSITE to the prediction)'}")
            if "disp_gap" in w:
                q = w.dropna(subset=["disp_gap"]).copy()
                if len(q) > 40:
                    q["bin"] = pd.qcut(q.disp_gap, 4, labels=["Q1 bal", "Q2", "Q3", "Q4 dom"],
                                       duplicates="drop")
                    hi = q[q.focal_erasure > q.partner_erasure]
                    lo = q[q.focal_erasure < q.partner_erasure]
                    print(f"    by displacement-gap quartile (more-erased minus less-erased):")
                    for b in q.bin.cat.categories:
                        a1 = hi[hi.bin == b].rel.mean(); a2 = lo[lo.bin == b].rel.mean()
                        print(f"      {b:<8} {a1-a2:+.3f}")
        print()


if __name__ == "__main__":
    main()
