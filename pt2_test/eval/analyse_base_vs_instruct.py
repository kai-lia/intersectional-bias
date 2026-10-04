"""
Analyse the base-vs-instruct probe.

Two claims were made from instruct-only data and could not be tested there:

  H1 (margin)      instruction/safety tuning moves the model away from the
                   decision boundary in some domains, so identity signal no
                   longer changes the answer.
  H2 (suppression) tuning shrinks the identity signal itself.

These have different implications -- H1 leaves the bias intact and merely
buffered, H2 removes it -- and the per-domain instruct data cannot separate
them.  The base checkpoints can.

Scale caveat, applied throughout
--------------------------------
Instruction tuning roughly doubles both the yes/no probability mass and
mean|delta| (granite 1.26 -> 1.73, llama 0.48 -> 1.54).  Instruct models are
both more willing to answer and more decisive.  A raw base-vs-instruct
comparison of margins or SDs therefore partly measures "instruct is more
confident overall" rather than anything domain-specific.  Every cross-variant
number here is normalised by that variant's own dispersion (SD of delta over
all its prompts), so it reads as "fraction of this model's own dynamic range".

delta is a genuine conditional log odds-ratio, log P(yes)/P(no), taken from a
full-vocabulary log_softmax, so delta = 0 is a real indifference point (unlike
the logit-lens deltas elsewhere, which carried a fitted offset).  Signs are
polarity-corrected so positive always means "toward the biased answer".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT_DIR = ROOT.parent / "data" / "eval"
PATTERNS = REPO / "data" / "templates" / "patterns_list_yes_no.csv"

CATEGORY = {0:"child_safety",1:"child_safety",8:"child_safety",10:"child_safety",
 13:"child_safety",14:"child_safety",15:"child_safety",23:"child_safety",
 2:"employment",5:"employment",22:"employment",31:"employment",
 3:"housing",4:"housing",11:"housing",17:"housing",18:"housing",19:"housing",
 24:"housing",28:"housing",
 25:"healthcare",32:"healthcare",33:"healthcare",34:"healthcare",35:"healthcare",
 36:"healthcare",
 30:"legal",
 6:"social",7:"social",9:"social",12:"social",16:"social",20:"social",21:"social",
 26:"social",27:"social",29:"social"}
DOMS = ["child_safety", "housing", "employment", "legal", "social", "healthcare"]

MASS_GATE = 0.05   # below this, delta compares two tokens the model would not emit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_full")
    args = ap.parse_args()

    d = pd.read_csv(OUT_DIR / f"base_vs_instruct{args.tag}.csv")
    pol = pd.read_csv(PATTERNS)["Biased Answer"].astype(str).str.strip().str.lower()
    d["sign"] = d.pattern_id.map(pol.map({"yes": 1.0, "no": -1.0}))
    d["delta"] = d.sign * d.delta_yes            # + = toward the biased answer
    d["domain"] = d.pattern_id.map(CATEGORY)

    print("=" * 92)
    print("VALIDITY GATE -- median yes/no probability mass (delta is meaningless if this is ~0)")
    print("=" * 92)
    gate = d.groupby(["model", "variant"]).yn_mass.median().unstack()
    print(gate.round(4).to_string())
    bad = gate[gate < MASS_GATE].stack()
    if len(bad):
        print(f"\n  *** below gate ({MASS_GATE}) -- exclude: {list(bad.index)} ***")
    else:
        print(f"\n  all variants pass (> {MASS_GATE})")

    # within-variant dispersion, used to make base and instruct comparable
    scale = d.groupby(["model", "variant"]).delta.std().rename("scale")
    d = d.merge(scale, on=["model", "variant"])
    d["delta_n"] = d.delta / d.scale

    print("\n" + "=" * 92)
    print("H1 (MARGIN) -- where does the no-stigma control sit, and where do identities push it?")
    print("  units: fractions of that variant's own dispersion.  + = biased side of indifference")
    print("=" * 92)
    for m in d.model.unique():
        print(f"\n{m}")
        print(f"  {'domain':<14}{'base:ctrl':>11}{'base:ident':>12}{'base:shift':>12}"
              f"{'inst:ctrl':>11}{'inst:ident':>12}{'inst:shift':>12}")
        for dom in DOMS:
            row = f"  {dom:<14}"
            for v in ["base", "instruct"]:
                s = d[(d.model == m) & (d.variant == v) & (d.domain == dom)]
                if s.empty:
                    row += f"{'-':>11}{'-':>12}{'-':>12}"; continue
                ctrl = s[s.condition == "base"].delta_n.mean()
                iden = s[s.condition == "individual"].delta_n.mean()
                row += f"{ctrl:>11.2f}{iden:>12.2f}{iden - ctrl:>+12.2f}"
            print(row)

    print("\n" + "=" * 92)
    print("H2 (SUPPRESSION) -- identity sensitivity: SD across the 112 identities,")
    print("  computed within pattern then averaged, normalised by variant dispersion")
    print("=" * 92)
    ind = d[d.condition == "individual"]
    sd = (ind.groupby(["model", "variant", "domain", "pattern_id"]).delta_n.std()
             .groupby(["model", "variant", "domain"]).mean().rename("identity_SD").reset_index())
    for m in d.model.unique():
        print(f"\n{m}")
        print(f"  {'domain':<14}{'base':>9}{'instruct':>11}{'ratio i/b':>11}   interpretation")
        for dom in DOMS:
            b = sd[(sd.model == m) & (sd.variant == "base") & (sd.domain == dom)].identity_SD
            i = sd[(sd.model == m) & (sd.variant == "instruct") & (sd.domain == dom)].identity_SD
            if b.empty or i.empty:
                continue
            b, i = float(b.iloc[0]), float(i.iloc[0])
            r = i / b if b else np.nan
            tag = "suppressed" if r < 0.8 else ("amplified" if r > 1.25 else "unchanged")
            print(f"  {dom:<14}{b:>9.3f}{i:>11.3f}{r:>11.2f}   {tag}")

    # overall verdict per model
    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    for m in d.model.unique():
        b = sd[(sd.model == m) & (sd.variant == "base")].identity_SD.mean()
        i = sd[(sd.model == m) & (sd.variant == "instruct")].identity_SD.mean()
        hc_b = sd[(sd.model == m) & (sd.variant == "base") & (sd.domain == "healthcare")].identity_SD
        hc_i = sd[(sd.model == m) & (sd.variant == "instruct") & (sd.domain == "healthcare")].identity_SD
        print(f"  {m}: identity_SD all-domain  base {b:.3f} -> instruct {i:.3f}  (ratio {i/b:.2f})")
        if len(hc_b) and len(hc_i):
            print(f"           healthcare only   base {float(hc_b.iloc[0]):.3f} -> "
                  f"instruct {float(hc_i.iloc[0]):.3f}  (ratio {float(hc_i.iloc[0])/float(hc_b.iloc[0]):.2f})")

    p = OUT_DIR / f"base_vs_instruct_summary{args.tag}.csv"
    sd.to_csv(p, index=False)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
