"""
Which identities absorb their partners, and is race special?

ABSORPTION.  For a pair (A,B) the model's compound response tracks one
constituent more closely than the other.  Naively counting who it tracks is
misleading: under an additive account with a constant sub-additive shift,

    c = a + b - base - delta   =>   |c-a| = |b-base-delta|,  |c-b| = |a-base-delta|

so whenever both solo rates exceed base+delta the HIGHER one wins *by
construction*.  Simulating that null recovers ~2/3 of the raw effect.  Every
number here is therefore an EXCESS over each identity's own mechanical
baseline: observed win rate minus the win rate the additive-plus-shift account
predicts for that identity against those same partners.

PSEUDO-REPLICATION.  The benchmark ships several labels for one underlying
category -- "African American"/"Black"/"Black/African American",
"Latina"/"Latino"/"Latina/Latino", eight body-weight labels for roughly two
conditions, seven mobility labels for roughly two.  Treating labels as
independent inflates significance everywhere, not only for race.  Results are
therefore reported at three levels of conservatism:

  identity  112 labels, treated as independent          (liberal)
  race-cluster  7 distinct racial categories             (within-race consistency)
  category  17 categories, one mean each                 (conservative)

The conservative test is the one to quote: it asks whether race is an outlier
among 17 category means, which is immune to how many labels the benchmark
happens to ship per category.

NO WHITE REFERENCE.  Every racial label in the benchmark is minoritised, so no
within-race contrast is available.  The claim is "race is absorbed relative to
other stigma categories", not "minoritised race is absorbed relative to White".
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "data" / "eval"
sys.path.insert(0, str(ROOT))
from taxonomy import CATEGORY, RACE_CLUSTER, ASCRIBED, ACQUIRED, category, race_cluster


def excess_table(model: str, tag: str) -> pd.DataFrame:
    """Per-identity excess absorption over its own mechanical baseline."""
    d = pd.read_csv(OUT_DIR / f"{model}_behavioral_additivity{tag}.csv")
    c = d[["bias_combo12", "bias_combo21"]].mean(axis=1)
    a, b, base = d.bias_ind1, d.bias_ind2, d.bias_base.iloc[0]
    delta = -(c - (a + b - base)).mean()
    c_null = a + b - base - delta

    def winner(cc):
        d1, d2 = (cc - a).abs(), (cc - b).abs()
        return np.where(d1 < d2, d.stigma1, d.stigma2), np.isclose(d1, d2)

    wo, t1 = winner(c); wn, t2 = winner(c_null); keep = ~(t1 | t2)
    e = pd.DataFrame({"s1": d.stigma1[keep].values, "s2": d.stigma2[keep].values,
                      "obs": wo[keep], "null": wn[keep]})
    solo = pd.concat([d.set_index("stigma1").bias_ind1.rename("v"),
                      d.set_index("stigma2").bias_ind2.rename("v")]).groupby(level=0).mean()
    rows = []
    for t in sorted(set(e.s1) | set(e.s2)):
        s = e[(e.s1 == t) | (e.s2 == t)]
        o = (s.obs == t).mean(); n_ = (s["null"] == t).mean()
        rows.append({"identity": t, "n_partners": len(s), "obs_rate": o, "null_rate": n_,
                     "excess": o - n_, "solo": solo.get(t, np.nan),
                     "category": category(t), "race_cluster": race_cluster(t)})
    return pd.DataFrame(rows)


def perm_diff(x, y, n_perm, rng):
    """Two-sample permutation test on the difference in means."""
    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y]); n = len(x)
    null = np.array([(lambda p: p[:n].mean() - p[n:].mean())(rng.permutation(pool))
                     for _ in range(n_perm)])
    return obs, float((np.abs(null) >= abs(obs)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_polarityfixed")
    ap.add_argument("--n-perm", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    per_model = {}
    for model in args.models:
        t = excess_table(model, args.tag)
        t.to_csv(OUT_DIR / f"{model}_absorption{args.tag}.csv", index=False)
        per_model[model] = t

        print("=" * 80)
        print(f"{model.upper()}   {len(t)} identities   "
              f"mechanical baseline removed (delta-shift null)")
        print("=" * 80)

        cat = t.groupby("category").excess.agg(["mean", "size"]).sort_values("mean")
        print("\n  excess absorption by category (positive = absorbs its partners)")
        for c, r in cat.iterrows():
            mark = "  <<< RACE" if c == "race" else ""
            print(f"    {c:22s} {r['mean']:+.3f}   n={int(r['size']):2d}{mark}")

        # --- level 1: identity, treated as independent (liberal) -------------
        r_i = t[t.category == "race"].excess.values
        o_i = t[t.category != "race"].excess.values
        diff, p = perm_diff(r_i, o_i, args.n_perm, rng)
        print(f"\n  [identity level, n={len(r_i)} vs {len(o_i)}]  "
              f"race - others = {diff:+.3f}   perm p = {p:.4f}")

        # --- level 2: within-race consistency across distinct categories -----
        rc = t[t.category == "race"].groupby("race_cluster").excess.mean()
        print(f"  [race clusters, n={len(rc)}]  "
              f"{', '.join(f'{k} {v:+.2f}' for k, v in rc.items())}")
        print(f"      all {len(rc)} clusters negative: {bool((rc < 0).all())}")

        # --- level 3: category means (conservative) --------------------------
        cm = t.groupby("category").excess.mean()
        rank = (cm.rank().loc["race"], len(cm))
        r_c = np.array([cm["race"]]); o_c = cm.drop("race").values
        # one-sample: how extreme is race among the category means?
        z = (cm["race"] - o_c.mean()) / o_c.std(ddof=1)
        p_rank = (o_c <= cm["race"]).mean()
        print(f"  [category level, n={len(cm)}]  race mean {cm['race']:+.3f}"
              f"   others {o_c.mean():+.3f}   z = {z:+.2f}")
        print(f"      race ranks {int(rank[0])}/{rank[1]} (1 = most absorbed);"
              f"  {p_rank:.1%} of categories are at or below race")

        # --- is it race, or ascribed attributes generally? -------------------
        asc = t[t.category.isin(ASCRIBED - {"race"})].excess
        acq = t[t.category.isin(ACQUIRED)].excess
        print(f"\n  ascribed (excl. race) {asc.mean():+.3f} n={len(asc)}   "
              f"acquired {acq.mean():+.3f} n={len(acq)}   race {r_i.mean():+.3f} n={len(r_i)}")
        print(f"  race solo-bias {t[t.category=='race'].solo.mean():.3f} vs "
              f"others {t[t.category!='race'].solo.mean():.3f}")
        print()

    # --- cross-model agreement ---------------------------------------------
    x = pd.DataFrame({m: v.set_index("identity").excess for m, v in per_model.items()})
    print("=" * 80)
    print("cross-model correlation of excess absorption")
    print(x.corr().round(3).to_string())
    cm = pd.DataFrame({m: v.groupby("category").excess.mean() for m, v in per_model.items()})
    print("\ncategory means across models (sorted by mean):")
    print(cm.assign(mean=cm.mean(axis=1)).sort_values("mean").round(3).to_string())


if __name__ == "__main__":
    main()
