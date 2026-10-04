"""
Does instruction tuning CAUSE the absorption of protected identities?

The question this settles
-------------------------
Protected categories are absorbed more than unprotected ones, but they also
carry 4-5x lower standalone bias, and the two explanations are observationally
identical in the instruct-only data:

  MEDIATION   tuning suppresses these categories in isolation, and that same
              suppression is why they lose under composition.  Solo bias is on
              the causal path, so controlling for it removes the effect.
  CONFOUND    protected categories are low-bias for unrelated reasons and
              absorption follows from low bias alone.  Tuning is irrelevant.

Base checkpoints discriminate them.  Under MEDIATION, base models should show
HIGH solo bias for protected categories AND LITTLE absorption of them, with
tuning moving both together.  Under CONFOUND, base models should already absorb
protected categories despite high solo bias, and tuning changes little.

Method
------
Activations for both variants were extracted with an identical plain-text prompt
(no chat template on either side, since applying one only to instruct would
confound tuning with prompt format).  Each variant is decoded through ITS OWN
output head -- base and instruct have different unembeddings, so sharing a head
would confound tuning with readout.

Decoding is logit-lens at the deepest extracted layer:

    h = x / rms(x) * norm_weight        (RMSNorm; LayerNorm branch centres x)
    logodds = logsumexp(h @ W_yes) - logsumexp(h @ W_no)

This is approximate at a non-final layer -- the remaining blocks are skipped --
so absolute values are not meaningful.  Only BASE-vs-INSTRUCT differences and
WITHIN-variant rankings are interpreted, both of which are robust to a shared
offset.

Absorption reuses the estimator from absorption.py exactly: which constituent
the compound tracks, minus the win rate predicted by an additive account with a
constant shift, so the mechanical component is removed the same way.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "data" / "eval"
HEAD_DIR = ROOT.parent / "data"
sys.path.insert(0, str(ROOT))
from taxonomy import category, protection


def load_head(model, variant):
    suffix = "" if variant == "instruct" else f"_{variant}"
    p = HEAD_DIR / f"lm_head_yesno_{model}{suffix}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        return {"rows": z["lm_head_rows"].astype(np.float32),
                "is_yes": z["is_yes"].astype(bool),
                "nw": z["norm_weight"].astype(np.float32),
                "type": str(z["norm_type"]),
                "eps": float(z["norm_eps"])}


def logodds(X, head, chunk=4000):
    """Yes/no log-odds via logit lens, applying the model's own final norm."""
    from scipy.special import logsumexp
    out = []
    W_yes = head["rows"][head["is_yes"]].T
    W_no = head["rows"][~head["is_yes"]].T
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk].astype(np.float32)
        if "LayerNorm" in head["type"]:
            x = x - x.mean(1, keepdims=True)
        rms = np.sqrt((x ** 2).mean(1, keepdims=True) + head["eps"])
        h = (x / rms) * head["nw"]
        out.append(logsumexp(h @ W_yes, axis=1) - logsumexp(h @ W_no, axis=1))
    return np.concatenate(out)


def absorption_from_logodds(sol, base_lo, pairs):
    """Same estimator as absorption.py, on decoded log-odds instead of rates."""
    a = np.array([sol[p[0]] for p in pairs.index])
    b = np.array([sol[p[1]] for p in pairs.index])
    c = pairs.values
    delta = -(c - (a + b - base_lo)).mean()
    c_null = a + b - base_lo - delta
    def win(cc):
        d1, d2 = np.abs(cc - a), np.abs(cc - b)
        return np.where(d1 < d2, 0, 1), np.isclose(d1, d2)
    wo, t1 = win(c); wn, t2 = win(c_null); keep = ~(t1 | t2)
    rows = []
    ids = sorted({t for p in pairs.index for t in p})
    for t in ids:
        m = np.array([t in p for p in pairs.index]) & keep
        if m.sum() < 5:
            continue
        pos = np.array([0 if p[0] == t else 1 for p in pairs.index])[m]
        rows.append({"identity": t, "n": int(m.sum()),
                     "obs": float((wo[m] == pos).mean()),
                     "null": float((wn[m] == pos).mean())})
    if not rows:
        raise ValueError("no identity had >=5 usable partners -- check that pair "
                         "members also appear as singles and that keys match")
    d = pd.DataFrame(rows)
    d["excess"] = d.obs - d.null
    return d


def run(model, variant, tag):
    p = OUT_DIR / f"pairact_{model}_{variant}{tag}.npz"
    head = load_head(model, variant)
    if not p.exists() or head is None:
        return None
    z = np.load(p, allow_pickle=True)
    layers = z["layers"]; L = int(max(layers))          # deepest available
    X = z[f"L{L}"]
    lo = logodds(X, head)
    cond = z["condition"]; s1 = z["s1"]; s2 = z["s2"]

    sol = pd.Series(lo[cond == "single"], index=[str(x) for x in s1[cond == "single"]]) \
            .groupby(level=0).mean()
    base_lo = float(lo[cond == "base"].mean())
    pm = np.isin(cond, ["combo12", "combo21"])
    key = [tuple(sorted([str(x), str(y)])) for x, y in zip(s1[pm], s2[pm])]
    # NB: pd.Index(list_of_tuples) silently builds a MultiIndex, which would
    # make the membership filter below iterate the CHARACTERS of the first
    # element.  Group on an explicit column so the index stays tuple-valued.
    pairs = pd.DataFrame({"k": key, "lo": lo[pm]}).groupby("k").lo.mean()
    pairs = pairs[[all(t in sol.index for t in k) for k in pairs.index]]

    d = absorption_from_logodds(sol.to_dict(), base_lo, pairs)
    d["solo"] = d.identity.map(sol)
    d["category"] = d.identity.map(category)
    d["prot"] = d.identity.map(protection)
    d["model"] = model; d["variant"] = variant; d["layer"] = L
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_p2250")
    args = ap.parse_args()

    all_rows = []
    for m in args.models:
        arms = {v: run(m, v, args.tag) for v in ("base", "instruct")}
        if any(a is None for a in arms.values()):
            missing = [v for v, a in arms.items() if a is None]
            print(f"[{m}] skipped -- missing activations or head for: {missing}")
            continue
        for v, d in arms.items():
            all_rows.append(d)
        j = arms["base"].merge(arms["instruct"], on="identity", suffixes=("_b", "_i"))
        j["d_solo"] = j.solo_i - j.solo_b
        j["d_excess"] = j.excess_i - j.excess_b

        print("=" * 78)
        print(f"{m.upper()}   layer {arms['base'].layer.iloc[0]}   n = {len(j)} identities")
        print("=" * 78)
        print(f"{'group':22s} {'solo base':>10s} {'solo inst':>10s} {'Δsolo':>8s}"
              f" {'abs base':>9s} {'abs inst':>9s} {'Δabs':>8s}")
        for g in ["protected", "conditional", "unprotected"]:
            k = j[j.prot_b == g]
            if not len(k):
                continue
            print(f"{g:22s} {k.solo_b.mean():10.2f} {k.solo_i.mean():10.2f} {k.d_solo.mean():+8.2f}"
                  f" {k.excess_b.mean():9.3f} {k.excess_i.mean():9.3f} {k.d_excess.mean():+8.3f}")
        r = j[j.category_b == "race"]
        print(f"{'  (race only)':22s} {r.solo_b.mean():10.2f} {r.solo_i.mean():10.2f}"
              f" {r.d_solo.mean():+8.2f} {r.excess_b.mean():9.3f} {r.excess_i.mean():9.3f}"
              f" {r.d_excess.mean():+8.3f}")

        print(f"\n  MEDIATION TEST -- do tuning-induced changes in solo bias and in")
        print(f"  absorption move together across identities?")
        print(f"    r(Δsolo, Δabsorption) = {j.d_solo.corr(j.d_excess):+.3f}   n={len(j)}")
        print(f"    base-only: r(solo, absorption)     = {j.solo_b.corr(j.excess_b):+.3f}")
        print(f"    instruct : r(solo, absorption)     = {j.solo_i.corr(j.excess_i):+.3f}")
        print(f"    protected absorbed in BASE already? "
              f"{j[j.prot_b=='protected'].excess_b.mean():+.3f} vs unprotected "
              f"{j[j.prot_b=='unprotected'].excess_b.mean():+.3f}")
        print()

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        p = OUT_DIR / f"base_vs_instruct_absorption{args.tag}.csv"
        out.to_csv(p, index=False)
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
