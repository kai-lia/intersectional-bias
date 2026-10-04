"""
Does a dominant partner cost the model its grip on what was ASSERTED about the
subordinate identity?

The measure
-----------
Both conditions contain both identity terms, matched in length and syntax, so
nothing here is a clause effect:

    A+ B+   "with autism and who has a criminal record"
    A+ B-   "with autism and who does not have a criminal record"

AUC separating them along v_B is how well the model registers whether B applies,
with A present.  The baseline is the same AUC for B with no partner at all,
computed on EXACTLY the identities, templates and layer used in the pair
analysis -- not a number carried over from a different subset.

    suppression = AUC(B alone) - AUC(B with a partner)

Five things this reports, each because a previous version of some claim in this
project died for want of one:

  all three models        llama has been the exception more than once
  matched-subset baseline so the drop is against a known ceiling
  template bootstrap      clustering cost a factor of thirty on an earlier
                          headline; per-template counts are reported alongside
  additive null           build synthetic pairs as B +/- A - base and rerun.
                          If addition reproduces the drop it is arithmetic
  dose-response           by quartile of partner displacement
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"
sys.path.insert(0, str(ROOT))


def auc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    a = np.concatenate([pos, neg])
    r = pd.Series(a).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def shard_pats(model):
    return sorted(int(re.match(rf"{model}_pattern(\d+)\.done", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*.done"))


def peak_layer(model):
    d = pd.read_csv(OUT_DIR / f"{model}_identity_probe_full.csv")
    d["gap"] = d.ceiling_auc_real - d.pair_auc_real
    return int(d.groupby("layer").gap.mean().idxmax())


def collect(model):
    """Everything the five tests need, at one matched layer."""
    z = np.load(OUT_DIR / f"modctl_{model}.npz", allow_pickle=True)
    ml = [int(x) for x in z["layers"]]
    L = peak_layer(model)
    if L not in ml:
        L = min(ml, key=lambda x: abs(x - L))
    ns = OUT_DIR / f"negsingles_{model}_matched.npz"
    if not ns.exists():
        return None
    zn = np.load(ns, allow_pickle=True)
    if L not in [int(x) for x in zn["layers"]]:
        return None

    pats = shard_pats(model)
    npat, na, nb = z["pattern_id"], np.array([str(x) for x in z["s1"]]), \
                   np.array([str(x) for x in z["s2"]])
    NEG, want = z[f"L{L}"], set(zip(z["s1"].astype(str).tolist(), z["s2"].astype(str).tolist()))

    POS, S, B, names = {}, [], [], None
    for pid in pats:
        p = ACT_DIR / f"{model}_pattern{pid}_layer{L}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as sh:
            sv = sh["singles_vec"].astype(np.float32)
            ss = [str(x) for x in sh["singles_stigma"]]
            base = sh["base_vec"].astype(np.float32).reshape(-1)
            cv = sh["combo_vec"].astype(np.float32)
            c1 = [str(x) for x in sh["combo_stigma1"]]
            c2 = [str(x) for x in sh["combo_stigma2"]]
        if names is None:
            names = ss
            idx = {n: j for j, n in enumerate(names)}
        S.append(sv[[idx[n] for n in names]]); B.append(base)
        for i, (a, b) in enumerate(zip(c1, c2)):
            if (a, b) in want:
                POS[(pid, a, b)] = cv[i]
        del cv
    S = np.stack(S); B = np.stack(B)

    # negated singles, aligned
    A_ns = zn[f"L{L}"]; ns_id = np.array([str(x) for x in zn["s1"]]); ns_pat = zn["pattern_id"]
    S_neg = np.zeros_like(S)
    for k, pid in enumerate(pats):
        m = ns_pat == pid
        for r, nm in zip(A_ns[m], ns_id[m]):
            if nm in idx:
                S_neg[k, idx[nm]] = r

    def dirs(ex):
        keep = [j for j in range(S.shape[0]) if j != ex]
        m = S[keep].mean(0); v = m - m.mean(0, keepdims=True)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    V = {k: dirs(k) for k in range(S.shape[0])}
    return dict(L=L, pats=pats, idx=idx, S=S, S_neg=S_neg, B=B, V=V,
                POS=POS, NEG=NEG, npat=npat, na=na, nb=nb,
                pp={pid: k for k, pid in enumerate(pats)})


def scores(D, arm):
    """Per (template, focal, partner) projections of B-affirmed and B-denied."""
    rows = []
    for i in range(len(D["NEG"])):
        pid = int(D["npat"][i]); a, b = D["na"][i], D["nb"][i]
        key = (pid, a, b)
        if key not in D["POS"] or pid not in D["pp"] or b not in D["idx"]:
            continue
        k = D["pp"][pid]; v = D["V"][k][D["idx"][b]]
        if arm == "real":
            p, n = D["POS"][key] @ v, D["NEG"][i] @ v
        else:                                   # additive null: B +/- A - base
            ja, jb = D["idx"][a], D["idx"][b]
            base = D["B"][k]
            p = (D["S"][k, jb] + D["S"][k, ja] - base) @ v
            n = (D["S_neg"][k, jb] + D["S"][k, ja] - base) @ v
        rows.append({"pattern_id": pid, "focal": a, "partner": b,
                     "pos": float(p), "neg": float(n)})
    return pd.DataFrame(rows)


def solo_scores(D):
    rows = []
    for k, pid in enumerate(D["pats"]):
        for nm, j in D["idx"].items():
            v = D["V"][k][j]
            rows.append({"pattern_id": pid, "partner": nm,
                         "pos": float(D["S"][k, j] @ v),
                         "neg": float(D["S_neg"][k, j] @ v)})
    return pd.DataFrame(rows)


def auc_by(df, keys=None):
    g = df.groupby(keys) if keys else [(None, df)]
    return pd.Series({k: auc(v.pos.values, v.neg.values) for k, v in g})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    for model in args.models:
        D = collect(model)
        if D is None:
            print(f"[{model}] matched-layer negated singles not ready"); continue
        real = scores(D, "real")
        addn = scores(D, "additive")
        solo = solo_scores(D)
        partners = set(real.partner)
        tmpl = sorted(set(real.pattern_id))
        # (2) baseline restricted to exactly these partners and templates
        solo = solo[solo.partner.isin(partners) & solo.pattern_id.isin(tmpl)]

        a_solo = auc(solo.pos.values, solo.neg.values)
        a_pair = auc(real.pos.values, real.neg.values)
        a_add = auc(addn.pos.values, addn.neg.values)

        print("=" * 76)
        print(f"{model.upper()}   layer {D['L']}   {len(partners)} partner identities,"
              f" {len(tmpl)} templates")
        print("=" * 76)
        print(f"  AUC(B asserted vs denied)")
        print(f"     B alone, matched subset      {a_solo:.3f}")
        print(f"     B with a partner             {a_pair:.3f}   drop {a_solo-a_pair:+.3f}")
        print(f"     additive null (B + A - base) {a_add:.3f}   drop {a_solo-a_add:+.3f}"
              f"   <- reproduces {(a_solo-a_add)/max(a_solo-a_pair,1e-9):.0%}")

        # (3) template bootstrap + per-template counts
        per_t = pd.DataFrame({
            "solo": auc_by(solo, "pattern_id"), "pair": auc_by(real, "pattern_id")}).dropna()
        per_t["delta"] = per_t.solo - per_t.pair   # not "drop": shadows DataFrame.drop
        boot = [per_t.delta.sample(len(per_t), replace=True,
                                  random_state=int(x)).mean()
                for x in rng.integers(0, 10 ** 6, args.n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n  template bootstrap: drop {per_t.delta.mean():+.3f}"
              f"  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"  {'excludes 0' if lo > 0 else 'INCLUDES 0'}")
        print(f"  templates showing a drop: {int((per_t.delta > 0).sum())}/{len(per_t)}")

        # (5) dose-response by partner displacement
        disp = pd.read_csv(OUT_DIR / f"{model}_erasure_additive_null_full.csv") \
                 .set_index("identity").disp
        real["d_focal"] = real.focal.map(disp)
        q = pd.qcut(real.d_focal, 4, labels=["Q1 weak", "Q2", "Q3", "Q4 strong"],
                    duplicates="drop")
        print(f"\n  by displacement of the ADDED identity:")
        for lab in q.cat.categories:
            sub = real[q == lab]
            print(f"    {lab:10s} AUC {auc(sub.pos.values, sub.neg.values):.3f}"
                  f"   n={len(sub):,}")


if __name__ == "__main__":
    main()
