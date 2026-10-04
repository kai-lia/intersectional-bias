"""
Is the recovery gap about identity COMPOSITION, or about attaching any second
clause?

The confound
------------
Pairing two identities makes each less linearly decodable.  But a pair phrase is
also longer, contains an "and", and introduces a second referent.  If a NEGATED
second element attenuates the focal identity as much as an affirmative one does,
the gap is about noun-phrase complexity, not about how social identities compose,
and the paper is about something else.

The comparison
--------------
Both arms contain both identity terms, matched in length, syntax and topic:

    affirmative   "who is working class or poor and has remitted psoriasis"
    negated       "who is working class or poor and does not have remitted psoriasis"

Recovery of the FOCAL identity is scored the same way in both, against the same
ceiling:

    ceiling   single_A vs other identities' singles
    pair      pairs containing A vs pairs not containing A
    gap       ceiling - pair

    gap_affirmative ~= gap_negated   -> generic modifier interference
    gap_negated much smaller         -> attenuation requires the second identity
                                        to be ASSERTED, and the finding is about
                                        composition

The same (focal, partner) combinations are used in both arms, so the contrast is
exact rather than distributional.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    args = ap.parse_args()

    for model in args.models:
        mp = OUT_DIR / f"modctl_{model}.npz"
        if not mp.exists():
            print(f"[{model}] no half-negated pairs"); continue
        z = np.load(mp, allow_pickle=True)
        mlayers = [int(x) for x in z["layers"]]
        L = peak_layer(model)
        if L not in mlayers:
            L = min(mlayers, key=lambda x: abs(x - L))
        pats = shard_pats(model)
        neg_pat = z["pattern_id"]
        neg_a = np.array([str(x) for x in z["s1"]])
        neg_b = np.array([str(x) for x in z["s2"]])
        NEG = z[f"L{L}"]
        want = set(zip(neg_a.tolist(), neg_b.tolist()))

        POS, S, names = {}, [], None
        for pid in pats:
            p = ACT_DIR / f"{model}_pattern{pid}_layer{L}.npz"
            if not p.exists():
                continue
            with np.load(p, allow_pickle=True) as sh:
                sv = sh["singles_vec"].astype(np.float32)
                ss = [str(x) for x in sh["singles_stigma"]]
                cv = sh["combo_vec"].astype(np.float32)
                c1 = [str(x) for x in sh["combo_stigma1"]]
                c2 = [str(x) for x in sh["combo_stigma2"]]
            if names is None:
                names = ss; idx = {n: j for j, n in enumerate(names)}
            S.append(sv[[idx[n] for n in names]])
            for i, (a, b) in enumerate(zip(c1, c2)):
                if (a, b) in want:
                    POS[(pid, a, b)] = cv[i]
            del cv
        S = np.stack(S); n_pat = S.shape[0]
        pat_pos = {pid: k for k, pid in enumerate(pats)}

        def dirs(exclude):
            keep = [j for j in range(n_pat) if j != exclude]
            m = S[keep].mean(0); v = m - m.mean(0, keepdims=True)
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        V = {k: dirs(k) for k in range(n_pat)}

        focals = sorted({a for a, _ in want})
        rows = []
        for a in focals:
            if a not in idx:
                continue
            j = idx[a]
            ceil_p, ceil_n = [], []
            aff_in, aff_out, neg_in, neg_out = [], [], [], []
            for k, pid in enumerate(pats):
                v = V[k][j]
                sc = S[k] @ v                       # ceiling: this identity vs others
                ceil_p.append(sc[j]); ceil_n.extend(np.delete(sc, j))
            for i in range(len(NEG)):
                pid = int(neg_pat[i]); key = (pid, neg_a[i], neg_b[i])
                if pid not in pat_pos or key not in POS:
                    continue
                v = V[pat_pos[pid]][j]
                pv, nv = float(POS[key] @ v), float(NEG[i] @ v)
                if neg_a[i] == a:                   # pair contains the focal
                    aff_in.append(pv); neg_in.append(nv)
                else:
                    aff_out.append(pv); neg_out.append(nv)
            c = auc(np.array(ceil_p), np.array(ceil_n))
            pa = auc(np.array(aff_in), np.array(aff_out))
            pn = auc(np.array(neg_in), np.array(neg_out))
            if np.isnan(pa) or np.isnan(pn):
                continue
            rows.append({"focal": a, "ceiling": c,
                         "pair_affirmative": pa, "pair_negated": pn,
                         "gap_affirmative": c - pa, "gap_negated": c - pn,
                         "n_in": len(aff_in)})
        d = pd.DataFrame(rows)
        if not len(d):
            print(f"[{model}] no usable focals"); continue
        d["gap_ratio"] = d.gap_negated / d.gap_affirmative.replace(0, np.nan)
        d.to_csv(OUT_DIR / f"{model}_modifier_interference.csv", index=False)

        rng = np.random.default_rng(0)
        diff = d.gap_affirmative - d.gap_negated
        boot = [diff.sample(len(diff), replace=True, random_state=int(x)).mean()
                for x in rng.integers(0, 10 ** 6, 2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])

        print("=" * 74)
        print(f"{model.upper()}   layer {L}   {len(d)} focal identities")
        print("=" * 74)
        print(f"  ceiling (focal alone)          {d.ceiling.mean():.3f}")
        print(f"  paired with AFFIRMATIVE second {d.pair_affirmative.mean():.3f}"
              f"   gap {d.gap_affirmative.mean():+.3f}")
        print(f"  paired with NEGATED second     {d.pair_negated.mean():.3f}"
              f"   gap {d.gap_negated.mean():+.3f}")
        print(f"\n  gap difference (affirmative - negated) {diff.mean():+.3f}"
              f"   95% CI [{lo:+.3f}, {hi:+.3f}]")
        share = d.gap_negated.mean() / d.gap_affirmative.mean()
        print(f"  a NEGATED second element reproduces {share:.0%} of the attenuation")
        verdict = ("GENERIC — attenuation does not require the second identity to be asserted"
                   if share > 0.75 else
                   "COMPOSITION — attenuation largely requires assertion" if share < 0.5 else
                   "MIXED — both a clause effect and an assertion effect")
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
