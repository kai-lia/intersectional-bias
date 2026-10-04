"""
Does a dominant identity suppress whether the model registers what was SAID
about the subordinate one?

The dissociation
----------------
Flattening currently cannot separate two stories:

  topic crowded out    B's content is down-weighted because B was mentioned
  assertion lost       B's content survives, but whether the person HAS B does not

Test 1 showed that for a single identity the direction is dominated by
topic-presence: negating it drops the projection to 10-31% of affirmative
without reversing it.  Yet the model still distinguishes the two cleanly,
AUC 0.84-0.96.  So assertion IS represented -- the question is whether it
survives composition.

The measure
-----------
Both conditions contain BOTH identity terms, so nothing here is about phrase
length or topic presence:

    A+ B+   "with autism and who has a criminal record"      (main extraction)
    A+ B-   "with autism and who does not have a criminal record"  (modctl)

Project both onto v_B and take AUC.  That is how well the model registers
whether B applies, in A's presence.  Compare against the same AUC for B alone:

    suppression = AUC(B+ vs B-, alone) - AUC(B+ vs B-, paired with A)

If suppression grows with A's displacement, then adding a strong identity costs
the model its grip on what was asserted about the weaker one.  That is a
legibility failure and it is stateable as a harm: the model knows a topic was
raised but not what was said about it.

No new generation is needed -- both arms already exist on disk.
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
        neg_pat, neg_a, neg_b = z["pattern_id"], np.array([str(x) for x in z["s1"]]), \
                                np.array([str(x) for x in z["s2"]])
        NEG = z[f"L{L}"]
        want = set(zip(neg_a.tolist(), neg_b.tolist()))

        # collect matching A+B+ from the main shards, and the singles
        pos_rows, S, names = {}, [], None
        for k, pid in enumerate(pats):
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
                    pos_rows.setdefault((pid, a, b), cv[i])
            del cv
        S = np.stack(S)
        n_pat = S.shape[0]

        # v_B, leave-one-template-out
        def dirs(exclude):
            keep = [j for j in range(n_pat) if j != exclude]
            m = S[keep].mean(0); v = m - m.mean(0, keepdims=True)
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        Vs = {k: dirs(k) for k in range(n_pat)}
        pat_pos = {pid: k for k, pid in enumerate(pats)}

        # displacement of the FOCAL identity A, to test the suppression gradient
        disp = pd.read_csv(OUT_DIR / f"{model}_erasure_additive_null_full.csv") \
                 .set_index("identity").disp

        rows = []
        for (a, b) in sorted(want):
            if b not in idx:
                continue
            j = idx[b]
            pos, neg = [], []
            for i in range(len(NEG)):
                if neg_a[i] != a or neg_b[i] != b:
                    continue
                pid = int(neg_pat[i])
                key = (pid, a, b)
                if key not in pos_rows or pid not in pat_pos:
                    continue
                v = Vs[pat_pos[pid]][j]
                pos.append(float(pos_rows[key] @ v))     # A+ B+
                neg.append(float(NEG[i] @ v))            # A+ B-
            if len(pos) >= 8:
                rows.append({"focal": a, "partner": b, "n": len(pos),
                             "auc_in_pair": auc(np.array(pos), np.array(neg)),
                             "disp_focal": disp.get(a, np.nan),
                             "disp_partner": disp.get(b, np.nan)})
        d = pd.DataFrame(rows).dropna()
        if not len(d):
            print(f"[{model}] no matched pairs"); continue

        # solo baseline: AUC(B+ vs B-) with no partner present
        npz = OUT_DIR / f"negsingles_{model}.npz"
        solo_auc = {}
        if npz.exists():
            zn = np.load(npz, allow_pickle=True)
            if L in [int(x) for x in zn["layers"]]:
                A = zn[f"L{L}"]; nid = np.array([str(x) for x in zn["s1"]])
                npt = zn["pattern_id"]
                S_neg = np.zeros_like(S)
                for k, pid in enumerate(pats):
                    m = npt == pid
                    for r, nm in zip(A[m], nid[m]):
                        if nm in idx:
                            S_neg[k, idx[nm]] = r
                for b in d.partner.unique():
                    j = idx[b]
                    p_, n_ = [], []
                    for k in range(n_pat):
                        v = Vs[k][j]
                        p_.append(float(S[k, j] @ v)); n_.append(float(S_neg[k, j] @ v))
                    solo_auc[b] = auc(np.array(p_), np.array(n_))
        d["auc_solo"] = d.partner.map(solo_auc)
        d["suppression"] = d.auc_solo - d.auc_in_pair
        d.to_csv(OUT_DIR / f"{model}_negation_in_pair.csv", index=False)

        print("=" * 74)
        print(f"{model.upper()}   layer {L}   {len(d)} (focal, partner) combinations")
        print("=" * 74)
        print(f"  AUC(B asserted vs denied)   alone {d.auc_solo.mean():.3f}"
              f"   in a pair {d.auc_in_pair.mean():.3f}"
              f"   suppression {d.suppression.mean():+.3f}")
        if d.disp_focal.notna().any():
            q = pd.qcut(d.disp_focal, 3, labels=["weak partner", "mid", "strong partner"],
                        duplicates="drop")
            g = d.groupby(q, observed=True).agg(auc_in_pair=("auc_in_pair", "mean"),
                                                supp=("suppression", "mean"),
                                                n=("n", "size"))
            print("\n  by displacement of the ADDED identity:")
            for k, r in g.iterrows():
                print(f"    {str(k):16s} AUC in pair {r.auc_in_pair:.3f}"
                      f"   suppression {r.supp:+.3f}   n={int(r.n)}")
            print(f"\n  r(focal displacement, suppression) = "
                  f"{d.disp_focal.corr(d.suppression):+.3f}")


if __name__ == "__main__":
    main()
