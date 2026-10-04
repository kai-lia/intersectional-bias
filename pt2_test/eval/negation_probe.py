"""
Does the identity direction encode ASSERTION, or merely topic-presence?

The question
------------
v_i is estimated from prompts asserting identity i ("with autism").  If it is a
semantic direction, the negated prompt ("without autism") should project
NEGATIVELY onto it.  Three outcomes, in descending order of how good they are for
the paper:

  strongly negative   semantic.  The direction tracks whether the attribute
                      holds, and displacement means what we have assumed.
  near zero           the direction encodes presence-of-topic, not assertion.
                      Every downstream claim becomes "the model responds to the
                      topic being raised", which is a weaker but still real claim.
  positive, near      LEXICAL.  The direction is tracking the tokens, and the
  the affirmative     frequency confound is live.  This is the outcome that would
                      most change the paper, which is why it is worth running.

Test 3 is the one that bears on the headline.  Marks & Tegmark (arXiv:2310.06824)
report that truth directions ROTATE with depth -- antipodal early, orthogonal in
the middle, aligned late.  If affirmative and negated identity directions rotate
similarly here, then geometric relationships in this setup are depth-dependent,
and a monotone recovery gap across layers may partly reflect rotation rather than
information loss.

Test 4 substitutes the model's own surprisal for unigram frequency.  It is the
better measure for this purpose -- it is what the representation actually
responds to, and it is already computed per model in {model}_lexical.csv.
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
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    a = np.concatenate([pos, neg])
    r = pd.Series(a).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_affirmative(model, layer, pats):
    """(n_pat, n_id, d) affirmative singles, aligned identity order."""
    mats, names = [], None
    for pid in pats:
        p = ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz"
        if not p.exists():
            return None, None
        with np.load(p, allow_pickle=True) as z:
            sv = z["singles_vec"].astype(np.float32)
            ss = [str(x) for x in z["singles_stigma"]]
        if names is None:
            names = ss
            idx = {n: k for k, n in enumerate(names)}
        mats.append(sv[[idx[n] for n in names]])
    return np.stack(mats), names


def directions(S, exclude_k):
    """LOTO diff-in-means; (n_id, d) unit rows."""
    keep = [k for k in range(S.shape[0]) if k != exclude_k]
    m = S[keep].mean(0)
    v = m - m.mean(0, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    args = ap.parse_args()

    for model in args.models:
        np_path = OUT_DIR / f"negsingles_{model}.npz"
        if not np_path.exists():
            print(f"[{model}] no negated singles yet"); continue
        z = np.load(np_path, allow_pickle=True)
        layers = [int(L) for L in z["layers"]]
        neg_pat = z["pattern_id"]; neg_id = np.array([str(x) for x in z["s1"]])

        rows = []
        for L in layers:
            pats = sorted(set(neg_pat.tolist()))
            S_aff, names = load_affirmative(model, L, pats)
            if S_aff is None:
                continue
            idx = {n: k for k, n in enumerate(names)}
            A = z[f"L{L}"]
            # reshape negated into (n_pat, n_id, d) aligned to `names`
            S_neg = np.zeros_like(S_aff)
            for k, pid in enumerate(pats):
                m = neg_pat == pid
                sub, ids = A[m], neg_id[m]
                for r, nm in zip(sub, ids):
                    if nm in idx:
                        S_neg[k, idx[nm]] = r
            n_pat, n_id, d = S_aff.shape

            proj_aff, proj_neg, proj_oth = [], [], []
            auc_neg, ang = [], []
            for k in range(n_pat):
                V = directions(S_aff, k)                    # affirmative, LOTO
                Vn = directions(S_neg, k)                   # negated, LOTO
                ra = S_aff[k] - S_aff[k].mean(0, keepdims=True)
                rn = S_neg[k] - S_neg[k].mean(0, keepdims=True)
                sa = (ra * V).sum(1)                        # own-direction, affirmative
                sn = (rn * V).sum(1)                        # own-direction, negated
                oth = (ra @ V.T)[~np.eye(n_id, dtype=bool)] # other identities
                proj_aff.append(sa); proj_neg.append(sn); proj_oth.append(oth)
                auc_neg.append(auc(sa, sn))
                ang.append(np.degrees(np.arccos(np.clip((V * Vn).sum(1), -1, 1))))
            rows.append({
                "layer": L,
                "proj_affirmative": float(np.concatenate(proj_aff).mean()),
                "proj_negated": float(np.concatenate(proj_neg).mean()),
                "proj_other": float(np.concatenate(proj_oth).mean()),
                "auc_aff_vs_neg": float(np.nanmean(auc_neg)),
                "angle_deg": float(np.concatenate(ang).mean()),
            })
        d = pd.DataFrame(rows)
        d.to_csv(OUT_DIR / f"{model}_negation_probe.csv", index=False)

        print("=" * 78)
        print(f"{model.upper()}   {len(d)} layers")
        print("=" * 78)
        print(d.round(3).to_string(index=False))

        deep = d[d.layer >= d.layer.median()]
        pa, pn = deep.proj_affirmative.mean(), deep.proj_negated.mean()
        verdict = ("SEMANTIC — negation reverses the direction" if pn < -0.25 * abs(pa) else
                   "TOPIC-PRESENCE — negation projects near zero" if abs(pn) < 0.25 * abs(pa) else
                   "LEXICAL — negation projects with the affirmative")
        print(f"\n  deeper half: affirmative {pa:+.3f}   negated {pn:+.3f}"
              f"   ratio {pn/pa if pa else np.nan:+.2f}")
        print(f"  -> {verdict}")
        print(f"  AUC(affirmative vs negated) = {deep.auc_aff_vs_neg.mean():.3f}"
              f"   angle(v_i, v_i^neg) = {deep.angle_deg.mean():.1f}°")
        print(f"  angle across depth: {d.angle_deg.iloc[0]:.0f}° (layer {d.layer.iloc[0]})"
              f" -> {d.angle_deg.iloc[-1]:.0f}° (layer {d.layer.iloc[-1]})")

        # Test 4 -- is weaker reversal associated with rarer phrasing?
        lex = OUT_DIR / f"{model}_lexical.csv"
        if lex.exists():
            per_id = []
            L = int(deep.layer.iloc[len(deep) // 2])
            pats = sorted(set(neg_pat.tolist()))
            S_aff, names = load_affirmative(model, L, pats)
            idx = {n: k for k, n in enumerate(names)}
            A = z[f"L{L}"]
            S_neg = np.zeros_like(S_aff)
            for k, pid in enumerate(pats):
                m = neg_pat == pid
                for r, nm in zip(A[m], neg_id[m]):
                    if nm in idx:
                        S_neg[k, idx[nm]] = r
            rev = np.zeros(len(names))
            for k in range(S_aff.shape[0]):
                V = directions(S_aff, k)
                ra = S_aff[k] - S_aff[k].mean(0, keepdims=True)
                rn = S_neg[k] - S_neg[k].mean(0, keepdims=True)
                rev += (ra * V).sum(1) - (rn * V).sum(1)
            rev /= S_aff.shape[0]
            t = pd.DataFrame({"identity": names, "reversal": rev}).set_index("identity")
            t = t.join(pd.read_csv(lex, index_col=0))
            print(f"\n  Test 4 (layer {L}): reversal magnitude vs lexical properties")
            for c in ["total", "per_token", "n_tok"]:
                if c in t:
                    print(f"    r(reversal, {c:9s}) = {t.reversal.corr(t[c]):+.3f}")


if __name__ == "__main__":
    main()
