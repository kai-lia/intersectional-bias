"""
What IS the emergent remainder?

emergence.py establishes that ~7-10% of a pair's direction lies outside the span
of its constituents, that this is pair-specific, generalises to held-out prompts,
and is absent from additive composition.  It says nothing about what the
remainder contains.

Four questions, in the order that determines whether the later ones are worth
asking:

  1 RANK      stack the 6,216 remainders and take the SVD.  Variance spread over
              thousands of components => the remainder is idiosyncratic per pair
              and there is nothing further to say.  A handful of components
              carrying most of it => a small set of recurring intersection
              effects worth inspecting.
  2 MEANING   if low-rank, sort pairs by loading on each top component and test
              the loadings against things we already have: the six Pachankis
              ratings of both members, whether the members share a cluster, the
              displacement gap.  A component tracking nothing is reported as
              uninterpreted rather than dressed up.
  3 READOUT   project the remainder onto the yes/no direction implied by the
              model's own lm_head.  Near-zero => the structure exists but the
              output layer cannot read it, which would explain the behavioural
              null.  Non-zero => it is behaviourally available and the null gets
              harder to explain.
  4 CONTROL   the remainder is what survives projecting out two estimated
              directions, so it inherits their noise.  Running the identical SVD
              on remainders from the NOISE-MATCHED additive null shows whether
              low rank is a property of the model or of the projection
              procedure.  Top components are also checked against the low-SNR
              pairs already excluded from the headline.

The noiseless additive null used in emergence.py is not usable here: its
remainder is numerical dust, so its spectrum is meaningless.  Noise matched to
the empirical residual scale (and divided by sqrt(n_prompts), since the pair
representation is a mean) is what makes the comparison fair.
"""
import argparse
import gc
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"
LM_HEAD = ROOT.parent / "data" / "lm_head_yesno.npz"
NEO = REPO / "data" / "templates" / "neostigmas.csv"
DIMS = ["Visibility", "Persistent Course", "Disrupt", "Unappealing Aesthetics",
        "Controllable Origin", "Peril"]

sys.path.insert(0, str(ROOT))
from emergence import collect, unit, decompose, shard_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def spectrum(X, label, n_show=8):
    """SVD spectrum + effective rank of row-stacked vectors."""
    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    var = sv ** 2
    var = var / var.sum()
    cum = np.cumsum(var)
    k50 = int(np.searchsorted(cum, 0.50) + 1)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    pr = float((var.sum() ** 2) / (var ** 2).sum())    # participation ratio
    print(f"    {label:<28} PC1 {100*var[0]:5.1f}%  top8 {100*cum[7]:5.1f}%  "
          f"k50 {k50:>4}  k90 {k90:>4}  eff.rank {pr:7.1f}")
    return var, cum, k50, k90, pr


def readout_direction(model):
    """Direction in residual space that moves the yes-vs-no logit gap.

    logits = (x / rms(x) * w) @ W.T, so d(logit_yes - logit_no)/dx is
    proportional to w (elementwise) times (mean yes row - mean no row)."""
    subprocess.run([sys.executable, "pt2_test/extract_lm_head_yesno.py", "--model", model],
                   cwd=REPO, check=True, capture_output=True)
    z = np.load(LM_HEAD, allow_pickle=True)
    rows, is_yes, w = z["lm_head_rows"], z["is_yes"], z["norm_weight"]
    return unit(w * (rows[is_yes].mean(0) - rows[~is_yes].mean(0)))


def empirical_sigma(model, layer, S, B, names, chunk=2048):
    """Per-element SD of (real combo - additive prediction), for the matched null."""
    pats, _ = shard_ids(model)
    iof = {n: j for j, n in enumerate(names)}
    ss = 0.0
    n = 0
    for k, pid in enumerate(pats):
        with np.load(ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz", allow_pickle=True) as z:
            cv = z["combo_vec"].astype(np.float32)
            c1 = [str(x) for x in z["combo_stigma1"]]
            c2 = [str(x) for x in z["combo_stigma2"]]
        a = np.array([iof[x] for x in c1]); b = np.array([iof[x] for x in c2])
        for s in range(0, len(a), chunk):
            e = slice(s, min(s + chunk, len(a)))
            r = cv[e] - (S[k][a[e]] + S[k][b[e]] - B[k])
            ss += float((r.astype(np.float64) ** 2).sum()); n += r.size
        del cv; gc.collect()
    return float(np.sqrt(ss / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite"])
    ap.add_argument("--layer-mode", choices=["gap", "ceiling"], default="gap")
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    neo = pd.read_csv(NEO).dropna(subset=["Stigma"]).drop_duplicates("Stigma").set_index("Stigma")
    for c in DIMS:
        neo[c] = neo[c].map(lambda s: float(str(s).split("/")[0].strip())
                            if pd.notna(s) else np.nan)

    for model in args.models:
        pr = pd.read_csv(OUT_DIR / f"{model}_identity_probe{args.tag}.csv")
        pr["gap"] = pr.ceiling_auc_real - pr.pair_auc_real
        L = int(pr.groupby("layer").gap.mean().idxmax() if args.layer_mode == "gap"
                else pr.groupby("layer").ceiling_auc_real.mean().idxmax())
        rng = np.random.default_rng(args.seed)
        log.info(f"[{model}] remainder structure at L{L} ({args.layer_mode}-peak)")

        sums, pairs, names, S, B, half, pats = collect(model, L, rng)
        ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
        Sm, Bm = S.mean(0), B.mean(0)
        VA = unit(Sm - Sm.mean(0, keepdims=True))

        P = sums[0]
        dev = P - P.mean(0, keepdims=True)
        mag = np.linalg.norm(dev, axis=1)
        hi = mag >= np.quantile(mag, 0.75)
        rem, frac = decompose(unit(dev), VA[ia], VA[ib])
        R = unit(rem)

        # noise-matched additive null: pair rep is a mean over the half's prompts
        sigma = empirical_sigma(model, L, S, B, names)
        n_half = max(len(half), 1)
        add = Sm[ia] + Sm[ib] - Bm
        add = add + rng.normal(0, sigma / np.sqrt(n_half), add.shape).astype(np.float32)
        dev_n = add - add.mean(0, keepdims=True)
        rem_n, frac_n = decompose(unit(dev_n), VA[ia], VA[ib])
        Rn = unit(rem_n)

        print("=" * 92)
        print(f"{model.upper()}  L{L}  ({args.layer_mode}-peak)   {len(pairs)} pairs   "
              f"empirical sigma {sigma:.4f}")
        print("=" * 92)
        print(f"\n  1. RANK of the remainder  (d = {R.shape[1]}, n = {len(R)})")
        v_r, c_r, k50, k90, pr_r = spectrum(R, "observed remainders")
        v_h, _, _, _, pr_h = spectrum(R[hi], "observed, high-SNR quartile")
        v_n, _, _, _, pr_n = spectrum(Rn, "NOISE-MATCHED additive null")
        print(f"      null orth fraction {frac_n.mean():.3f} vs observed {frac.mean():.3f}")
        verdict = ("LOW-RANK relative to null -> recurring structure"
                   if pr_r < 0.5 * pr_n else
                   "comparable to null -> remainder is idiosyncratic / projection noise")
        print(f"      => {verdict}")

        # 3. readout: is the remainder visible to the output layer?
        try:
            w = readout_direction(model)
            proj_r = np.abs(R @ w)
            proj_full = np.abs(unit(dev) @ w)
            rand = unit(rng.normal(size=(2000, R.shape[1])))
            print(f"\n  3. READOUT (yes/no direction from the model's own lm_head)")
            print(f"      |cos(remainder, readout)|      mean {proj_r.mean():.4f}")
            print(f"      |cos(full pair dir, readout)|  mean {proj_full.mean():.4f}")
            print(f"      |cos(random dir, readout)|     mean {np.abs(rand @ w).mean():.4f}"
                  f"   (chance for d={R.shape[1]})")
            ratio = proj_r.mean() / max(np.abs(rand @ w).mean(), 1e-12)
            print(f"      remainder is {ratio:.1f}x chance -> "
                  f"{'readable by the output layer' if ratio > 3 else 'NOT meaningfully readable'}")
        except Exception as exc:
            log.warning(f"  readout step skipped: {type(exc).__name__}: {exc}")

        # 2. meaning of the top components
        Rc = R - R.mean(0, keepdims=True)
        U, sv, Vt = np.linalg.svd(Rc, full_matrices=False)
        load = U[:, :4] * sv[:4]
        disp = np.linalg.norm(S - B[:, None, :], axis=2).mean(0); disp = disp / disp.mean()
        feat = pd.DataFrame({"disp_gap": np.abs(disp[ia] - disp[ib]),
                             "disp_mean": (disp[ia] + disp[ib]) / 2,
                             "low_snr": (~hi).astype(float)})
        for c in DIMS:
            va = neo[c].reindex([names[i] for i in ia]).to_numpy()
            vb = neo[c].reindex([names[i] for i in ib]).to_numpy()
            feat[f"{c}|mean"] = (va + vb) / 2
            feat[f"{c}|diff"] = np.abs(va - vb)
        if "Cluster" in neo.columns:
            ca = neo["Cluster"].reindex([names[i] for i in ia]).to_numpy()
            cb = neo["Cluster"].reindex([names[i] for i in ib]).to_numpy()
            feat["same_cluster"] = (ca == cb).astype(float)

        print(f"\n  2. WHAT DO THE TOP COMPONENTS TRACK?  (|r| > 0.2 shown)")
        for j in range(4):
            cors = {k: np.corrcoef(load[:, j], feat[k].to_numpy())[0, 1]
                    for k in feat.columns if feat[k].notna().all()}
            strong = {k: v for k, v in sorted(cors.items(), key=lambda kv: -abs(kv[1]))
                      if abs(v) > 0.2}
            head = f"      PC{j+1} ({100*v_r[j]:.1f}% var):"
            print(f"{head} " + (", ".join(f"{k} {v:+.2f}" for k, v in list(strong.items())[:4])
                                if strong else "nothing above |r| = 0.2  -> UNINTERPRETED"))
            top = np.argsort(-load[:, j])[:3]; bot = np.argsort(load[:, j])[:3]
            print(f"          high: " + "; ".join(f"{names[ia[t]]} + {names[ib[t]]}" for t in top))
            print(f"          low : " + "; ".join(f"{names[ia[t]]} + {names[ib[t]]}" for t in bot))
        print()


if __name__ == "__main__":
    main()
