"""
Is the pair representation decomposable into its constituents, or is there
something at the intersection that neither part contains?

Why this is needed
------------------
Every direction in identity_probe.py is estimated from SINGLE-identity prompts.
So "B is recoverable from AB" means "AB resembles B as B appears alone".  If the
intersection produces a third thing that is neither A nor B, the probe registers
that as absence of B -- which we have been calling erasure.  The instrument
cannot distinguish:

    (i)  the subordinate identity was flattened away, from
    (ii) something emerged that a parts-based decoder was never built to see.

This script separates them.  For each pair, fit a direction from the PAIR
representations themselves and decompose it against the span of its
constituents' directions:

    v_AB = normalise( mean_p(rep_AB) - global mean )
    in-span component  = projection onto span(v_A, v_B)
    orthogonal remainder = what is left

Three outcomes
--------------
  1  remainder ~ 0                     no emergence; flattening is the whole story
  2  substantial remainder, PAIR-SPECIFIC   something exists at the intersection
                                       that neither constituent contains
  3  substantial remainder, SHARED     a generic "two identity terms in the
                                       prompt" offset -- longer text, extra
                                       clause -- with nothing intersectional
                                       about it.  This is the trap and is the
                                       likely default.

Outcome 3 is ruled out by measuring whether remainders from different pairs
point the same way (mean pairwise cosine), then removing the shared component
and asking what survives.  Only pair-specific structure counts as emergence.

Two floors are computed through the identical pipeline:
  additive null   synthetic AB = ind_A + ind_B - base.  By construction its
                  pair direction lies exactly in span(v_A, v_B), so its
                  remainder is the numerical noise floor.
  held-out AUC    v_AB is fitted on one half of the prompts and tested on the
                  other, so a "remainder" that is just per-pair noise cannot
                  masquerade as structure.

Finally: does emergence trade off against flattening?  Pairs with a large
displacement gap show strong flattening.  If the balanced pairs instead show
more orthogonal structure, flattening and emergence are alternative outcomes of
composition rather than competing descriptions of one.
"""
import argparse
import gc
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def mean_pairwise_cosine(R):
    """Mean off-diagonal cosine among unit rows, without forming the n x n matrix."""
    n = len(R)
    s = R.sum(0)
    return float((s @ s - n) / (n * n - n))


def shard_ids(model):
    pats = sorted(int(re.match(rf"{model}_pattern(\d+)\.done", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*.done"))
    layers = sorted(int(re.match(rf"{model}_pattern{pats[0]}_layer(\d+)\.npz", f.name).group(1))
                    for f in ACT_DIR.glob(f"{model}_pattern{pats[0]}_layer*.npz"))
    return pats, layers


def collect(model, layer, rng):
    """Per-pair mean representations, split into two prompt halves."""
    pats, _ = shard_ids(model)
    half = set(rng.permutation(pats)[:len(pats) // 2])

    singles, bases, names = {}, {}, None
    for pid in pats:
        with np.load(ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz", allow_pickle=True) as z:
            sv = z["singles_vec"].astype(np.float32)
            ss = [str(x) for x in z["singles_stigma"]]
            bases[pid] = z["base_vec"][0].astype(np.float32)
        if names is None:
            names = ss
        idx = {n: k for k, n in enumerate(ss)}
        singles[pid] = sv[[idx[n] for n in names]]
    iof = {n: j for j, n in enumerate(names)}
    nI = len(names)
    d = singles[pats[0]].shape[1]

    pair_key, acc = {}, None
    sums = {0: None, 1: None}
    cnts = {0: None, 1: None}
    for pid in pats:
        h = 0 if pid in half else 1
        with np.load(ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz", allow_pickle=True) as z:
            cv = z["combo_vec"].astype(np.float32)
            c1 = [str(x) for x in z["combo_stigma1"]]
            c2 = [str(x) for x in z["combo_stigma2"]]
        a = np.array([iof[x] for x in c1]); b = np.array([iof[x] for x in c2])
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        keys = lo * nI + hi
        if acc is None:
            uniq = np.unique(keys)
            pair_key = {k: i for i, k in enumerate(uniq)}
            for hh in (0, 1):
                sums[hh] = np.zeros((len(uniq), d), np.float32)
                cnts[hh] = np.zeros(len(uniq), np.float32)
            acc = True
        ridx = np.array([pair_key[k] for k in keys])
        np.add.at(sums[h], ridx, cv)
        np.add.at(cnts[h], ridx, 1.0)
        del cv; gc.collect()

    inv = {v: k for k, v in pair_key.items()}
    pairs = [(inv[i] // nI, inv[i] % nI) for i in range(len(pair_key))]
    for hh in (0, 1):
        sums[hh] /= np.maximum(cnts[hh], 1)[:, None]
    S = np.stack([singles[p] for p in pats])          # (n_pat, nI, d)
    B = np.stack([bases[p] for p in pats])            # (n_pat, d)
    return sums, pairs, names, S, B, half, pats


def decompose(v_ab, v_a, v_b):
    """Orthogonal-remainder fraction of unit v_ab against span(v_a, v_b)."""
    e1 = unit(v_a)
    w = v_b - (e1 * v_b).sum(-1, keepdims=True) * e1
    e2 = unit(w)
    c1 = (v_ab * e1).sum(-1, keepdims=True)
    c2 = (v_ab * e2).sum(-1, keepdims=True)
    rem = v_ab - c1 * e1 - c2 * e2
    return rem, np.linalg.norm(rem, axis=-1) ** 2       # v_ab is unit -> frac in [0,1]


def analyse(tag, P, pairs, VA, disp_gap, log_prefix=""):
    """P: (n_pair, d) per-pair mean representations."""
    g = P.mean(0, keepdims=True)
    mag = np.linalg.norm(P - g, axis=1)          # signal magnitude per pair
    v_ab = unit(P - g)
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    rem, frac = decompose(v_ab, VA[ia], VA[ib])

    R = unit(rem)
    cos_raw = mean_pairwise_cosine(R)
    shared = unit(R.mean(0, keepdims=True))
    R2 = R - (R * shared).sum(-1, keepdims=True) * shared
    keep = np.linalg.norm(R2, axis=-1) ** 2            # share surviving shared removal
    R2u = unit(R2)
    cos_after = mean_pairwise_cosine(R2u)

    print(f"\n  {log_prefix}{tag}")
    print(f"    orthogonal remainder, mean fraction of v_AB : {frac.mean():.3f}  "
          f"(median {np.median(frac):.3f})")
    print(f"    mean pairwise cosine among remainders        : {cos_raw:+.3f}"
          f"   {'-> SHARED offset dominates' if cos_raw > 0.3 else '-> largely pair-specific'}")
    print(f"    variance surviving removal of shared component: {keep.mean():.3f}")
    print(f"    mean pairwise cosine after removal           : {cos_after:+.3f}")
    ps = frac * keep
    hi = mag >= np.quantile(mag, 0.75)
    # the all-pairs figure is inflated by low-signal pairs, so the high-SNR
    # quartile is the defensible estimate
    print(f"    => pair-specific orthogonal share of v_AB    : {ps.mean():.3f}"
          f"   (high-SNR quartile: {ps[hi].mean():.3f})")
    print(f"    r(||dev||, orth_frac) = {np.corrcoef(mag, frac)[0,1]:+.3f}"
          f"   {'<- low-signal pairs inflate the remainder' if np.corrcoef(mag,frac)[0,1] < -0.3 else ''}")
    return {"tag": tag, "orth_frac": float(frac.mean()),
            "orth_frac_hiSNR": float(frac[hi].mean()),
            "pair_specific_hiSNR": float(ps[hi].mean()),
            "r_mag_orth": float(np.corrcoef(mag, frac)[0, 1]), "cos_raw": cos_raw,
            "surviving": float(keep.mean()), "cos_after": cos_after,
            "pair_specific": float((frac * keep).mean()),
            "_frac": frac, "_keep": keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite"])
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer-mode", choices=["gap", "ceiling"], default="gap",
                    help="gap: layer of maximum erasure gap. ceiling: layer where "
                         "identity is most strongly encoded in singles.")
    ap.add_argument("--layer", type=int, default=None, help="override layer choice")
    args = ap.parse_args()

    rows = []
    for model in args.models:
        pr = pd.read_csv(OUT_DIR / f"{model}_identity_probe{args.tag}.csv")
        pr["gap"] = pr.ceiling_auc_real - pr.pair_auc_real
        if args.layer is not None:
            L = args.layer
        elif args.layer_mode == "ceiling":
            L = int(pr.groupby("layer").ceiling_auc_real.mean().idxmax())
        else:
            L = int(pr.groupby("layer").gap.mean().idxmax())
        rng = np.random.default_rng(args.seed)
        log.info(f"[{model}] emergence at {args.layer_mode}-peak layer L{L}")
        sums, pairs, names, S, B, half, pats = collect(model, L, rng)

        # constituent directions from single-identity prompts
        VA = unit(S.mean(0) - S.mean(0).mean(0, keepdims=True))

        # displacement per identity, for the trade-off analysis
        disp = np.linalg.norm(S - B[:, None, :], axis=2).mean(0)
        disp = disp / disp.mean()
        ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
        disp_gap = np.abs(disp[ia] - disp[ib])

        print("=" * 88)
        print(f"{model.upper()}  layer {L}   {len(pairs)} pairs")
        print("=" * 88)

        real = analyse("OBSERVED pair representations", sums[0], pairs, VA, disp_gap)
        _dev = sums[0] - sums[0].mean(0, keepdims=True)
        _perm = rng.permutation(len(ia))
        _, _fm = decompose(unit(_dev), VA[ia[_perm]], VA[ib[_perm]])
        print(f"    MISMATCHED constituents (specificity control) : {_fm.mean():.3f}"
              f"   vs correct {real['orth_frac']:.3f}, chance {1-2/VA.shape[1]:.4f}")

        # additive null through the identical pipeline
        Sm, Bm = S.mean(0), B.mean(0)
        add = Sm[ia] + Sm[ib] - Bm
        null = analyse("ADDITIVE NULL (ind_A + ind_B - base)", add, pairs, VA, disp_gap)

        # held-out check: is v_AB from half 1 recoverable on half 2?
        v1 = unit(sums[0] - sums[0].mean(0, keepdims=True))
        h2 = sums[1] - sums[1].mean(0, keepdims=True)
        own = np.einsum('ij,ij->i', h2, v1)
        perm = rng.permutation(len(v1))
        other = np.einsum('ij,ij->i', h2, v1[perm])
        print(f"\n  HELD-OUT: v_AB fitted on {len(half)} prompts, tested on the other "
              f"{len(pats)-len(half)}")
        print(f"    projection onto OWN pair direction   : {own.mean():.3f}")
        print(f"    projection onto a DIFFERENT pair's   : {other.mean():.3f}")
        print(f"    separation {own.mean()-other.mean():.3f}   "
              f"-> pair directions {'generalise' if own.mean() > 2*abs(other.mean()) else 'DO NOT generalise'}")

        # does emergence trade off against flattening?
        q = pd.DataFrame({"gap": disp_gap, "orth": real["_frac"],
                          "pair_specific": real["_frac"] * real["_keep"]})
        q["bin"] = pd.qcut(q.gap, 4, labels=["Q1 balanced", "Q2", "Q3", "Q4 dominated"])
        print(f"\n  TRADE-OFF: pair-specific orthogonal share by displacement-gap quartile")
        for b, gg in q.groupby("bin", observed=True):
            print(f"    {b:<14} mean gap {gg.gap.mean():.3f}   "
                  f"pair-specific orth {gg.pair_specific.mean():.3f}")
        print(f"    r(displacement gap, pair-specific orth) = "
              f"{q.gap.corr(q.pair_specific):+.3f}")

        for r in (real, null):
            rows.append({k: v for k, v in r.items() if not k.startswith("_")}
                        | {"model": model, "layer": L})
        print()

    pd.DataFrame(rows).to_csv(OUT_DIR / f"emergence_{args.layer_mode}{args.tag}.csv", index=False)
    print(f"saved -> {OUT_DIR}/emergence_{args.layer_mode}{args.tag}.csv")


if __name__ == "__main__":
    main()
