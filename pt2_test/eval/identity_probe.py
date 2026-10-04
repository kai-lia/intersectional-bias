"""
Is an identity still linearly recoverable from the representation of a PAIR?

Motivation
----------
Earlier work in this project asserted "erasure" from a lean/dominance measure
that turned out to correlate with raw displacement magnitude at r = 0.926 --
i.e. it largely restated "whichever identity moves the residual stream further
wins".  A probe answers the question directly: given the AB representation, can
we still detect that A is present?  That is a recoverability claim, measured,
rather than a dominance claim asserted.

Method: difference-in-means directions (run this before fitted probes)
---------------------------------------------------------------------
For identity i at layer L,
    dir_i = normalise( mean(singles_i) - mean(singles_all) )
No fitting, so no overfitting risk with only 37 prompts per identity, and it is
standard practice for linearly-encoded concepts.  Fitted logistic probes are a
follow-up if this shows signal.

LEAVE-ONE-TEMPLATE-OUT: dir_i for scoring pattern p is built from the other 36
patterns.  There is no fitting, but the direction would otherwise be estimated
from the same template wording it is evaluated on.

Scoring is threshold-free.  For each (identity, layer):
    ceiling_auc  -- separating single-identity-i residuals from other identities'
                    singles.  What the direction can do at all at this layer.
    pair_auc     -- separating PAIR residuals that contain i from pairs that do
                    not.  Recoverability of i from a composed representation.
Reporting pair_auc against ceiling_auc matters: a low pair_auc at a layer where
the ceiling is also low says nothing about erasure.

Controls
--------
  shuffled   identity labels permuted before building directions; establishes
             the chance floor under the same leave-one-template-out scheme.
  random     norm-matched random directions.  Required here specifically because
             dominance and displacement correlate at 0.926 -- without this we
             cannot show recovery is anything more than projection onto whatever
             happens to be large.
  self-pair  for pairs NOT containing i, the score distribution acts as a
             within-direction baseline, controlling for how well estimated
             dir_i is rather than comparing across identities.

Memory: combo vectors are projected onto all 112 directions immediately after
each shard is read, so 12,432 x 4096 floats collapse to 12,432 x 112 and no
layer ever holds the full combo matrix.

Output: {model}_identity_probe{tag}.csv  (one row per identity x layer x arm)
        {model}_identity_probe_pairs{tag}.csv (per pair x member x layer scores)
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


def shard_ids(model):
    pats = sorted(int(re.match(rf"{model}_pattern(\d+)\.done", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*.done"))
    layers = sorted(int(re.match(rf"{model}_pattern{pats[0]}_layer(\d+)\.npz", f.name).group(1))
                    for f in ACT_DIR.glob(f"{model}_pattern{pats[0]}_layer*.npz"))
    return pats, layers


def auc(pos, neg):
    """Rank-based AUC, no sklearn dependency."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    a = np.concatenate([pos, neg])
    r = pd.Series(a).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run_layer(model, layer, pats, rng, n_random=8):
    """Returns (per-identity rows, per-pair-member rows) for one layer."""
    # --- pass 1: collect singles for every pattern (small) --------------------
    singles, names = {}, None
    for pid in pats:
        p = ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            sv = z["singles_vec"].astype(np.float32)
            ss = [str(x) for x in z["singles_stigma"]]
        if names is None:
            names = ss
            idx = {n: k for k, n in enumerate(names)}
        singles[pid] = sv[[idx[n] for n in names]]        # align row order
    S = np.stack([singles[p] for p in sorted(singles)])    # (n_pat, n_id, d)
    pat_order = sorted(singles)
    n_pat, n_id, d = S.shape

    def directions(mat, exclude_i):
        """Leave-one-template-out diff-in-means directions from mat (n_pat,n_id,d)."""
        keep = [k for k in range(n_pat) if k != exclude_i]
        m = mat[keep].mean(0)                              # (n_id, d) per-identity mean
        g = m.mean(0, keepdims=True)                       # global mean
        v = m - g
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v                                           # (n_id, d) unit rows

    # shuffled-label control: permute identity assignment before averaging
    perm = rng.permutation(n_id)
    S_shuf = S[:, perm, :]

    # norm-matched random directions
    R = rng.normal(size=(n_random, d)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)

    id_rows, pair_rows = [], []
    # accumulate scores across held-out patterns
    ceil_pos = {a: {i: [] for i in range(n_id)} for a in ("real", "shuf")}
    ceil_neg = {a: {i: [] for i in range(n_id)} for a in ("real", "shuf")}
    pair_in  = {a: {i: [] for i in range(n_id)} for a in ("real", "shuf", "rand")}
    pair_out = {a: {i: [] for i in range(n_id)} for a in ("real", "shuf", "rand")}

    for k, pid in enumerate(pat_order):
        D_real = directions(S, k)
        D_shuf = directions(S_shuf, k)

        # ceiling: score this pattern's singles
        for arm, D, mat in (("real", D_real, S[k]), ("shuf", D_shuf, S_shuf[k])):
            sc = mat @ D.T                                  # (n_id identities, n_id dirs)
            for i in range(n_id):
                ceil_pos[arm][i].append(sc[i, i])
                ceil_neg[arm][i].extend(np.delete(sc[:, i], i))

        # --- pass 2: this pattern's combos, projected immediately -------------
        p = ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz"
        with np.load(p, allow_pickle=True) as z:
            cv = z["combo_vec"].astype(np.float32)
            c1 = [str(x) for x in z["combo_stigma1"]]
            c2 = [str(x) for x in z["combo_stigma2"]]
        P_real = cv @ D_real.T                              # (n_combo, n_id)
        P_shuf = cv @ D_shuf.T
        P_rand = cv @ R.T                                   # (n_combo, n_random)
        del cv; gc.collect()

        i_of = {n: j for j, n in enumerate(names)}
        a_idx = np.array([i_of[x] for x in c1])
        b_idx = np.array([i_of[x] for x in c2])
        for i in range(n_id):
            member = (a_idx == i) | (b_idx == i)
            pair_in["real"][i].append(P_real[member, i])
            pair_out["real"][i].append(P_real[~member, i])
            pair_in["shuf"][i].append(P_shuf[member, i])
            pair_out["shuf"][i].append(P_shuf[~member, i])
            pair_in["rand"][i].append(P_rand[member, i % n_random])
            pair_out["rand"][i].append(P_rand[~member, i % n_random])

        # per-pair member scores, real arm only, for the recovery-gap regression
        for j in range(len(a_idx)):
            pair_rows.append({"layer": layer, "pattern_id": pid,
                              "s1": c1[j], "s2": c2[j],
                              "score_s1": float(P_real[j, a_idx[j]]),
                              "score_s2": float(P_real[j, b_idx[j]])})
        del P_real, P_shuf, P_rand; gc.collect()

    for i, nm in enumerate(names):
        row = {"layer": layer, "identity": nm}
        for arm in ("real", "shuf"):
            row[f"ceiling_auc_{arm}"] = auc(np.array(ceil_pos[arm][i]),
                                            np.array(ceil_neg[arm][i]))
        for arm in ("real", "shuf", "rand"):
            row[f"pair_auc_{arm}"] = auc(np.concatenate(pair_in[arm][i]),
                                         np.concatenate(pair_out[arm][i]))
        id_rows.append(row)
    return id_rows, pair_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite"])
    ap.add_argument("--layer-stride", type=int, default=4,
                    help="1 = every layer. Default 4 gives the curve shape quickly.")
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-pairs", action="store_true",
                    help="also write per-pair member scores (large)")
    args = ap.parse_args()

    for model in args.models:
        pats, layers = shard_ids(model)
        use = layers[::args.layer_stride]
        log.info(f"[{model}] {len(pats)} patterns, {len(layers)} layers, probing {len(use)}: {use}")
        rng = np.random.default_rng(args.seed)
        all_id, all_pair = [], []
        for L in use:
            import time; t = time.time()
            idr, pr = run_layer(model, L, pats, rng)
            all_id.extend(idr)
            if args.save_pairs:
                all_pair.extend(pr)
            df = pd.DataFrame(idr)
            log.info(f"[{model}] layer {L} in {time.time()-t:.0f}s  "
                     f"ceiling {df.ceiling_auc_real.mean():.3f}  "
                     f"pair {df.pair_auc_real.mean():.3f}  "
                     f"(shuf {df.pair_auc_shuf.mean():.3f}, rand {df.pair_auc_rand.mean():.3f})")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_id).to_csv(OUT_DIR / f"{model}_identity_probe{args.tag}.csv", index=False)
        log.info(f"[{model}] saved -> {OUT_DIR}/{model}_identity_probe{args.tag}.csv")
        if args.save_pairs:
            pd.DataFrame(all_pair).to_csv(
                OUT_DIR / f"{model}_identity_probe_pairs{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
