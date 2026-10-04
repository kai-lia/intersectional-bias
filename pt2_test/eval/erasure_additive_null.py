"""
Is the erasure/displacement relationship linear algebra, or learned composition?

The worry, stated precisely
---------------------------
displacement_i    = ||ind_i - base||
probe direction_i = normalise( mean(ind_i) - mean(all identities) )

These are not the same vector, but they are close enough that the observed
r(gap, displacement) ~= -0.7 may be definitional rather than empirical.  Under
additive interference, detection SNR along v_i goes roughly as ||v_i|| divided
by the interference projected onto v_i.  A negative gap/displacement relation
then follows from geometry alone, with the model having learned no interaction
whatsoever.

The test
--------
Build pair representations that contain NO learned interaction:

    synthetic_AB = ind_A + ind_B - base            (pure additive composition)
                   + noise matched to the empirical residual scale

then run the identical probe pipeline and compute r(gap, displacement) under
that null.  Two variants are reported:

    additive+noise  matched total variance, so the null has as much "going on"
                    as the real representation, just no interaction
    additive only   zero noise; the pure-geometry floor

Interpretation
--------------
    null ~= observed   -> the relationship is linear algebra.  The finding
                          becomes descriptive (erasure follows magnitude, and
                          magnitude tracks attributed blame) -- still policy
                          relevant, but NOT a claim about nonlinear composition.
    null much weaker   -> the model is doing something beyond additive
                          interference, and the mechanism claim stands.

Cheap, because every vector already exists on disk.
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


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    a = np.concatenate([pos, neg])
    r = pd.Series(a).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def shard_ids(model):
    pats = sorted(int(re.match(rf"{model}_pattern(\d+)\.done", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*.done"))
    layers = sorted(int(re.match(rf"{model}_pattern{pats[0]}_layer(\d+)\.npz", f.name).group(1))
                    for f in ACT_DIR.glob(f"{model}_pattern{pats[0]}_layer*.npz"))
    return pats, layers


def run(model, layer, rng, chunk=2048):
    pats, _ = shard_ids(model)

    # pass 1: singles + base, and the empirical residual scale
    singles, bases, names, resid_sq, resid_n = {}, {}, None, 0.0, 0
    for pid in pats:
        p = ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            sv = z["singles_vec"].astype(np.float32)
            ss = [str(x) for x in z["singles_stigma"]]
            bv = z["base_vec"][0].astype(np.float32)
            cv = z["combo_vec"].astype(np.float32)
            c1 = [str(x) for x in z["combo_stigma1"]]
            c2 = [str(x) for x in z["combo_stigma2"]]
        if names is None:
            names = ss
        idx = {n: k for k, n in enumerate(ss)}
        order = [idx[n] for n in names]
        singles[pid], bases[pid] = sv[order], bv
        a = np.array([idx[x] for x in c1]); b = np.array([idx[x] for x in c2])
        for s in range(0, len(a), chunk):
            e = slice(s, min(s + chunk, len(a)))
            r = cv[e] - (sv[a[e]] + sv[b[e]] - bv)
            resid_sq += float((r.astype(np.float64) ** 2).sum()); resid_n += r.size
        del cv, sv; gc.collect()

    sigma = float(np.sqrt(resid_sq / resid_n))
    pat_order = sorted(singles)
    S = np.stack([singles[p] for p in pat_order])
    n_pat, n_id, d = S.shape
    log.info(f"  [{model} L{layer}] empirical residual sigma = {sigma:.4f}")

    def directions(exclude):
        keep = [k for k in range(n_pat) if k != exclude]
        m = S[keep].mean(0)
        v = m - m.mean(0, keepdims=True)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    arms = ["real", "add_noise", "add_only"]
    ceil_pos = {i: [] for i in range(n_id)}
    ceil_neg = {i: [] for i in range(n_id)}
    p_in = {a: {i: [] for i in range(n_id)} for a in arms}
    p_out = {a: {i: [] for i in range(n_id)} for a in arms}

    for k, pid in enumerate(pat_order):
        D = directions(k)
        sc = S[k] @ D.T
        for i in range(n_id):
            ceil_pos[i].append(sc[i, i])
            ceil_neg[i].extend(np.delete(sc[:, i], i))

        p = ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz"
        with np.load(p, allow_pickle=True) as z:
            cv = z["combo_vec"].astype(np.float32)
            c1 = [str(x) for x in z["combo_stigma1"]]
            c2 = [str(x) for x in z["combo_stigma2"]]
        idx = {n: j for j, n in enumerate(names)}
        a = np.array([idx[x] for x in c1]); b = np.array([idx[x] for x in c2])

        proj = {arm: np.empty((len(a), n_id), np.float32) for arm in arms}
        for s in range(0, len(a), chunk):
            e = slice(s, min(s + chunk, len(a)))
            add = S[k][a[e]] + S[k][b[e]] - bases[pid]
            proj["real"][e]      = cv[e] @ D.T
            proj["add_only"][e]  = add @ D.T
            proj["add_noise"][e] = (add + rng.normal(0, sigma, add.shape).astype(np.float32)) @ D.T
            del add
        del cv; gc.collect()

        for i in range(n_id):
            member = (a == i) | (b == i)
            for arm in arms:
                p_in[arm][i].append(proj[arm][member, i])
                p_out[arm][i].append(proj[arm][~member, i])
        del proj; gc.collect()

    rows = []
    for i, nm in enumerate(names):
        r = {"identity": nm, "ceiling": auc(np.array(ceil_pos[i]), np.array(ceil_neg[i]))}
        for arm in arms:
            r[f"pair_{arm}"] = auc(np.concatenate(p_in[arm][i]), np.concatenate(p_out[arm][i]))
            r[f"gap_{arm}"] = r["ceiling"] - r[f"pair_{arm}"]
        rows.append(r)
    return pd.DataFrame(rows), sigma


def shift_norms(model, n_layers=6):
    pats, al = shard_ids(model)
    acc = {}
    for L in [al[i] for i in np.linspace(0, len(al) - 1, n_layers).astype(int)]:
        pl = {}
        for pid in pats:
            p = ACT_DIR / f"{model}_pattern{pid}_layer{L}.npz"
            if not p.exists():
                continue
            with np.load(p, allow_pickle=True) as z:
                sv, ss, bv = z["singles_vec"], z["singles_stigma"], z["base_vec"][0]
                nrm = np.linalg.norm(sv.astype(np.float32) - bv.astype(np.float32), axis=1)
                for s, v in zip(ss, nrm):
                    pl.setdefault(str(s), []).append(float(v))
        mu = np.mean([v for vs in pl.values() for v in vs])
        for s, vs in pl.items():
            acc.setdefault(s, []).append(np.mean(vs) / mu)
    return pd.Series({s: np.mean(v) for s, v in acc.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = []
    for model in args.models:
        probe = pd.read_csv(OUT_DIR / f"{model}_identity_probe{args.tag}.csv")
        probe["gap"] = probe.ceiling_auc_real - probe.pair_auc_real
        peak = int(probe.groupby("layer").gap.mean().idxmax())
        log.info(f"[{model}] gap peak at L{peak}; running additive null there")
        df, sigma = run(model, peak, np.random.default_rng(args.seed))
        df["disp"] = df.identity.map(shift_norms(model))
        df = df.dropna(subset=["disp"])

        print("\n" + "=" * 84)
        print(f"{model.upper()}  layer {peak}   (empirical residual sigma {sigma:.4f})")
        print("=" * 84)
        print(f"  {'arm':<26}{'mean gap':>10}{'r(gap, displacement)':>24}")
        for arm, lbl in [("real", "observed"),
                         ("add_noise", "additive + matched noise"),
                         ("add_only", "additive only (no noise)")]:
            g, r = df[f"gap_{arm}"], df[f"gap_{arm}"].corr(df["disp"])
            print(f"  {lbl:<26}{g.mean():>10.3f}{r:>24.3f}")
            out.append({"model": model, "layer": peak, "arm": arm,
                        "mean_gap": g.mean(), "r_gap_disp": r, "sigma": sigma})
        r_obs = df.gap_real.corr(df["disp"]); r_null = df.gap_add_noise.corr(df["disp"])
        print(f"\n  observed {r_obs:+.3f} vs additive null {r_null:+.3f}  ->  ", end="")
        if abs(r_null) >= 0.8 * abs(r_obs):
            print("LINEAR ALGEBRA: the null reproduces it; claim must be descriptive")
        elif abs(r_null) <= 0.5 * abs(r_obs):
            print("BEYOND ADDITIVE: null is much weaker; mechanism claim stands")
        else:
            print("PARTIAL: the null explains some but not all of it")
        df.to_csv(OUT_DIR / f"{model}_erasure_additive_null{args.tag}.csv", index=False)

    pd.DataFrame(out).to_csv(OUT_DIR / f"erasure_additive_null{args.tag}.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}/erasure_additive_null{args.tag}.csv")


if __name__ == "__main__":
    main()
