"""
Is the behaviour/representation dissociation real, or an artefact of measuring
the wrong subspace?

The problem
-----------
The dissociation was measured with a full-residual-stream norm ratio

    non_additive_frac = ||combo - (ind1 + ind2 - base)|| / ||combo - base||

computed over all d dimensions.  That norm is dominated by whichever directions
carry the most variance, and those need not be identity-related at all -- token
identity, position, template syntax.  If the numerator is mostly non-identity
variance, the measure would fail to track behaviour EVEN IF identity composition
tracks behaviour perfectly.  A reliable measure can still be reliably measuring
the wrong thing, so the split-half control does not address this.

What this does
--------------
Recomputes exactly the same quantity inside the IDENTITY SUBSPACE: the span of
the 112 difference-in-means identity directions from identity_probe.py.  Both
measures are computed in one pass over the same layers, so the only thing that
differs between them is the subspace -- not layer selection, not the estimator,
not the scenario set.

A THIRD contaminant, found after the first two were ruled out: r carries a large
per-pattern CONSTANT offset (~80-85% of its energy).  predicted = ind1+ind2-base
puts coefficients 1,1,-1 on the shared anisotropic direction, which cancels only
if its magnitude matches across conditions -- and it does not, because compound
prompts are longer than singles, which are longer than base.  That offset is
identical for every pair in a pattern, so it inflates all norms and swamps the
per-pair variation.  It survives split-half reliability (it is perfectly stable)
and sits largely inside the identity subspace, so neither earlier control caught
it.  Every measure is therefore reported raw AND offset-removed; the raw and
offset-removed versions correlate only ~0.35, so the raw metric is mostly offset.

    raw_frac  ||r||        / ||combo - base||          (full space)
    sub_frac  ||P_id r||   / ||P_id (combo - base)||   (identity subspace)
    in_share  ||P_id r||^2 / ||r||^2                   (how much of the residual
                                                        is identity-related at all)

P_id is built by QR-orthonormalising the direction set, so ||P_id x|| is a true
subspace norm rather than a coordinate norm in a non-orthogonal frame.  (The
probe's AUC scoring uses raw coordinates, which is correct for scoring but not
for norms.)

Leave-one-template-out is preserved: directions used to score pattern p are
estimated from the other 36 patterns, exactly as in identity_probe.py.

Reading the result
------------------
  dissociation survives in-subspace  -> it is real; output-level audits do not
                                        track internal composition
  correlation appears in-subspace    -> the full-space null was a subspace
                                        artefact, and the finding becomes
                                        "measure inside the identity subspace"
  in_share very small                -> the original measure was mostly
                                        non-identity variance, which is itself
                                        worth reporting whichever way it lands
"""
import argparse
import gc
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def shard_ids(model):
    pats = sorted(int(re.match(rf"{model}_pattern(\d+)\.done", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*.done"))
    layers = sorted(int(re.match(rf"{model}_pattern{pats[0]}_layer(\d+)\.npz", f.name).group(1))
                    for f in ACT_DIR.glob(f"{model}_pattern{pats[0]}_layer*.npz"))
    return pats, layers


def peak_layer(model, tag="_full"):
    """Layer of maximum erasure gap -- where identity is most at stake."""
    p = OUT_DIR / f"{model}_identity_probe{tag}.csv"
    d = pd.read_csv(p)
    d["gap"] = d.ceiling_auc_real - d.pair_auc_real
    return int(d.groupby("layer").gap.mean().idxmax())


def readout_direction(model):
    """Unit vector in activation space whose component sets the yes/no logit gap.

    For RMSNorm the final logit difference is

        logit_yes - logit_no = ((x / rms(x)) * norm_weight) . dw
                             = (1 / rms(x)) * x . (norm_weight * dw)

    so up to the positive scalar 1/rms(x) -- which changes magnitude but not
    direction -- the logit gap is a LINEAR projection of x onto

        w_eff = norm_weight * dw,     dw = mean(yes rows) - mean(no rows)

    That makes the decomposition well defined: the part of a residual lying
    along w_eff is the part that can move the answer, and everything orthogonal
    to it cannot, no matter how large.  LayerNorm additionally subtracts the
    mean, which is a projection killing the all-ones direction, so w_eff is
    centred in that case.

    Returns None when the head has not been extracted for this model.
    """
    p = ROOT.parent / "data" / f"lm_head_yesno_{model}.npz"
    legacy = ROOT.parent / "data" / "lm_head_yesno.npz"
    p = p if p.exists() else legacy
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        rows = z["lm_head_rows"].astype(np.float32)
        is_yes = z["is_yes"].astype(bool)
        nw = z["norm_weight"].astype(np.float32)
        ntype = str(z["norm_type"])
    dw = rows[is_yes].mean(0) - rows[~is_yes].mean(0)
    w = nw * dw
    if "LayerNorm" in ntype:
        w = w - w.mean()
    return w / (np.linalg.norm(w) + 1e-9)


def directions(S, exclude_k):
    """LOTO diff-in-means identity directions; (n_id, d) unit rows."""
    keep = [k for k in range(S.shape[0]) if k != exclude_k]
    m = S[keep].mean(0)
    v = m - m.mean(0, keepdims=True)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


def run_layer(model, layer, pats, w):
    """Per-(pair, pattern) raw and in-subspace non-additive fractions."""
    singles, base, names = {}, {}, None
    for pid in pats:
        p = ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            sv = z["singles_vec"].astype(np.float32)
            ss = [str(x) for x in z["singles_stigma"]]
            base[pid] = z["base_vec"].astype(np.float32).reshape(-1)
        if names is None:
            names = ss
            idx = {n: k for k, n in enumerate(names)}
        singles[pid] = sv[[idx[n] for n in names]]
    pat_order = sorted(singles)
    S = np.stack([singles[p] for p in pat_order])          # (n_pat, n_id, d)
    nid = {n: i for i, n in enumerate(names)}

    rows = []
    for k, pid in enumerate(pat_order):
        D = directions(S, k)                                # (n_id, d)
        # orthonormal basis for span(D) so subspace norms are true norms
        Q, _ = np.linalg.qr(D.T)                            # (d, r)
        b = base[pid]
        with np.load(ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz", allow_pickle=True) as z:
            cv = z["combo_vec"].astype(np.float32)
            c1 = [str(x) for x in z["combo_stigma1"]]
            c2 = [str(x) for x in z["combo_stigma2"]]
        i1 = np.array([nid[a] for a in c1]); i2 = np.array([nid[a] for a in c2])
        # chunked over combo rows: the full-size intermediates would be ~800 MB
        # per pattern, and extraction may be holding memory concurrently
        CH, d = 2000, S.shape[2]
        # PASS 1 -- the per-pattern offset.
        # predicted = ind1 + ind2 - base puts coefficients 1,1,-1 on the shared
        # anisotropic component, which cancels only if its magnitude is equal
        # across conditions.  It is not: compound prompts are longer than
        # singles, which are longer than base.  So r carries a large fixed
        # direction (~80-85% of its energy) that is identical for every pair in
        # the pattern.  It inflates every norm and swamps the per-pair variation
        # the correlation with behaviour depends on.  Accumulated in chunks so
        # the full (n_combo, d) residual is never materialised.
        r_sum = np.zeros(d, np.float64); sh_sum = np.zeros(d, np.float64); n_acc = 0
        for s in range(0, len(cv), CH):
            c = cv[s:s + CH]
            r = c - (S[k][i1[s:s + CH]] + S[k][i2[s:s + CH]] - b)
            sh = c - b
            r_sum += r.sum(0); sh_sum += sh.sum(0); n_acc += len(c)
            del r, sh
        r_bar = (r_sum / n_acc).astype(np.float32)
        sh_bar = (sh_sum / n_acc).astype(np.float32)

        # PASS 2 -- norms, raw and offset-removed, in each geometry
        acc = {k_: [] for k_ in ("raw_n", "raw_d", "cen_n", "cen_d", "sub_n", "sub_d",
                                 "subc_n", "subc_d", "rd_n", "rd_d", "rdc_n", "rdc_d")}
        for s in range(0, len(cv), CH):
            c = cv[s:s + CH]
            r = c - (S[k][i1[s:s + CH]] + S[k][i2[s:s + CH]] - b)
            sh = c - b
            rc = r - r_bar; shc = sh - sh_bar
            acc["raw_n"].append(np.linalg.norm(r, axis=1))
            acc["raw_d"].append(np.linalg.norm(sh, axis=1))
            acc["cen_n"].append(np.linalg.norm(rc, axis=1))
            acc["cen_d"].append(np.linalg.norm(shc, axis=1))
            acc["sub_n"].append(np.linalg.norm(r @ Q, axis=1))
            acc["sub_d"].append(np.linalg.norm(sh @ Q, axis=1))
            acc["subc_n"].append(np.linalg.norm(rc @ Q, axis=1))
            acc["subc_d"].append(np.linalg.norm(shc @ Q, axis=1))
            if w is not None:
                acc["rd_n"].append(r @ w); acc["rd_d"].append(sh @ w)
                acc["rdc_n"].append(rc @ w); acc["rdc_d"].append(shc @ w)
            del r, sh, rc, shc
        a = {k_: (np.concatenate(v) if v else None) for k_, v in acc.items()}
        del cv; gc.collect()

        def frac(num, den):
            return num / np.where(np.abs(den) > 1e-9, np.abs(den), np.nan)

        out = {
            "pattern_id": pid, "s1": c1, "s2": c2,
            "raw_frac": frac(a["raw_n"], a["raw_d"]),
            "cen_frac": frac(a["cen_n"], a["cen_d"]),
            "sub_frac": frac(a["sub_n"], a["sub_d"]),
            "sub_cen_frac": frac(a["subc_n"], a["subc_d"]),
            "in_share": (a["sub_n"] ** 2) / np.where(a["raw_n"] > 1e-9, a["raw_n"] ** 2, np.nan),
            "in_share_cen": (a["subc_n"] ** 2) / np.where(a["cen_n"] > 1e-9, a["cen_n"] ** 2, np.nan),
        }
        if w is not None:
            # w is a unit vector, so (r.w)^2 / ||r||^2 is exactly the share of the
            # residual lying along the direction that sets the answer
            out["readout_share"] = (a["rd_n"] ** 2) / np.where(a["raw_n"] > 1e-9, a["raw_n"] ** 2, np.nan)
            out["readout_share_cen"] = (a["rdc_n"] ** 2) / np.where(a["cen_n"] > 1e-9, a["cen_n"] ** 2, np.nan)
            out["readout_frac"] = frac(np.abs(a["rd_n"]), a["rd_d"])
            out["readout_frac_cen"] = frac(np.abs(a["rdc_n"]), a["rdc_d"])
            # UNNORMALISED magnitude along the answer direction.  The ratio
            # versions above divide by |shift . w|, which both injects noise
            # (the denominator passes through zero) and removes the magnitude
            # that actually carries the signal -- |r . w| reproduces the decoded
            # log-odds residual at r = 0.99, the ratio does not.
            out["readout_abs"] = np.abs(a["rd_n"])
        rows.append(pd.DataFrame(out))
        log.info(f"    pattern {pid}: done")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="explicit layers; overrides --layer-mode")
    ap.add_argument("--layer-mode", default="peak", choices=["peak", "deepest"],
                    help="peak: max erasure gap (identity most at stake).  "
                         "deepest: final hidden state, where the readout direction "
                         "is exact rather than a logit-lens approximation")
    ap.add_argument("--probe-tag", default="_full",
                    help="tag of the identity_probe file used to locate the peak layer")
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--beh-tag", default="_polarityfixed")
    args = ap.parse_args()

    for model in args.models:
        pats, all_layers = shard_ids(model)
        w = readout_direction(model)
        if args.layers:
            layers = args.layers
        elif args.layer_mode == "deepest":
            layers = [max(all_layers)]
        else:
            layers = [peak_layer(model, args.probe_tag)]
        log.info(f"[{model}] layers {layers} of {len(all_layers)} available, {len(pats)} patterns"
                 + ("" if w is not None else "  [no lm_head yet -- readout_share skipped]"))

        out = []
        for L in layers:
            log.info(f"  layer {L}")
            d = run_layer(model, L, pats, w)
            d["layer"] = L
            out.append(d)
        d = pd.concat(out, ignore_index=True)
        d["pair"] = [tuple(sorted([a, b])) for a, b in zip(d.s1, d.s2)]

        aggs = {c: (c, "mean") for c in
                ["raw_frac", "cen_frac", "sub_frac", "sub_cen_frac",
                 "in_share", "in_share_cen", "readout_share", "readout_share_cen",
                 "readout_frac", "readout_frac_cen", "readout_abs"]
                if c in d.columns}
        per_pair = d.groupby("pair").agg(**aggs).reset_index()
        p = OUT_DIR / f"{model}_subspace_additivity{args.tag}.csv"
        per_pair.assign(stigma1=[k[0] for k in per_pair.pair],
                        stigma2=[k[1] for k in per_pair.pair]).drop(columns="pair").to_csv(p, index=False)

        beh = pd.read_csv(OUT_DIR / f"{model}_behavioral_additivity{args.beh_tag}.csv")
        beh["pair"] = [tuple(sorted([a, b])) for a, b in zip(beh.stigma1, beh.stigma2)]
        beh["beh_nonadd"] = beh[["behavioral_residual12", "behavioral_residual21"]].mean(axis=1).abs()
        j = per_pair.merge(beh[["pair", "beh_nonadd"]], on="pair")

        print("=" * 78)
        print(f"{model.upper()}   layers {layers}   n = {len(j)} pairs")
        print(f"  identity-subspace share of the residual: raw {j.in_share.mean():.4f}"
              f"   offset-removed {j.in_share_cen.mean():.4f}   (112 directions)")
        print(f"  mean non-additive fraction: raw {j.raw_frac.mean():.3f}"
              f"   offset-removed {j.cen_frac.mean():.3f}")
        print()
        print(f"  CORRELATION WITH BEHAVIOUR                  raw     offset-removed")
        print(f"    full space                            {j.beh_nonadd.corr(j.raw_frac):+.3f}"
              f"          {j.beh_nonadd.corr(j.cen_frac):+.3f}")
        print(f"    identity subspace                     {j.beh_nonadd.corr(j.sub_frac):+.3f}"
              f"          {j.beh_nonadd.corr(j.sub_cen_frac):+.3f}")
        print(f"  r(raw, offset-removed) = {j.raw_frac.corr(j.cen_frac):+.3f}"
              f"   (low => the raw metric is mostly the fixed offset)")
        if "readout_share" in j.columns:
            print()
            print(f"  READOUT share of the residual along the answer direction:"
                  f" raw {j.readout_share.mean():.5f}   offset-removed {j.readout_share_cen.mean():.5f}")
            print(f"    => {100*(1-j.readout_share_cen.mean()):.2f}% of the non-additive work"
                  f" is orthogonal to what sets the answer (offset-removed)")
            print(f"    r(behaviour, |r.w| unnormalised)   = {j.beh_nonadd.corr(j.readout_abs):+.3f}"
                  f"   <- the component that reaches the answer")
            print(f"    r(behaviour, |r.w|/|shift.w| ratio) = {j.beh_nonadd.corr(j.readout_frac):+.3f}"
                  f"   (ratio form; denominator crosses zero)")
        print(f"  -> {p.name}")
        print()


if __name__ == "__main__":
    main()
