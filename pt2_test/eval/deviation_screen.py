"""
Deviation-from-additivity screen for intersectional pairs.

The claim behind the screen
---------------------------
The subspace-additivity work fitted two global scalars, a and b, such that
`r_AB ~ a * r_A + b * r_B` at the model's peak layer, with held-out R^2 in the
0.86-0.92 range across the three models.  So ~90% of what an intersectional
pair does is already explained by its two singles.

Operationally, that flips into a screen:

    score(pair) = | actual_yes_lean(pair) - predicted_yes_lean(pair) |

The pairs with the largest scores are the pairs where the model does something
on the compound that the singles do NOT predict.  Everything else can be
approximated from just the 112 singles.

Why the readout direction, not raw activations
----------------------------------------------
`w_eff` is the direction whose inner product with the residual sets the yes/no
logit gap (up to a positive RMS scale).  A residual orthogonal to w_eff cannot
change the model's answer, no matter how large; only the projection along w_eff
matters for behaviour.  So the operational quantity per pair is a single scalar:

    yes_lean(x) = w_eff . (x - base)

Fitted on that scalar, the additive model has 2 free parameters per (model,
layer), the fit is trivially cheap, and the residual is directly in units the
downstream answer cares about.

Why LOTO by template
--------------------
An auditor would fit the additive law on some templates and screen on new ones;
the paper's operational claim needs the same procedure.  Leaving one template
out at fit time yields residuals that never saw their own template's covariance.

Since (a, b) are two scalars fit over ~460k pair-observations, the difference
from in-sample fitting is small in practice -- but the LOTO version is the one
the auditor could actually run.

What this saves an auditor
--------------------------
The screen is computed from singles + base + one pair-forward-pass per pair at
one layer.  Compared to full behavioural evaluation (many samples per pair, or
per-generation annotation), this is a factor of ~20-50x cheaper.  A practitioner
who trusts the screen can restrict expensive evaluation to the top ~10% of
pairs by score and cover the ~90% share of behavioural deviation with it (this
recovery number is measured, not assumed -- see the behavioural check below).

Reliability and validity checks the screen must pass
----------------------------------------------------
  1. Split-half reliability: correlate scores from odd- vs even-numbered
     templates, Spearman-Brown corrected.  A score with r < 0.5 is measurement
     noise dressed as a ranking; anything the paper claims about "the top-K"
     pairs must survive this.

  2. Cross-model rank agreement (Kendall tau, Spearman rho): if the top pairs
     are genuinely a property of intersectional composition, they should
     overlap across models to a degree that random rankings would not.  If they
     do not overlap, the screen is measuring model-specific quirks and the
     paper cannot claim "these are intersectionally-interesting pairs."

  3. Permutation null: shuffle the (A, B) labels within each template, refit
     the additive law, and see whether the observed |residual| distribution
     dominates the permutation-null distribution.  This says the score is not
     just what you get from ANY two singles being added.

  4. Behavioural recovery: given the already-computed per-pair
     abs_behavioral_residual (from behavioral_additivity_polarityfixed.csv),
     does the representational score rank the same pairs high?  If so, an
     auditor can use the cheaper representational screen as a proxy for the
     expensive behavioural one.  If not, the screen predicts geometry only, not
     behaviour, and the operational pitch is weaker.
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
DATA_DIR = ROOT.parent / "data"
sys.path.insert(0, str(ROOT))


def peak_layer(model):
    d = pd.read_csv(OUT_DIR / f"{model}_identity_probe_full.csv")
    d["gap"] = d.ceiling_auc_real - d.pair_auc_real
    return int(d.groupby("layer").gap.mean().idxmax())


def readout_direction(model):
    """Unit vector whose activation-space inner product is the yes/no logit
    gap (up to a positive RMS scale).  See subspace_additivity.readout_direction
    for the derivation."""
    p = DATA_DIR / f"lm_head_yesno_{model}.npz"
    legacy = DATA_DIR / "lm_head_yesno.npz"
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


def shard_pats(model):
    return sorted(int(re.match(rf"{model}_pattern(\d+)_layer\d+\.npz$", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*_layer*.npz")
                  if re.match(rf"{model}_pattern(\d+)_layer\d+\.npz$", f.name))


def collect(model, L, w):
    """Stream shards, project everything onto w, return long-form scalars.

    Columns: pattern_id, ord (0=combo12, 1=combo21), s1, s2, y_pair, y_a, y_b
    (all displacements from base).
    """
    rows = []
    pats = sorted(set(shard_pats(model)))
    for pid in pats:
        p = ACT_DIR / f"{model}_pattern{pid}_layer{L}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as sh:
            base = sh["base_vec"].astype(np.float32).reshape(-1)
            sv = sh["singles_vec"].astype(np.float32)
            ss = np.array([str(x) for x in sh["singles_stigma"]])
            cv = sh["combo_vec"].astype(np.float32)
            c1 = np.array([str(x) for x in sh["combo_stigma1"]])
            c2 = np.array([str(x) for x in sh["combo_stigma2"]])
        y_base = float(base @ w)
        y_single = sv @ w - y_base                     # displacement, scalar
        idx = {n: k for k, n in enumerate(ss)}
        y_pair = cv @ w - y_base
        # order flag: an (A,B) row where A appears in combo_stigma2 first is ord=1
        # in this dataset ord is determined by row layout; we recover it from
        # whether the pair (c1, c2) has been seen before
        seen = {}
        for i, (a, b) in enumerate(zip(c1, c2)):
            o = seen.get((min(a, b), max(a, b)), 0)
            seen[(min(a, b), max(a, b))] = 1
            rows.append((pid, o, a, b, float(y_pair[i]),
                         float(y_single[idx[a]]), float(y_single[idx[b]])))
    return pd.DataFrame(rows, columns=["pattern_id", "ord", "s1", "s2",
                                       "y_pair", "y_a", "y_b"])


def fit_ab(df, held_out_pat=None):
    """OLS on y_pair = a * y_a + b * y_b, optionally excluding one template."""
    d = df if held_out_pat is None else df[df.pattern_id != held_out_pat]
    X = d[["y_a", "y_b"]].to_numpy(dtype=np.float64)
    y = d["y_pair"].to_numpy(dtype=np.float64)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


def score(model, out_prefix=""):
    L = peak_layer(model)
    w = readout_direction(model)
    if w is None:
        print(f"[{model}] no readout direction (lm_head not extracted) -- skipping"); return None
    print(f"[{model}] peak layer L={L}, readout ||w||={np.linalg.norm(w):.2f}")

    df = collect(model, L, w)
    if df.empty:
        print(f"[{model}] no pattern shards found at layer {L} -- skipping"); return None
    print(f"[{model}] {len(df):,} pair observations   "
          f"{df.pattern_id.nunique()} templates   "
          f"{df.groupby(['s1','s2']).ngroups} pair-orderings")

    a_all, b_all = fit_ab(df)
    df["pred_pool"] = a_all * df.y_a + b_all * df.y_b
    ss_res = ((df.y_pair - df.pred_pool) ** 2).sum()
    ss_tot = ((df.y_pair - df.y_pair.mean()) ** 2).sum()
    r2_pool = 1 - ss_res / ss_tot
    print(f"[{model}] pooled OLS  a={a_all:.3f}  b={b_all:.3f}  R^2={r2_pool:.3f}")

    # LOTO: fit on 36 templates, score on the 37th.  Residuals are what an
    # auditor's screen would produce -- they never saw the pair's own template.
    preds = np.full(len(df), np.nan)
    for pid, sub in df.groupby("pattern_id"):
        a, b = fit_ab(df, held_out_pat=pid)
        preds[sub.index] = a * sub.y_a.values + b * sub.y_b.values
    df["pred"] = preds
    df["resid"] = df.y_pair - df.pred

    # per-pair score, symmetric in ordering (avg over both), and per-model
    # SD of yes_lean so the number is comparable across models
    yl_sd = df.y_pair.std()
    pair = (df.groupby(["s1", "s2"])
              .agg(y_pair=("y_pair", "mean"), pred=("pred", "mean"),
                   resid=("resid", "mean"), abs_resid=("resid", lambda x: x.abs().mean()),
                   n_obs=("resid", "size")).reset_index())
    pair["z_abs_resid"] = pair.abs_resid / yl_sd     # in units of pair-yes-lean SD

    # Singles-extremity correction.
    # About 60% of |resid| in raw form is explained by how large the singles'
    # yes-lean already is: a linear fit will trivially leave larger absolute
    # residuals on pairs with larger predictions, and one identity ("Sex
    # Offender") tops the singles-extremity list AND dominates the raw top-100.
    # `abs_resid_adj` regresses out max and sum of |single yes-lean|, so what
    # remains is composition-specific rather than magnitude-driven.  It is the
    # ranking an auditor should sort by; `abs_resid` is kept for transparency.
    y_single_abs = (df.groupby("s1").y_a.apply(lambda x: x.abs().mean()))
    single_ext = y_single_abs.reindex(pair.s1).values, y_single_abs.reindex(pair.s2).values
    ya_ab, yb_ab = single_ext
    pair["ya_abs"], pair["yb_abs"] = ya_ab, yb_ab
    X = np.column_stack([np.maximum(ya_ab, yb_ab), ya_ab + yb_ab, np.ones(len(pair))])
    coef = np.linalg.lstsq(X, pair.abs_resid.values, rcond=None)[0]
    pair["abs_resid_adj"] = pair.abs_resid.values - X @ coef
    ext_r2 = 1 - ((pair.abs_resid_adj ** 2).sum()
                  / ((pair.abs_resid - pair.abs_resid.mean()) ** 2).sum())
    print(f"[{model}] singles-extremity explains R^2={ext_r2:.3f} of raw screen; "
          f"abs_resid_adj is the extremity-adjusted rank")

    # symmetrise: (A,B) and (B,A) are the same underlying pair
    pair["key"] = pair.apply(lambda r: tuple(sorted([r.s1, r.s2])), axis=1)
    sym = (pair.groupby("key")
             .agg(abs_resid=("abs_resid", "mean"),
                  abs_resid_adj=("abs_resid_adj", "mean"),
                  z_abs_resid=("z_abs_resid", "mean"),
                  signed_resid=("resid", "mean"),
                  n_obs=("n_obs", "sum")).reset_index())
    sym[["s1", "s2"]] = pd.DataFrame(sym["key"].tolist(), index=sym.index)
    sym = sym.drop(columns=["key"])[["s1", "s2", "abs_resid", "abs_resid_adj",
                                     "z_abs_resid", "signed_resid", "n_obs"]]

    # split-half reliability across templates -- odd vs even pattern_id
    pats = sorted(df.pattern_id.unique())
    odd, even = pats[::2], pats[1::2]
    def half_scores(pat_set):
        sub = df[df.pattern_id.isin(pat_set)]
        g = sub.groupby([sub.s1.where(sub.s1 < sub.s2, sub.s2),
                         sub.s1.where(sub.s1 >= sub.s2, sub.s2)])
        return g.resid.apply(lambda x: x.abs().mean())
    ho = half_scores(odd); he = half_scores(even)
    common = ho.index.intersection(he.index)
    r_half = float(np.corrcoef(ho.loc[common].values, he.loc[common].values)[0, 1])
    sb = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
    print(f"[{model}] split-half r={r_half:.3f}  Spearman-Brown={sb:.3f}  n={len(common):,}")

    # permutation null: shuffle A/B labels WITHIN each template, refit, score.
    # The additive contribution is preserved (still a linear function of two
    # singles); any pair-specific composition is broken.
    rng = np.random.default_rng(0)
    null_scores = []
    for _ in range(5):                               # 5 shuffles is enough here
        d2 = df.copy()
        for pid, sub in d2.groupby("pattern_id"):
            perm = rng.permutation(sub.index.values)
            d2.loc[sub.index, "y_a"] = d2.loc[perm, "y_a"].values
            d2.loc[sub.index, "y_b"] = d2.loc[perm, "y_b"].values
        a_n, b_n = fit_ab(d2)
        d2["res_n"] = d2.y_pair - a_n * d2.y_a - b_n * d2.y_b
        null_scores.append(d2.res_n.abs().mean())
    print(f"[{model}] |resid| observed {df.resid.abs().mean():.4f}   "
          f"permutation-null mean {np.mean(null_scores):.4f}   "
          f"ratio {df.resid.abs().mean() / np.mean(null_scores):.2f}x")

    # tag reliability and other diagnostics onto every row so callers can filter
    sym["r_half"] = r_half; sym["spearman_brown"] = sb
    sym["r2_pooled"] = r2_pool; sym["a_pooled"] = a_all; sym["b_pooled"] = b_all

    out = OUT_DIR / f"{out_prefix}{model}_deviation_screen.csv"
    sym.sort_values("abs_resid", ascending=False).to_csv(out, index=False)
    print(f"[{model}] wrote {out}   {len(sym):,} unique pairs")
    return sym


def compare_models(scores):
    """Rank agreement across models, and correlation with behavioural residual."""
    if len(scores) < 2:
        return
    from itertools import combinations
    print("\n" + "=" * 72)
    print("CROSS-MODEL RANK AGREEMENT (Spearman rho on shared pairs)")
    print("=" * 72)
    for (m1, s1), (m2, s2) in combinations(scores.items(), 2):
        j = s1.merge(s2, on=["s1", "s2"], suffixes=(f"_{m1}", f"_{m2}"))
        rho = j[f"abs_resid_{m1}"].corr(j[f"abs_resid_{m2}"], method="spearman")
        print(f"  {m1:8s} vs {m2:8s}  n={len(j):,}  rho={rho:+.3f}")

    print("\n" + "=" * 72)
    print("REPRESENTATIONAL SCREEN vs BEHAVIOURAL RESIDUAL")
    print("(on the logit scale -- probability-scale residuals are dominated by")
    print(" the [0,1] cap: high-bias singles predict >1.0, which is not a")
    print(" composition finding, it is arithmetic)")
    print("=" * 72)

    def logit(p):
        p = np.clip(p, 1 / 75, 74 / 75)         # 37 templates x 2 orderings
        return np.log(p / (1 - p))

    for m, s in scores.items():
        bp = OUT_DIR / f"{m}_behavioral_additivity_polarityfixed.csv"
        if not bp.exists():
            print(f"  {m}: no polarity-fixed behavioural residual file"); continue
        b = pd.read_csv(bp)
        dA = logit(b.bias_ind1) - logit(b.bias_base)
        dB = logit(b.bias_ind2) - logit(b.bias_base)
        dP = ((logit(b.bias_combo12) + logit(b.bias_combo21)) / 2) - logit(b.bias_base)
        X = np.column_stack([dA, dB]); y = dP.values
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        b["logit_resid"] = y - X @ coef
        b["abs_logit_resid"] = np.abs(b.logit_resid)
        # naive (predicted-in-probability) residual, kept only to show that the
        # apparent behavioural nonadditivity in it is ceiling, not composition
        b["abs_prob_resid"] = b.abs_behavioral_residual

        b["key"] = b.apply(lambda r: tuple(sorted([r.stigma1, r.stigma2])), axis=1)
        s["key"] = s.apply(lambda r: tuple(sorted([r.s1, r.s2])), axis=1)
        agg = b.groupby("key").agg(abs_logit_resid=("abs_logit_resid", "mean"),
                                   abs_prob_resid=("abs_prob_resid", "mean")).reset_index()
        j = s.merge(agg, on="key")

        rho_raw = j.abs_resid.corr(j.abs_logit_resid, method="spearman")
        rho_adj = j.abs_resid_adj.corr(j.abs_logit_resid, method="spearman")
        rho_prob = j.abs_resid.corr(j.abs_prob_resid, method="spearman")
        print(f"\n  {m:8s}  behavioural R^2 (logit fit) a={coef[0]:.3f} b={coef[1]:.3f}")
        print(f"           screen rho vs LOGIT residual  raw={rho_raw:+.3f}"
              f"   adjusted={rho_adj:+.3f}"
              f"   (probability residual={rho_prob:+.3f}: mostly cap)")
        n = len(j); top_beh = set(j.nlargest(n // 10, "abs_logit_resid").index)
        for col, tag in [("abs_resid", "raw     "), ("abs_resid_adj", "adjusted")]:
            for K in [5, 10, 20, 50]:
                top_rep = set(j.nlargest(n * K // 100, col).index)
                hit = 100 * len(top_rep & top_beh) / len(top_beh)
                print(f"             {tag} top-{K:>2}% catches {hit:5.1f}% "
                      f"of behavioural top-10%  (lift {hit/K:.1f}x)")

        # persist logit-scale numbers alongside the screen
        s.merge(agg, on="key").to_csv(
            OUT_DIR / f"{m}_deviation_screen.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    scores = {}
    for m in args.models:
        s = score(m, out_prefix=args.prefix)
        if s is not None:
            scores[m] = s
        print()
    compare_models(scores)


if __name__ == "__main__":
    main()
