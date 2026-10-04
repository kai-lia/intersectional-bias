"""
MAIHDA on identity recoverability under composition.

Question: of the variance in how detectable an identity remains inside a pair,
how much belongs to the PAIR as a stratum, and how much is just the additive
contribution of its two constituent identities?  MAIHDA answers this by
comparing a null model (stratum only) with an adjusted model (stratum plus
constituent main effects) and reading the proportional change in the stratum
variance.

  VPC  = var(pair) / total var, from the NULL model.
         share of variation attributable to which pair you are looking at,
         before accounting for constituents.
  PCV  = (var_pair_null - var_pair_adjusted) / var_pair_null.
         how much of the stratum variance the constituents explain.  What
         remains is the genuinely intersectional part.

Two specifications, because the outcome is a per-MEMBER quantity and the
choice is not cosmetic:

  pair    outcome averaged over the two members -> a property of the stratum.
          Constituents enter as MULTI-MEMBERSHIP: both identities draw from one
          set of 112 effects, entered as (u[i] + u[j]) / 2.  This matters
          because the pairs are unordered -- modelling id1 and id2 as two
          separate crossed factors would split each identity's information
          across slots and is simply wrong here.  Standard MAIHDA writeups do
          not cover this case, since in epidemiology the axes (sex, race) are
          distinct rather than interchangeable.

  member  one row per (pair, prompt, member): how well does the focal identity
          survive when paired with this partner.  Here focal and partner are
          NOT interchangeable, so multi-membership does not apply and they
          enter as two separate crossed factors.  This is the specification
          that speaks directly to the flattening asymmetry.

Outcome scale
-------------
d-prime, (score_AB - mean_absent) / sd_absent.  An earlier parameterisation
divided by (single - absent); that denominator is a difference of two means and
approached zero for 2 of 4144 identity-pattern cells, producing values from
-4488 to +3220 (sd 25.9, skew -53).  Normalising by the absent distribution's
SD instead -- estimated from ~6.1k pairs per cell -- gives sd 0.93 and skew
-0.09, so a Gaussian likelihood is appropriate and no Beta/logit is needed.

Reporting follows the Lizotte critique: predicted values per stratum combine
the fixed and random parts, never the random effects alone.
"""
import argparse
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def codes(s):
    c = pd.Categorical(s)
    return c.codes.astype("int32"), len(c.categories), list(c.categories)


def build(df, spec, adjusted):
    """Null vs adjusted, for either specification."""
    y = df.y.to_numpy("float64")
    pair_i, n_pair, _ = codes(df.pair)
    prom_i, n_prom, _ = codes(df.prompt)

    with pm.Model() as mdl:
        intercept = pm.Normal("intercept", 0.0, 1.0)
        sd_pair = pm.Exponential("sd_pair", 1.0)
        sd_prom = pm.Exponential("sd_prompt", 1.0)
        sigma = pm.Exponential("sigma", 1.0)

        # non-centred: crossed effects with many levels sample badly otherwise
        u_pair = pm.Normal("u_pair", 0.0, 1.0, shape=n_pair) * sd_pair
        u_prom = pm.Normal("u_prompt", 0.0, 1.0, shape=n_prom) * sd_prom
        mu = intercept + u_pair[pair_i] + u_prom[prom_i]

        if adjusted and spec == "pair":
            i1, n_id, _ = codes(df.id1)
            i2, _, _ = codes(df.id2)
            sd_id = pm.Exponential("sd_identity", 1.0)
            u_id = pm.Normal("u_identity", 0.0, 1.0, shape=n_id) * sd_id
            # multi-membership: mean of the two members' effects, matching
            # brms mm() default equal weights
            mu = mu + 0.5 * (u_id[i1] + u_id[i2])
        elif adjusted and spec == "member":
            fi, n_id, _ = codes(df.focal)
            pi, _, _ = codes(df.partner)
            sd_f = pm.Exponential("sd_focal", 1.0)
            sd_p = pm.Exponential("sd_partner", 1.0)
            u_f = pm.Normal("u_focal", 0.0, 1.0, shape=n_id) * sd_f
            u_p = pm.Normal("u_partner", 0.0, 1.0, shape=n_id) * sd_p
            mu = mu + u_f[fi] + u_p[pi]

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return mdl


def variance_components(idata, adjusted, spec):
    post = idata.posterior
    parts = {"pair": post["sd_pair"].values ** 2,
             "prompt": post["sd_prompt"].values ** 2,
             "residual": post["sigma"].values ** 2}
    if adjusted and spec == "pair":
        parts["identity"] = post["sd_identity"].values ** 2
    elif adjusted and spec == "member":
        parts["focal"] = post["sd_focal"].values ** 2
        parts["partner"] = post["sd_partner"].values ** 2
    total = sum(parts.values())
    return parts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="granite")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--spec", choices=["pair", "member"], default="pair")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--subsample-pairs", type=int, default=None,
                    help="fit on a random subset of pairs; use for a fast first pass")
    args = ap.parse_args()

    if args.layer is None:
        pr = pd.read_csv(OUT_DIR / f"{args.model}_identity_probe_full.csv")
        pr["gap"] = pr.ceiling_auc_real - pr.pair_auc_real
        args.layer = int(pr.groupby("layer").gap.mean().idxmax())
    src = OUT_DIR / f"{args.model}_maihda_input_L{args.layer}.csv"
    d = pd.read_csv(src)
    d["id1"], d["id2"] = d.id1_idx.astype(str), d.id2_idx.astype(str)

    if args.spec == "pair":
        d = (d.groupby(["pair", "prompt", "id1", "id2"], as_index=False)
               .d_prime.mean().rename(columns={"d_prime": "y"}))
    else:
        d = d.rename(columns={"d_prime": "y"})

    if args.subsample_pairs:
        keep = pd.Series(d.pair.unique()).sample(args.subsample_pairs, random_state=0)
        d = d[d.pair.isin(set(keep))]

    # Standardise the outcome.  d-prime has mean ~1.31, so an intercept prior of
    # Normal(0,1) is in conflict with the data and drags the sampler; Exponential(1)
    # on the SDs is likewise calibrated to a unit-scale outcome.  Variance SHARES
    # (VPC, PCV) are scale-invariant, so this changes nothing we report.
    y_mu, y_sd = d.y.mean(), d.y.std()
    d["y"] = (d.y - y_mu) / y_sd
    log.info(f"[{args.model} L{args.layer}] spec={args.spec}  rows={len(d):,}  "
             f"pairs={d.pair.nunique():,}  prompts={d.prompt.nunique()}  "
             f"(y standardised from mean {y_mu:.3f}, sd {y_sd:.3f})")

    res = {}
    for adjusted in (False, True):
        tag = "adjusted" if adjusted else "null"
        log.info(f"  fitting {tag} model ...")
        with build(d, args.spec, adjusted):
            idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                              target_accept=0.95, progressbar=False,
                              idata_kwargs={"log_likelihood": False})
        div = int(idata.sample_stats.diverging.values.sum())
        rhat = float(az.rhat(idata).to_array().max())
        log.info(f"  {tag}: divergences={div}  max R-hat={rhat:.3f}")
        parts, total = variance_components(idata, adjusted, args.spec)
        res[tag] = {"parts": parts, "total": total, "div": div, "rhat": rhat}
        idata.to_netcdf(str(OUT_DIR / f"{args.model}_maihda_{args.spec}_{tag}_L{args.layer}.nc"))

    def q(x):
        return np.percentile(x, [2.5, 50, 97.5])

    print("\n" + "=" * 86)
    print(f"MAIHDA  {args.model}  layer {args.layer}  spec={args.spec}")
    print("=" * 86)
    for tag in ("null", "adjusted"):
        r = res[tag]
        print(f"\n  {tag.upper()}  (divergences {r['div']}, max R-hat {r['rhat']:.3f})")
        for k, v in r["parts"].items():
            share = 100 * v / r["total"]
            lo, md, hi = q(share)
            print(f"    var {k:<10} {md:>6.1f}% of total   [{lo:.1f}, {hi:.1f}]")

    vpc = 100 * res["null"]["parts"]["pair"] / res["null"]["total"]
    lo, md, hi = q(vpc)
    print(f"\n  VPC (null): pair-level share of variance = {md:.1f}%  [{lo:.1f}, {hi:.1f}]")

    pcv = 100 * (res["null"]["parts"]["pair"] - res["adjusted"]["parts"]["pair"]) \
          / res["null"]["parts"]["pair"]
    lo, md, hi = q(pcv)
    print(f"  PCV: constituents explain {md:.1f}% of the pair variance  [{lo:.1f}, {hi:.1f}]")
    print(f"  -> residual intersectional share = {100-md:.1f}% of the original pair variance")

    pd.DataFrame([{"model": args.model, "layer": args.layer, "spec": args.spec,
                   "vpc_median": np.median(vpc), "pcv_median": np.median(pcv),
                   "div_null": res["null"]["div"], "div_adj": res["adjusted"]["div"],
                   "rhat_null": res["null"]["rhat"], "rhat_adj": res["adjusted"]["rhat"]}]
                 ).to_csv(OUT_DIR / f"{args.model}_maihda_summary_{args.spec}_L{args.layer}.csv",
                          index=False)
    print(f"\nsaved -> {OUT_DIR}/{args.model}_maihda_summary_{args.spec}_L{args.layer}.csv")


if __name__ == "__main__":
    main()
