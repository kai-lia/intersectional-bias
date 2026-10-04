"""
Is the erasure gap a real composition effect, or an artifact of direction quality?

The gap is  ceiling_auc - pair_auc  from identity_probe.py.  The worry: identities
that displace the residual stream less have noisier difference-in-means
directions, and a noisier direction might degrade faster when a second identity
is added -- producing a gap with no compositional meaning.

The two correlations separate these:

  r(gap, ceiling)              if large and negative, gaps ARE tracking direction
                               quality and the effect is suspect at that layer
  partial r(gap, displacement | ceiling)
                               the displacement relationship with direction
                               quality held fixed.  If this SURVIVES or
                               STRENGTHENS relative to the raw correlation, the
                               effect is not mediated by direction quality.

Reported per layer because the two mechanisms dominate at different depths: in
granite the ceiling relationship is strong early (L7, r = -0.62) and gone by the
gap peak (L24, r = -0.11).

Caveat that belongs in the writeup: displacement and the probe are computed from
the same activations, so they are not independent instruments.  This rules out
mediation through direction quality, not shared origin.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"


def shift_norms(model, n_layers=6):
    """Per-identity ||ind - base||, normalised within layer."""
    pats = sorted(int(re.match(rf"{model}_pattern(\d+)\.done", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*.done"))
    al = sorted(int(re.match(rf"{model}_pattern{pats[0]}_layer(\d+)\.npz", f.name).group(1))
                for f in ACT_DIR.glob(f"{model}_pattern{pats[0]}_layer*.npz"))
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


def partial_corr(y, x, z):
    """corr(y, x) with z partialled out of both."""
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    return float(np.corrcoef(ry, rx)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_full")
    args = ap.parse_args()

    rows = []
    for model in args.models:
        p = OUT_DIR / f"{model}_identity_probe{args.tag}.csv"
        if not p.exists():
            print(f"[{model}] no probe results yet -- skipping")
            continue
        d = pd.read_csv(p)
        d["gap"] = d.ceiling_auc_real - d.pair_auc_real
        sn = shift_norms(model)

        by_layer = d.groupby("layer").gap.mean()
        peak = int(by_layer.idxmax())
        print("=" * 92)
        print(f"{model.upper()}   gap peaks at L{peak} ({by_layer.max():.3f}); "
              f"{d.layer.nunique()} layers, {d.identity.nunique()} identities")
        print("=" * 92)
        print(f"  {'layer':<8}{'mean gap':>10}{'r(gap,ceiling)':>17}"
              f"{'r(gap,disp)':>14}{'partial r(gap,disp|ceiling)':>30}")
        layers = sorted(d.layer.unique())
        show = [layers[i] for i in np.linspace(0, len(layers) - 1, 6).astype(int)]
        if peak not in show:
            show = sorted(set(show + [peak]))
        for L in show:
            s = d[d.layer == L].copy()
            s["disp"] = s.identity.map(sn)
            s = s.dropna(subset=["disp", "gap", "ceiling_auc_real"])
            r_c = s.gap.corr(s.ceiling_auc_real)
            r_d = s.gap.corr(s["disp"])
            pr = partial_corr(s.gap.values, s["disp"].values, s.ceiling_auc_real.values)
            mark = "  <- gap peak" if L == peak else ""
            print(f"  L{L:<7}{s.gap.mean():>10.3f}{r_c:>17.3f}{r_d:>14.3f}{pr:>30.3f}{mark}")
            rows.append({"model": model, "layer": L, "mean_gap": s.gap.mean(),
                         "r_gap_ceiling": r_c, "r_gap_disp": r_d,
                         "partial_r_gap_disp_given_ceiling": pr, "is_peak": L == peak})

        s = d[d.layer == peak].copy()
        s["disp"] = s.identity.map(sn)
        print(f"\n  at the gap peak, most erased : "
              f"{', '.join(s.nlargest(5,'gap').identity.tolist())}")
        print(f"  at the gap peak, least erased: "
              f"{', '.join(s.nsmallest(5,'gap').identity.tolist())}\n")

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(OUT_DIR / f"erasure_confound_check{args.tag}.csv", index=False)
        print("=" * 92)
        print("CROSS-MODEL: does the effect replicate at each model's gap peak?")
        print("=" * 92)
        pk = out[out.is_peak]
        print(f"  {'model':<10}{'peak layer':>12}{'mean gap':>10}{'r(gap,ceiling)':>17}"
              f"{'partial r':>12}   verdict")
        for _, r in pk.iterrows():
            ok = abs(r.r_gap_ceiling) < 0.3 and r.partial_r_gap_disp_given_ceiling < -0.3
            print(f"  {r.model:<10}{int(r.layer):>12}{r.mean_gap:>10.3f}"
                  f"{r.r_gap_ceiling:>17.3f}{r.partial_r_gap_disp_given_ceiling:>12.3f}"
                  f"   {'not a direction-quality artifact' if ok else 'INSPECT'}")
        print(f"\nsaved -> {OUT_DIR}/erasure_confound_check{args.tag}.csv")


if __name__ == "__main__":
    main()
