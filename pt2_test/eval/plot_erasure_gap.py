"""
Plot the erasure gap by layer from identity_probe.py.

erasure gap = ceiling_auc - pair_auc
  ceiling_auc : can a diff-in-means direction detect identity i in SINGLE-identity
                residuals at this layer (what the direction can do at all)
  pair_auc    : can it still detect i in the residual for a PAIR containing i

Reading pair_auc alone is misleading: an early layer can show low pair recovery
purely because identity is weakly encoded there.  The gap is the quantity that
isolates loss due to composition.

Reads the per-model CSVs when they exist, otherwise falls back to parsing the
run log, so the figure can be drawn while the sweep is still going.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

# dataviz reference palette, categorical slots 1-2 (light mode).
# Documented as passing the adjacent-pairlist gates used by line charts:
# CVD dE 9.1, normal-vision dE 19.6.
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_SOFT  = "#52514e"
CEILING_C = "#2a78d6"   # slot 1, blue
PAIR_C    = "#eb6834"   # slot 2, orange
NULL_C    = "#b8b7b1"   # recessive gray: the control arms are a reference floor,
                        # not a series of interest, so they take no categorical slot


def from_log(path: Path) -> pd.DataFrame:
    pat = re.compile(r"\[(\w+)\] layer (\d+) in \d+s\s+ceiling ([\d.]+)\s+pair ([\d.]+)\s+"
                     r"\(shuf ([\d.]+), rand ([\d.]+)\)")
    rows = [{"model": m.group(1), "layer": int(m.group(2)),
             "ceiling_auc_real": float(m.group(3)), "pair_auc_real": float(m.group(4)),
             "pair_auc_shuf": float(m.group(5)), "pair_auc_rand": float(m.group(6))}
            for m in (pat.search(l) for l in path.read_text().splitlines()) if m]
    return pd.DataFrame(rows)


def from_csvs(tag: str) -> pd.DataFrame:
    out = []
    for p in sorted(OUT_DIR.glob(f"*_identity_probe{tag}.csv")):
        model = p.name.split("_identity_probe")[0]
        d = pd.read_csv(p)
        g = d.groupby("layer")[["ceiling_auc_real", "pair_auc_real",
                                 "pair_auc_shuf", "pair_auc_rand"]].mean().reset_index()
        # spread across identities, for the band on the gap panel
        d["gap"] = d.ceiling_auc_real - d.pair_auc_real
        q = d.groupby("layer").gap.quantile([0.25, 0.75]).unstack()
        g["gap_lo"], g["gap_hi"] = q[0.25].values, q[0.75].values
        g["model"] = model
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--log", default="/tmp/probe.log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = from_csvs(args.tag)
    source = "per-identity CSVs"
    if df.empty:
        df = from_log(Path(args.log))
        source = "run log (sweep still in progress)"
    if df.empty:
        raise SystemExit("no probe results found yet")
    df = df.sort_values(["model", "layer"])
    df["gap"] = df.ceiling_auc_real - df.pair_auc_real

    models = list(df.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d5d4ce")
        ax.tick_params(colors=INK_SOFT, labelsize=9)

    m = models[0]
    d = df[df.model == m]

    # --- left: absolute AUCs, gap shown as the area between the two -----------
    ax = axes[0]
    lo = np.minimum(d.pair_auc_shuf, d.pair_auc_rand)
    hi = np.maximum(d.pair_auc_shuf, d.pair_auc_rand)
    ax.fill_between(d.layer, lo, hi, color=NULL_C, alpha=0.55, linewidth=0,
                    label="control floor (shuffled / random)")
    ax.fill_between(d.layer, d.pair_auc_real, d.ceiling_auc_real,
                    color=CEILING_C, alpha=0.13, linewidth=0)
    ax.plot(d.layer, d.ceiling_auc_real, color=CEILING_C, linewidth=2,
            marker="o", markersize=4, label="ceiling: identity in singles")
    ax.plot(d.layer, d.pair_auc_real, color=PAIR_C, linewidth=2,
            marker="o", markersize=4, label="recovery: identity in pairs")
    ax.set_xlabel("Layer", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("AUC", color=INK_SOFT, fontsize=10)
    ax.set_title("Identity detectability by depth", color=INK, fontsize=11.5, pad=10)
    ax.set_ylim(0.4, 1.0)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_SOFT)

    # --- right: the gap itself (single series -> no legend) -------------------
    ax = axes[1]
    ax.axhline(0, color="#d5d4ce", linewidth=1.5)
    if "gap_lo" in d:
        ax.fill_between(d.layer, d.gap_lo, d.gap_hi, color=CEILING_C, alpha=0.15,
                        linewidth=0)
    ax.plot(d.layer, d.gap, color=CEILING_C, linewidth=2, marker="o", markersize=4)
    # Only call it a peak if it is interior; on a partial sweep the max sits at
    # the boundary and is just where we stopped looking.
    imax = d.gap.idxmax()
    peak = d.loc[imax]
    at_edge = peak.layer == d.layer.max()
    label = (f"still rising at L{int(peak.layer)} ({peak.gap:.3f})" if at_edge
             else f"peak {peak.gap:.3f} @ L{int(peak.layer)}")
    ax.annotate(label, xy=(peak.layer, peak.gap),
                xytext=(-10 if at_edge else 8, -16), textcoords="offset points",
                fontsize=9, color=INK_SOFT,
                ha="right" if at_edge else "left")
    ax.set_xlabel("Layer", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Erasure gap  (ceiling AUC − pair AUC)", color=INK_SOFT, fontsize=10)
    ax.set_title("Recoverability lost to composition", color=INK, fontsize=11.5, pad=10)

    fig.suptitle(f"{m}: is an identity still linearly recoverable once composed into a pair?"
                 f"\ndifference-in-means directions, leave-one-template-out; "
                 f"both control arms at chance",
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = args.out or (OUT_DIR / f"erasure_gap_by_layer{args.tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"source: {source}")
    print(f"layers plotted: {len(d)}  ({int(d.layer.min())}-{int(d.layer.max())})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
