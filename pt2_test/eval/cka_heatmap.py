"""
Standalone heatmap of CKA magnitude (layer x pair) for the four
intersectional-vs-single-axis pairs -- BL-B, BL-L, LB-B, LB-L.

Reads an existing {model}_cka.csv produced by cka_sweep.py; does not recompute
anything, so it's cheap to re-run/tweak independent of the (slow) permutation
sweep.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data" / "eval"

PAIRS = ["BL-B", "BL-L", "LB-B", "LB-L"]

# validated sequential blue ramp (light->dark), see dataviz skill references/palette.md
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CKA_CMAP = LinearSegmentedColormap.from_list("cka_blue", _BLUE_RAMP)


def plot_heatmap(cka_csv: Path, out_png: Path, model_name: str) -> None:
    cka_df = pd.read_csv(cka_csv)
    sub = cka_df[cka_df["pair"].isin(PAIRS)]
    pivot = sub.pivot(index="pair", columns="layer", values="cka").reindex(PAIRS)
    layers = pivot.columns.to_numpy()

    values = pivot.to_numpy()
    # a handful of early layers (tokenization/embedding-dominated) sit far below
    # the rest -- clip the floor to a robust percentile so those layers saturate
    # to the floor color instead of stretching the whole scale and flattening
    # the real layer-to-layer texture in the remaining (majority) band.
    vmin, vmax = np.percentile(values, 10), values.max()
    clipped = (values < vmin).any()

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.35), 3.2))
    im = ax.imshow(values, aspect="auto", cmap=CKA_CMAP, vmin=vmin, vmax=vmax)

    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels(PAIRS)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7, rotation=90)
    ax.set_xlabel("Layer")
    title = f"{model_name}: CKA magnitude, intersectional vs. single-axis pairs"
    if clipped:
        title += "\n(lightest cells are floor-clipped -- early layers sit well below this scale's minimum)"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Linear CKA")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite")
    args = parser.parse_args()

    cka_csv = DATA_DIR / f"{args.model}_cka.csv"
    if not cka_csv.exists():
        raise FileNotFoundError(f"{cka_csv} not found -- run cka_sweep.py first.")

    out_png = DATA_DIR / f"{args.model}_cka_heatmap.png"
    plot_heatmap(cka_csv, out_png, args.model)


if __name__ == "__main__":
    main()