"""
Per-pair (not aggregated) view of the random-100-stigma-pair additivity /
lean results from additivity_random.py -- shows whether the pooled findings
(majority-emergent composition; combo21's late-network lean toward ind1) are
broad properties across most pairs, or driven by a handful of outliers.

Note on "lean": additivity_random.py's lean metric uses CKA, which needs a
sample of examples to build its Gram matrix (only 3 patterns per pair here,
too few for a reliable per-pair CKA). For a per-scenario/per-pair breakdown,
this script uses cosine similarity instead -- well-defined for a single
vector pair, at the cost of being a cruder directional measure than CKA.

Outputs:
  {model}_additivity_random_pairs.csv          -- one row per stigma pair,
                                                   mean non-additive fraction
                                                   and mean cosine-lean across
                                                   all layers and patterns,
                                                   ranked, with stigma names
  {model}_additivity_random_pair_heatmap_*.png -- pair x layer heatmaps for
                                                   non-additive fraction and
                                                   lean, both combo orderings
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT    = Path(__file__).resolve().parent
ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"

# validated sequential blue ramp (light->dark), see dataviz skill references/palette.md
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)
DIVERGING_CMAP = "RdBu_r"  # lean can be positive or negative -> diverging


def discover_layers(model_name: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(model_name)}_layer(\d+)\.npz$")
    layers = [int(m.group(1)) for f in ACT_DIR.glob(f"{model_name}_layer*.npz")
              if (m := pattern.match(f.name))]
    return sorted(layers)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compute_per_scenario(model_name: str, layers: list[int]) -> pd.DataFrame:
    """Long-form: one row per (stigma1, stigma2, pattern_id, layer)."""
    rows = []
    for layer in layers:
        data = np.load(ACT_DIR / f"{model_name}_layer{layer}.npz", allow_pickle=True)
        ind1, ind2 = data["ind1"], data["ind2"]
        combo12, combo21, base = data["combo12"], data["combo21"], data["base"]
        scenario_ids = data["scenario_ids"]

        predicted = ind1 + ind2 - base
        shift12 = combo12 - base
        shift21 = combo21 - base
        frac12 = np.linalg.norm(combo12 - predicted, axis=1) / np.linalg.norm(shift12, axis=1)
        frac21 = np.linalg.norm(combo21 - predicted, axis=1) / np.linalg.norm(shift21, axis=1)

        lean12 = cosine_sim(combo12, ind1) - cosine_sim(combo12, ind2)
        lean21 = cosine_sim(combo21, ind1) - cosine_sim(combo21, ind2)

        for i, (s1, s2, pid) in enumerate(scenario_ids):
            rows.append({
                "stigma1": s1, "stigma2": s2, "pattern_id": pid, "layer": layer,
                "non_additive_frac_combo12": frac12[i], "non_additive_frac_combo21": frac21[i],
                "lean_combo12": lean12[i], "lean_combo21": lean21[i],
            })
    return pd.DataFrame(rows)


def build_pivot(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = long_df.copy()
    df["pair"] = df["stigma1"] + " + " + df["stigma2"]
    pivot = df.groupby(["pair", "layer"])[value_col].mean().reset_index().pivot(
        index="pair", columns="layer", values=value_col
    )
    order = pivot.mean(axis=1).sort_values().index
    return pivot.reindex(order)


def plot_pair_heatmap(model_name: str, pivot: pd.DataFrame, label: str, diverging: bool) -> None:
    values = pivot.to_numpy()
    if diverging:
        vmax = np.percentile(np.abs(values), 98)
        vmin, cmap = -vmax, DIVERGING_CMAP
    else:
        vmin, vmax = np.percentile(values, 2), np.percentile(values, 98)
        cmap = SEQ_CMAP

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.3), max(6, pivot.shape[0] * 0.08)))
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=6, rotation=90)
    ax.set_yticks([])  # ~100 rows -- too many for readable per-row labels
    ax.set_xlabel("Layer")
    ax.set_ylabel("Stigma pair (sorted by row mean, low -> high)")
    ax.set_title(f"{model_name}: {label}, 100 random stigma pairs")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(label)
    fig.tight_layout()
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(OUT_DIR / f"{model_name}_additivity_random_pair_heatmap_{safe_label}.png", dpi=150)
    plt.close(fig)


def build_ranked_table(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.groupby(["stigma1", "stigma2"])[
            ["non_additive_frac_combo12", "non_additive_frac_combo21", "lean_combo12", "lean_combo21"]
        ]
        .mean()
        .reset_index()
        .sort_values("non_additive_frac_combo12")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite")
    args = parser.parse_args()

    layers = discover_layers(args.model)
    if not layers:
        raise FileNotFoundError(f"No activation files for '{args.model}' in {ACT_DIR}")

    long_df = compute_per_scenario(args.model, layers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for value_col, label, diverging in [
        ("non_additive_frac_combo12", "non-additive fraction (combo12)", False),
        ("non_additive_frac_combo21", "non-additive fraction (combo21)", False),
        ("lean_combo12", "lean toward ind1 (combo12)", True),
        ("lean_combo21", "lean toward ind1 (combo21)", True),
    ]:
        pivot = build_pivot(long_df, value_col)
        plot_pair_heatmap(args.model, pivot, label, diverging)
        print(f"saved -> heatmap for {label}")

    ranked = build_ranked_table(long_df)
    out_csv = OUT_DIR / f"{args.model}_additivity_random_pairs.csv"
    ranked.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")

    cols = ["stigma1", "stigma2", "non_additive_frac_combo12", "non_additive_frac_combo21",
            "lean_combo12", "lean_combo21"]
    print("\nmost additive pairs (lowest non-additive fraction, combo12):")
    print(ranked.head(5)[cols].to_string(index=False))
    print("\nmost emergent pairs (highest non-additive fraction, combo12):")
    print(ranked.tail(5)[cols].to_string(index=False))

    print("\nstrongest lean toward ind1 (combo21, since that's where the pooled effect was):")
    print(ranked.sort_values("lean_combo21", ascending=False).head(5)[cols].to_string(index=False))
    print("\nstrongest lean toward ind2 (combo21):")
    print(ranked.sort_values("lean_combo21", ascending=True).head(5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()