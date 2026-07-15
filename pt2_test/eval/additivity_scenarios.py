"""
Per-scenario (not aggregated) view of the non-additive/emergent fraction from
additivity.py -- shows which individual scenarios drive the aggregate trend,
rather than only the layer-mean.

Outputs:
  {model}_additivity_scenarios.csv               -- one row per scenario, mean
                                                     fraction across all layers
                                                     (BL and LB) plus the actual
                                                     prompt text, ranked from
                                                     most-additive to most-emergent
  {model}_additivity_scenario_heatmap_BL.png
  {model}_additivity_scenario_heatmap_LB.png     -- scenario x layer heatmaps,
                                                     rows sorted most-additive
                                                     (top) to most-emergent (bottom)
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT      = Path(__file__).resolve().parent
ACT_DIR   = ROOT.parent / "data" / "activations"
CLEAN_CSV = ROOT.parent / "data" / "stigma_reasoning_clean.csv"
OUT_DIR   = ROOT.parent / "data" / "eval"

# validated sequential blue ramp (light->dark), see dataviz skill references/palette.md
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CKA_CMAP = LinearSegmentedColormap.from_list("cka_blue", _BLUE_RAMP)


def discover_layers(model_name: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(model_name)}_layer(\d+)\.npz$")
    layers = [int(m.group(1)) for f in ACT_DIR.glob(f"{model_name}_layer*.npz")
              if (m := pattern.match(f.name))]
    return sorted(layers)


def _condition(row) -> str:
    s1, s2 = row["stigma1"], row["stigma2"]
    s2 = None if pd.isna(s2) else s2
    if s1 == "Black" and s2 is None: return "B"
    if s1 == "Lesbian" and s2 is None: return "L"
    if s1 == "Black" and s2 == "Lesbian": return "BL"
    if s1 == "Lesbian" and s2 == "Black": return "LB"
    return ""


def compute_fractions(model_name: str, layers: list[int]) -> pd.DataFrame:
    """Long-form: one row per (pattern_id, prompt_style, layer), no aggregation."""
    rows = []
    for layer in layers:
        data = np.load(ACT_DIR / f"{model_name}_layer{layer}.npz", allow_pickle=True)
        B, L, BL, LB = data["B"], data["L"], data["BL"], data["LB"]
        scenario_ids = data["scenario_ids"]

        by_pattern: dict = {}
        for i, (pid, style) in enumerate(scenario_ids):
            by_pattern.setdefault(pid, {})[style] = i

        for pid, styles in by_pattern.items():
            if "base" not in styles:
                continue
            base_vec = B[styles["base"]]  # B/L/BL/LB are bit-identical at the base row
            for style, i in styles.items():
                if style == "base":
                    continue
                predicted = B[i] + L[i] - base_vec
                shift_bl = BL[i] - base_vec
                shift_lb = LB[i] - base_vec
                rows.append({
                    "pattern_id": pid, "prompt_style": style, "layer": layer,
                    "non_additive_frac_BL": np.linalg.norm(BL[i] - predicted) / np.linalg.norm(shift_bl),
                    "non_additive_frac_LB": np.linalg.norm(LB[i] - predicted) / np.linalg.norm(shift_lb),
                })
    return pd.DataFrame(rows)


def build_pivot(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = long_df.copy()
    df["scenario"] = df["pattern_id"].astype(str) + "_" + df["prompt_style"]
    pivot = df.pivot(index="scenario", columns="layer", values=value_col)
    # sort rows by each scenario's own mean fraction: most additive (top) -> most emergent (bottom)
    order = pivot.mean(axis=1).sort_values().index
    return pivot.reindex(order)


def plot_scenario_heatmap(model_name: str, pivot: pd.DataFrame, condition: str) -> None:
    values = pivot.to_numpy()
    vmin, vmax = np.percentile(values, 2), np.percentile(values, 98)

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.3), max(6, pivot.shape[0] * 0.08)))
    im = ax.imshow(values, aspect="auto", cmap=CKA_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=6, rotation=90)
    ax.set_yticks([])  # ~100+ rows -- too many for readable per-row labels
    ax.set_xlabel("Layer")
    ax.set_ylabel("Scenario (sorted: most additive top -> most emergent bottom)")
    ax.set_title(f"{model_name}: per-scenario non-additive fraction ({condition})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("non-additive fraction")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_additivity_scenario_heatmap_{condition}.png", dpi=150)
    plt.close(fig)


def build_ranked_table(model_name: str, long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby(["pattern_id", "prompt_style"])[["non_additive_frac_BL", "non_additive_frac_LB"]]
        .mean()
        .reset_index()
    )

    if CLEAN_CSV.exists():
        clean = pd.read_csv(CLEAN_CSV)
        # extract_activations.py only ever extracted stigma_col == "With Stigma"
        # (see its STIGMA_COL constant) -- the clean CSV has all four phrasing
        # variants, so this filter is required or the merge below fans out into
        # duplicate rows (one per variant) for every scenario.
        clean = clean[(clean["model"] == model_name) & (clean["stigma_col"] == "With Stigma")].copy()
        clean["condition"] = clean.apply(_condition, axis=1)
        text = clean[clean["condition"] == "BL"][["pattern_id", "prompt_style", "stigma_phrase", "prompt"]]
        summary = summary.merge(text, on=["pattern_id", "prompt_style"], how="left")

    return summary.sort_values("non_additive_frac_BL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite")
    args = parser.parse_args()

    layers = discover_layers(args.model)
    if not layers:
        raise FileNotFoundError(f"No activation files for '{args.model}' in {ACT_DIR}")

    long_df = compute_fractions(args.model, layers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for condition in ["BL", "LB"]:
        pivot = build_pivot(long_df, f"non_additive_frac_{condition}")
        plot_scenario_heatmap(args.model, pivot, condition)
        print(f"saved -> {OUT_DIR}/{args.model}_additivity_scenario_heatmap_{condition}.png")

    ranked = build_ranked_table(args.model, long_df)
    out_csv = OUT_DIR / f"{args.model}_additivity_scenarios.csv"
    ranked.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")

    cols = ["pattern_id", "prompt_style", "non_additive_frac_BL", "non_additive_frac_LB"]
    print("\nmost additive scenarios (lowest non-additive fraction):")
    print(ranked.head(5)[cols].to_string(index=False))
    print("\nmost emergent scenarios (highest non-additive fraction):")
    print(ranked.tail(5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()