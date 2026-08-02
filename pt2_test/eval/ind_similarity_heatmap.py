"""
Heatmap of pairwise cosine similarity between fixed identities' solo
("who is <trait>") residual-stream vectors, averaged across all layers and
patterns -- visualizes the clustering structure behind the
similarity/non-additive-fraction correlation found in additivity_random.py's
follow-up analysis (representationally similar trait pairs compose less
additively than dissimilar ones).

Identity set is read from the data itself (every stigma appearing in
scenario_ids), not hardcoded -- works unchanged whether the underlying
activations came from random_sample_activations.py's --identities fixed15
(15 traits) or --identities full (112 traits) run.

Rows/columns are hierarchically clustered (not alphabetical) so the block
structure -- which traits the model treats as "near" each other -- is
visible at a glance, rather than requiring the reader to hunt for it. Tick
labels are shown only when there are few enough identities to read them
(mirrors additivity_random_scenarios.py's convention of dropping per-row
labels once there are too many to render); at 112x112 the block structure
from clustering is still visible without labels.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from activation_io import discover_layers, load_scenarios

ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)

_MAX_LABELED_IDENTITIES = 20  # above this, tick labels would just overlap into noise


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def build_matrix(model_name: str, layers: list[int]) -> pd.DataFrame:
    rows = []
    identities: set[str] = set()
    for layer in layers:
        data = load_scenarios(ACT_DIR, model_name, layer)
        sids = data["scenario_ids"]
        sim = cos(data["ind1"], data["ind2"])
        for i, (s1, s2, pid) in enumerate(sids):
            rows.append({"stigma1": s1, "stigma2": s2, "sim": sim[i]})
            identities.add(s1); identities.add(s2)
    pair_sim = pd.DataFrame(rows).groupby(["stigma1", "stigma2"])["sim"].mean().reset_index()

    ordered_identities = sorted(identities)
    mat = pd.DataFrame(1.0, index=ordered_identities, columns=ordered_identities)
    for _, r in pair_sim.iterrows():
        mat.loc[r.stigma1, r.stigma2] = r.sim
        mat.loc[r.stigma2, r.stigma1] = r.sim
    return mat


def cluster_order(mat: pd.DataFrame) -> list[str]:
    dist = 1 - mat.to_numpy()
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    return [mat.index[i] for i in order]


def plot_heatmap(model_name: str, mat: pd.DataFrame, tag: str = "") -> None:
    order = cluster_order(mat)
    ordered = mat.loc[order, order]
    n = len(order)
    labeled = n <= _MAX_LABELED_IDENTITIES

    side = max(9, n * 0.35) if labeled else max(8, min(n * 0.09, 22))
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    vmin, vmax = np.percentile(ordered.to_numpy()[~np.eye(n, dtype=bool)], [2, 98])
    im = ax.imshow(ordered.to_numpy(), cmap=SEQ_CMAP, vmin=vmin, vmax=vmax)
    if labeled:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(order, fontsize=8)
    else:
        ax.set_xticks([]); ax.set_yticks([])  # too many identities for readable labels
    ax.set_title(f"{model_name}: solo-identity cosine similarity (hierarchically clustered)\n"
                 f"{n} identities, averaged across all layers & patterns -- narrow range, see color bar", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_ind_similarity_heatmap{tag}.png", dpi=150)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite")
    parser.add_argument("--tag", default="", help="output filename suffix, e.g. _full112 -- "
                         "keeps a full-scale rerun from overwriting existing fixed-15 results")
    args = parser.parse_args()

    layers = discover_layers(ACT_DIR, args.model)
    mat = build_matrix(args.model, layers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mat.to_csv(OUT_DIR / f"{args.model}_ind_similarity_matrix{args.tag}.csv")
    plot_heatmap(args.model, mat, args.tag)
    print(f"saved -> {OUT_DIR}/{args.model}_ind_similarity_heatmap{args.tag}.png")
    print(f"saved -> {OUT_DIR}/{args.model}_ind_similarity_matrix{args.tag}.csv")


if __name__ == "__main__":
    main()