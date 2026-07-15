"""
CKA sweep -- compares intersectional (BL/LB) vs. single-axis (B/L) activations
across layers, using pt2_test/extract_activations.py's per-model, per-layer
.npz files (data/activations/{model}_layer{N}.npz: B/L/BL/LB arrays row-matched
by scenario_id).

Equidistance test: does BL ("who is Black and is Lesbian") sit equidistant
from B and L, or collapse toward one axis? LB (phrasing order swapped) is
tested the same way as a replication check.
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from metrics import linear_cka_gram, permutation_null, bootstrap_diff, bh_fdr

ACT_DIR = ROOT.parent / "data" / "activations"
OUT_DIR = ROOT.parent / "data" / "eval"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONDITIONS = ["B", "L", "BL", "LB"]

# all six pairs are reported descriptively; only the four that pit the
# intersectional condition against a single-axis one get a permutation null
PAIRS = [("BL", "B"), ("BL", "L"), ("LB", "B"), ("LB", "L"), ("BL", "LB"), ("B", "L")]
NULL_PAIRS = [("BL", "B"), ("BL", "L"), ("LB", "B"), ("LB", "L")]

# (intersectional, axis_a, axis_b) -> diff = CKA(inter,axis_a) - CKA(inter,axis_b)
DIFF_TRIPLES = [("BL", "B", "L"), ("LB", "B", "L")]


def discover_layers(model_name: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(model_name)}_layer(\d+)\.npz$")
    layers = [int(m.group(1)) for f in ACT_DIR.glob(f"{model_name}_layer*.npz")
              if (m := pattern.match(f.name))]
    return sorted(layers)


def run_model(model_name: str, n_perm: int, n_boot: int) -> None:
    layers = discover_layers(model_name)
    if not layers:
        raise FileNotFoundError(
            f"No activation files for '{model_name}' in {ACT_DIR} "
            f"-- run pt2_test/extract_activations.py first."
        )
    log.info(f"[{model_name}] found {len(layers)} layers")

    cka_rows, diff_rows = [], []

    for layer in layers:
        data = np.load(ACT_DIR / f"{model_name}_layer{layer}.npz")
        vecs = {c: data[c] for c in CONDITIONS}
        n_scenarios = vecs["B"].shape[0]

        scores = {pair: linear_cka_gram(vecs[pair[0]], vecs[pair[1]]) for pair in PAIRS}

        for pair, score in scores.items():
            row = {
                "model": model_name, "layer": layer,
                "pair": f"{pair[0]}-{pair[1]}", "cka": score,
                "n_scenarios": n_scenarios,
            }
            if pair in NULL_PAIRS:
                null = permutation_null(vecs[pair[0]], vecs[pair[1]], n_perm=n_perm)
                row["z_score"] = (score - null.mean()) / null.std()
                row["p_value"] = float(np.mean(null >= score))
            cka_rows.append(row)

        for inter, a, b in DIFF_TRIPLES:
            diffs = bootstrap_diff(vecs[inter], vecs[a], vecs[b], n_boot=n_boot)
            diff_rows.append({
                "model": model_name, "layer": layer,
                "comparison": f"{inter}_vs_{a}_minus_{inter}_vs_{b}",
                "diff": scores[(inter, a)] - scores[(inter, b)],
                "ci_low":  float(np.percentile(diffs, 2.5)),
                "ci_high": float(np.percentile(diffs, 97.5)),
            })

    cka_df = pd.DataFrame(cka_rows)
    for pair in NULL_PAIRS:
        mask = cka_df["pair"] == f"{pair[0]}-{pair[1]}"
        cka_df.loc[mask, "p_value_fdr"] = bh_fdr(cka_df.loc[mask, "p_value"].to_numpy())
    diff_df = pd.DataFrame(diff_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cka_df.to_csv(OUT_DIR / f"{model_name}_cka.csv", index=False)
    diff_df.to_csv(OUT_DIR / f"{model_name}_diff.csv", index=False)
    log.info(f"[{model_name}] saved -> {OUT_DIR}/{model_name}_cka.csv, {model_name}_diff.csv")

    plot_trajectories(model_name, cka_df)
    plot_diff(model_name, diff_df)


def plot_trajectories(model_name: str, cka_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair, g in cka_df.groupby("pair"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["cka"], marker="o", label=pair)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title(f"{model_name}: pairwise CKA across layers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_cka_trajectories.png", dpi=150)
    plt.close(fig)


def plot_diff(model_name: str, diff_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for comparison, g in diff_df.groupby("comparison"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["diff"], marker="o", label=comparison)
        ax.fill_between(g["layer"], g["ci_low"], g["ci_high"], alpha=0.2)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA diff (positive -> closer to first axis)")
    ax.set_title(f"{model_name}: intersectional equidistance test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_diff_ci.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    for model_name in args.models:
        try:
            run_model(model_name, args.n_perm, args.n_boot)
        except FileNotFoundError as exc:
            log.error(str(exc))


if __name__ == "__main__":
    main()
