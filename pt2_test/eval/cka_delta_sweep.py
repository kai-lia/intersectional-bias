"""
CKA sweep on delta (baseline-subtracted) activations, rather than raw ones.

cka_sweep.py compares raw B/L/BL/LB activations directly -- but those are
dominated by shared scenario content (the same landlord/employee/etc. context
in every condition), so its permutation-null test mostly detects "same
scenario" rather than anything about the identity phrase itself.

This sweep instead compares delta_B = B - Base, delta_L = L - Base, etc.
(Base = the same pattern's "base"/no-stigma scenario, bit-identical across
B/L/BL/LB -- see additivity.py) -- isolating just the effect of inserting
each identity phrase, before asking whether BL's identity-effect looks more
like B's or L's. Same statistical machinery as cka_sweep.py (permutation
null, BH-FDR, bootstrap-CI equidistance diff), applied to deltas instead of
raw activations.
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
PAIRS = [("BL", "B"), ("BL", "L"), ("LB", "B"), ("LB", "L"), ("BL", "LB"), ("B", "L")]
NULL_PAIRS = [("BL", "B"), ("BL", "L"), ("LB", "B"), ("LB", "L")]
DIFF_TRIPLES = [("BL", "B", "L"), ("LB", "B", "L")]


def discover_layers(model_name: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(model_name)}_layer(\d+)\.npz$")
    layers = [int(m.group(1)) for f in ACT_DIR.glob(f"{model_name}_layer*.npz")
              if (m := pattern.match(f.name))]
    return sorted(layers)


def load_deltas(model_name: str, layer: int):
    """Return {"B":delta_B, "L":delta_L, "BL":delta_BL, "LB":delta_LB}, one row
    per non-base scenario, each already baseline-subtracted. None if no
    pattern in this layer has both a base and a non-base scenario."""
    data = np.load(ACT_DIR / f"{model_name}_layer{layer}.npz", allow_pickle=True)
    raw = {c: data[c] for c in CONDITIONS}
    scenario_ids = data["scenario_ids"]

    by_pattern: dict = {}
    for i, (pid, style) in enumerate(scenario_ids):
        by_pattern.setdefault(pid, {})[style] = i

    base_idx, other_idx = [], []
    for pid, styles in by_pattern.items():
        if "base" not in styles:
            continue
        base_i = styles["base"]
        for style, i in styles.items():
            if style == "base":
                continue
            base_idx.append(base_i)
            other_idx.append(i)

    if not other_idx:
        return None

    base_idx = np.array(base_idx)
    other_idx = np.array(other_idx)
    base_vecs = raw["B"][base_idx]  # B/L/BL/LB are bit-identical at the base row
    return {c: raw[c][other_idx] - base_vecs for c in CONDITIONS}


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
        vecs = load_deltas(model_name, layer)
        if vecs is None:
            log.warning(f"[{model_name}] layer {layer}: no pattern has both a "
                        f"base and non-base scenario -- skipping")
            continue
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
    cka_df.to_csv(OUT_DIR / f"{model_name}_cka_delta.csv", index=False)
    diff_df.to_csv(OUT_DIR / f"{model_name}_diff_delta.csv", index=False)
    log.info(f"[{model_name}] saved -> {OUT_DIR}/{model_name}_cka_delta.csv, {model_name}_diff_delta.csv")

    plot_trajectories(model_name, cka_df)
    plot_diff(model_name, diff_df)


def plot_trajectories(model_name: str, cka_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair, g in cka_df.groupby("pair"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["cka"], marker="o", label=pair)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA (on delta vectors)")
    ax.set_title(f"{model_name}: pairwise CKA of identity-effect deltas across layers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_cka_delta_trajectories.png", dpi=150)
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
    ax.set_title(f"{model_name}: intersectional equidistance test (delta space)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_diff_delta_ci.png", dpi=150)
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