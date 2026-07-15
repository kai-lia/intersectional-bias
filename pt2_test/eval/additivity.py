"""
Additive-compositionality test: does the intersectional condition's (BL/LB)
shift away from the no-stigma baseline decompose as delta_B + delta_L (each
single-axis condition's own shift from baseline), or is there a residual
component additive composition can't explain -- the "non-additive"/emergent
part of the intersectional representation?

Baseline: pt2_test/data/activations/{model}_layer{N}.npz includes a "base"
prompt_style scenario per pattern_id, where the prompt text ignores the stigma
phrase entirely (config/settings.py's "base": control, no stigma mentioned) --
B/L/BL/LB activations are bit-identical there (verified: max abs diff 0.0).
That's a genuine no-stigma reference vector per pattern, already in the data.

For each non-base scenario of the same pattern_id:
    predicted     = Base + (B - Base) + (L - Base)  =  B + L - Base
    residual_BL   = BL - predicted
    non_additive_fraction_BL = ||residual_BL|| / ||BL - Base||

A full permutation null repeats that re-pairing n_perm times (breaking the true
B/L pairing) to build a null distribution of the layer-mean fraction under
arbitrary, non-matched combination -- giving a z-score and p-value (BH-FDR
corrected across layers) for "the true pairing beats chance," not just a
single-shuffle reference line.
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
from metrics import bh_fdr

ACT_DIR = ROOT.parent / "data" / "activations"
OUT_DIR = ROOT.parent / "data" / "eval"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def discover_layers(model_name: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(model_name)}_layer(\d+)\.npz$")
    layers = [int(m.group(1)) for f in ACT_DIR.glob(f"{model_name}_layer*.npz")
              if (m := pattern.match(f.name))]
    return sorted(layers)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_null(B: np.ndarray, L: np.ndarray, BL: np.ndarray, LB: np.ndarray,
                      base_vecs: np.ndarray, other_idx: np.ndarray,
                      shift_bl: np.ndarray, shift_lb: np.ndarray,
                      n_perm: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Null distribution (length n_perm) of the layer-mean non-additive
    fraction under a random, broken B/L pairing -- re-pair each row's delta_B
    with a *different*, randomly chosen row's delta_L on every draw."""
    rng = np.random.default_rng(seed)
    null_bl = np.empty(n_perm)
    null_lb = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(other_idx))
        predicted_null = B[other_idx] + L[other_idx][perm] - base_vecs
        null_bl[i] = (np.linalg.norm(BL[other_idx] - predicted_null, axis=1) / np.linalg.norm(shift_bl, axis=1)).mean()
        null_lb[i] = (np.linalg.norm(LB[other_idx] - predicted_null, axis=1) / np.linalg.norm(shift_lb, axis=1)).mean()
    return null_bl, null_lb


def run_model(model_name: str, n_boot: int, n_perm: int, seed: int) -> None:
    layers = discover_layers(model_name)
    if not layers:
        raise FileNotFoundError(
            f"No activation files for '{model_name}' in {ACT_DIR} "
            f"-- run pt2_test/extract_activations.py first."
        )
    log.info(f"[{model_name}] found {len(layers)} layers")

    rows = []
    for layer in layers:
        data = np.load(ACT_DIR / f"{model_name}_layer{layer}.npz", allow_pickle=True)
        B, L, BL, LB = data["B"], data["L"], data["BL"], data["LB"]
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
            log.warning(f"[{model_name}] layer {layer}: no pattern has both a "
                        f"base and non-base scenario -- skipping")
            continue

        base_idx = np.array(base_idx)
        other_idx = np.array(other_idx)

        base_vecs = B[base_idx]  # B/L/BL/LB are bit-identical at the base row
        predicted = B[other_idx] + L[other_idx] - base_vecs
        shift_bl  = BL[other_idx] - base_vecs
        shift_lb  = LB[other_idx] - base_vecs
        frac_bl   = np.linalg.norm(BL[other_idx] - predicted, axis=1) / np.linalg.norm(shift_bl, axis=1)
        frac_lb   = np.linalg.norm(LB[other_idx] - predicted, axis=1) / np.linalg.norm(shift_lb, axis=1)

        ci_bl = bootstrap_ci(frac_bl, n_boot=n_boot, seed=seed)
        ci_lb = bootstrap_ci(frac_lb, n_boot=n_boot, seed=seed)

        # full permutation null: is the true B/L pairing's mean fraction
        # significantly lower (more additive) than a random, broken pairing?
        null_bl, null_lb = permutation_null(
            B, L, BL, LB, base_vecs, other_idx, shift_bl, shift_lb,
            n_perm=n_perm, seed=seed,
        )
        obs_bl, obs_lb = frac_bl.mean(), frac_lb.mean()
        # one-sided: lower fraction = more additive = "beats chance"
        z_bl = (obs_bl - null_bl.mean()) / null_bl.std()
        z_lb = (obs_lb - null_lb.mean()) / null_lb.std()
        p_bl = float(np.mean(null_bl <= obs_bl))
        p_lb = float(np.mean(null_lb <= obs_lb))

        rows.append({
            "model": model_name, "layer": layer, "n_scenarios": len(other_idx),
            "non_additive_frac_BL_mean":   obs_bl,
            "non_additive_frac_BL_median": float(np.median(frac_bl)),
            "non_additive_frac_BL_ci_low":  ci_bl[0],
            "non_additive_frac_BL_ci_high": ci_bl[1],
            "non_additive_frac_LB_mean":   obs_lb,
            "non_additive_frac_LB_median": float(np.median(frac_lb)),
            "non_additive_frac_LB_ci_low":  ci_lb[0],
            "non_additive_frac_LB_ci_high": ci_lb[1],
            "null_frac_BL_mean": null_bl.mean(),
            "null_frac_LB_mean": null_lb.mean(),
            "z_score_BL": z_bl, "p_value_BL": p_bl,
            "z_score_LB": z_lb, "p_value_LB": p_lb,
        })

    df = pd.DataFrame(rows)
    df["p_value_BL_fdr"] = bh_fdr(df["p_value_BL"].to_numpy())
    df["p_value_LB_fdr"] = bh_fdr(df["p_value_LB"].to_numpy())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{model_name}_additivity.csv", index=False)
    log.info(f"[{model_name}] saved -> {OUT_DIR}/{model_name}_additivity.csv")

    plot_additivity(model_name, df)


def plot_additivity(model_name: str, df: pd.DataFrame) -> None:
    df = df.sort_values("layer")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["layer"], df["non_additive_frac_BL_mean"], marker="o", label="BL")
    ax.fill_between(df["layer"], df["non_additive_frac_BL_ci_low"], df["non_additive_frac_BL_ci_high"], alpha=0.2)
    ax.plot(df["layer"], df["non_additive_frac_LB_mean"], marker="o", label="LB")
    ax.fill_between(df["layer"], df["non_additive_frac_LB_ci_low"], df["non_additive_frac_LB_ci_high"], alpha=0.2)
    ax.plot(df["layer"], df["null_frac_BL_mean"], linestyle="--", color="gray", label="chance-level null")
    ax.set_xlabel("Layer")
    ax.set_ylabel("non-additive fraction  ( ||residual|| / ||shift from baseline|| )")
    ax.set_title(f"{model_name}: emergent (non-additive) share of the intersectional shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_additivity.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for model_name in args.models:
        try:
            run_model(model_name, args.n_boot, args.n_perm, args.seed)
        except FileNotFoundError as exc:
            log.error(str(exc))


if __name__ == "__main__":
    main()