"""
Additive-compositionality test on the random 100-stigma-pair sample
(pt2_test/random_sample_activations.py), rather than just Black/Lesbian --
tests whether the non-additive/emergent pattern found for that one pair
generalizes across many different identity/trait combinations.

Same logic as additivity.py: for each (pair, pattern) scenario,
    predicted        = ind1 + ind2 - base
    non_additive_frac_combo12 = ||combo12 - predicted|| / ||combo12 - base||
    non_additive_frac_combo21 = ||combo21 - predicted|| / ||combo21 - base||

but here "ind1"/"ind2" are a different, randomly sampled pair of stigmas on
every row -- so the aggregate result answers "does this pattern hold in
general," not "does it hold for Black/Lesbian specifically."
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from metrics import bh_fdr, linear_cka_gram, bootstrap_diff
from activation_io import discover_layers, load_scenarios

ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_null(ind1, ind2, combo12, combo21, base_vecs,
                      shift12, shift21, n_perm: int = 1000, seed: int = 0):
    """Null: pair each row's ind1-shift with a *different*, randomly chosen
    row's ind2-shift (breaking the true stigma1/stigma2 pairing)."""
    rng = np.random.default_rng(seed)
    n = len(ind1)
    null12 = np.empty(n_perm)
    null21 = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        predicted_null = ind1 + ind2[perm] - base_vecs
        null12[i] = (np.linalg.norm(combo12 - predicted_null, axis=1) / np.linalg.norm(shift12, axis=1)).mean()
        null21[i] = (np.linalg.norm(combo21 - predicted_null, axis=1) / np.linalg.norm(shift21, axis=1)).mean()
    return null12, null21


def run_model(model_name: str, n_boot: int, n_perm: int, seed: int, tag: str = "") -> None:
    layers = discover_layers(ACT_DIR, model_name)
    if not layers:
        raise FileNotFoundError(
            f"No activation files for '{model_name}' in {ACT_DIR} "
            f"-- run pt2_test/random_sample_activations.py first."
        )
    log.info(f"[{model_name}] found {len(layers)} layers")

    # incremental checkpoint so a killed/interrupted run doesn't lose
    # already-computed layers (this analysis takes long enough on the full
    # 3885-scenario dataset that resuming matters) -- no FDR column yet,
    # since BH-FDR is a correction across the *complete* set of layers and
    # would be wrong to compute on a partial set.
    partial_path = OUT_DIR / f"{model_name}_additivity_random{tag}_partial.csv"
    rows = []
    done_layers = set()
    if partial_path.exists():
        prev = pd.read_csv(partial_path)
        rows = prev.to_dict("records")
        done_layers = set(prev["layer"].tolist())
        log.info(f"[{model_name}] resuming -- {len(done_layers)} layers already done: {sorted(done_layers)}")

    for layer in layers:
        if layer in done_layers:
            continue
        t_layer = time.time()
        data = load_scenarios(ACT_DIR, model_name, layer)
        ind1, ind2 = data["ind1"], data["ind2"]
        combo12, combo21, base = data["combo12"], data["combo21"], data["base"]
        n = len(ind1)

        predicted = ind1 + ind2 - base
        shift12 = combo12 - base
        shift21 = combo21 - base
        frac12 = np.linalg.norm(combo12 - predicted, axis=1) / np.linalg.norm(shift12, axis=1)
        frac21 = np.linalg.norm(combo21 - predicted, axis=1) / np.linalg.norm(shift21, axis=1)

        ci12 = bootstrap_ci(frac12, n_boot=n_boot, seed=seed)
        ci21 = bootstrap_ci(frac21, n_boot=n_boot, seed=seed)

        null12, null21 = permutation_null(ind1, ind2, combo12, combo21, base,
                                           shift12, shift21, n_perm=n_perm, seed=seed)
        obs12, obs21 = frac12.mean(), frac21.mean()
        z12 = (obs12 - null12.mean()) / null12.std()
        z21 = (obs21 - null21.mean()) / null21.std()
        p12 = float(np.mean(null12 <= obs12))
        p21 = float(np.mean(null21 <= obs21))

        # does combo12/combo21 lean toward ind1 or ind2? CKA is scale-invariant
        # (normalized by Gram-matrix Frobenius norms), so unlike raw distance
        # it isn't distorted by the residual-stream norm growth at deep layers.
        cka_12_1 = linear_cka_gram(combo12, ind1)
        cka_12_2 = linear_cka_gram(combo12, ind2)
        lean12 = cka_12_1 - cka_12_2  # positive -> combo12 closer to ind1
        lean12_ci = bootstrap_diff(combo12, ind1, ind2, n_boot=n_boot, seed=seed)
        lean12_ci_low, lean12_ci_high = np.percentile(lean12_ci, [2.5, 97.5])

        cka_21_1 = linear_cka_gram(combo21, ind1)
        cka_21_2 = linear_cka_gram(combo21, ind2)
        lean21 = cka_21_1 - cka_21_2  # positive -> combo21 closer to ind1
        lean21_ci = bootstrap_diff(combo21, ind1, ind2, n_boot=n_boot, seed=seed)
        lean21_ci_low, lean21_ci_high = np.percentile(lean21_ci, [2.5, 97.5])

        rows.append({
            "model": model_name, "layer": layer, "n_scenarios": n,
            "non_additive_frac_combo12_mean": obs12,
            "non_additive_frac_combo12_ci_low": ci12[0],
            "non_additive_frac_combo12_ci_high": ci12[1],
            "non_additive_frac_combo21_mean": obs21,
            "non_additive_frac_combo21_ci_low": ci21[0],
            "non_additive_frac_combo21_ci_high": ci21[1],
            "null_frac_combo12_mean": null12.mean(),
            "null_frac_combo21_mean": null21.mean(),
            "z_score_combo12": z12, "p_value_combo12": p12,
            "z_score_combo21": z21, "p_value_combo21": p21,
            "lean_combo12_toward_ind1": lean12,
            "lean_combo12_ci_low": lean12_ci_low, "lean_combo12_ci_high": lean12_ci_high,
            "lean_combo21_toward_ind1": lean21,
            "lean_combo21_ci_low": lean21_ci_low, "lean_combo21_ci_high": lean21_ci_high,
        })

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        log.info(f"[{model_name}] layer {layer} done in {time.time() - t_layer:.1f}s "
                 f"({len(rows)}/{len(layers)} layers, p12={p12:.3f}, p21={p21:.3f})")

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    df["p_value_combo12_fdr"] = bh_fdr(df["p_value_combo12"].to_numpy())
    df["p_value_combo21_fdr"] = bh_fdr(df["p_value_combo21"].to_numpy())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{model_name}_additivity_random{tag}.csv", index=False)
    partial_path.unlink(missing_ok=True)
    log.info(f"[{model_name}] saved -> {OUT_DIR}/{model_name}_additivity_random{tag}.csv")

    plot_additivity(model_name, df, tag)
    plot_lean(model_name, df, tag)


def plot_lean(model_name: str, df: pd.DataFrame, tag: str = "") -> None:
    df = df.sort_values("layer")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["layer"], df["lean_combo12_toward_ind1"], marker="o", label="combo12")
    ax.fill_between(df["layer"], df["lean_combo12_ci_low"], df["lean_combo12_ci_high"], alpha=0.2)
    ax.plot(df["layer"], df["lean_combo21_toward_ind1"], marker="o", label="combo21")
    ax.fill_between(df["layer"], df["lean_combo21_ci_low"], df["lean_combo21_ci_high"], alpha=0.2)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA(combo, ind1) - CKA(combo, ind2)  (positive -> leans toward ind1)")
    ax.set_title(f"{model_name}: does the combo lean toward the first stigma, 100 random pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_additivity_random{tag}_lean.png", dpi=150)
    plt.close(fig)


def plot_additivity(model_name: str, df: pd.DataFrame, tag: str = "") -> None:
    df = df.sort_values("layer")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["layer"], df["non_additive_frac_combo12_mean"], marker="o", label="combo12")
    ax.fill_between(df["layer"], df["non_additive_frac_combo12_ci_low"], df["non_additive_frac_combo12_ci_high"], alpha=0.2)
    ax.plot(df["layer"], df["non_additive_frac_combo21_mean"], marker="o", label="combo21")
    ax.fill_between(df["layer"], df["non_additive_frac_combo21_ci_low"], df["non_additive_frac_combo21_ci_high"], alpha=0.2)
    ax.plot(df["layer"], df["null_frac_combo12_mean"], linestyle="--", color="gray", label="chance-level null")
    ax.set_xlabel("Layer")
    ax.set_ylabel("non-additive fraction  ( ||residual|| / ||shift from baseline|| )")
    ax.set_title(f"{model_name}: emergent (non-additive) share, 100 random stigma pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_additivity_random{tag}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="output filename suffix, e.g. _full112 -- "
                         "keeps a full-scale rerun from overwriting existing fixed-15 results")
    args = parser.parse_args()

    for model_name in args.models:
        try:
            run_model(model_name, args.n_boot, args.n_perm, args.seed, args.tag)
        except FileNotFoundError as exc:
            log.error(str(exc))


if __name__ == "__main__":
    main()
