"""
Permutation-null-only sibling of additivity_random.py -- computes the observed
non-additive fraction and its permutation p-value per layer, without the
bootstrap_diff / CKA pieces that pushed additivity_random.py's RSS to ~40 GB
and got the process OS-killed on this machine. Same statistical logic as the
permutation_null() call in additivity_random.py; scenarios script's memory
footprint (~a few GB per layer) as the reference point since neither builds
raw Gram matrices.

Null: pair each row's ind1-shift with a *different*, randomly chosen row's
ind2-shift (breaking the true stigma1/stigma2 pairing while preserving the
marginal distribution of individual shifts). p-value = fraction of permuted
non-additive-fraction means <= observed. Small p -> observed is at the low
(more-additive) end of the null; large p -> observed is at the high end
(more-non-additive than chance).

Output: pt2_test/data/eval/{model}_permutation_null{tag}.csv, one row per
layer, with BH-FDR-corrected p-values across the layer set.
"""
import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from metrics import bh_fdr
from activation_io import discover_layers, load_scenarios

ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_layer(model_name: str, layer: int, n_perm: int, seed: int) -> dict:
    data = load_scenarios(ACT_DIR, model_name, layer)
    ind1, ind2 = data["ind1"], data["ind2"]
    combo12, combo21, base = data["combo12"], data["combo21"], data["base"]
    n = len(ind1)

    predicted = ind1 + ind2 - base
    shift12 = combo12 - base
    shift21 = combo21 - base
    denom12 = np.linalg.norm(shift12, axis=1)
    denom21 = np.linalg.norm(shift21, axis=1)
    frac12 = np.linalg.norm(combo12 - predicted, axis=1) / denom12
    frac21 = np.linalg.norm(combo21 - predicted, axis=1) / denom21
    obs12, obs21 = float(frac12.mean()), float(frac21.mean())

    rng = np.random.default_rng(seed)
    null12 = np.empty(n_perm)
    null21 = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        predicted_null = ind1 + ind2[perm] - base
        null12[i] = (np.linalg.norm(combo12 - predicted_null, axis=1) / denom12).mean()
        null21[i] = (np.linalg.norm(combo21 - predicted_null, axis=1) / denom21).mean()

    p12 = float(np.mean(null12 <= obs12))
    p21 = float(np.mean(null21 <= obs21))
    z12 = float((obs12 - null12.mean()) / null12.std())
    z21 = float((obs21 - null21.mean()) / null21.std())

    row = {
        "model": model_name, "layer": layer, "n_scenarios": n, "n_perm": n_perm,
        "non_additive_frac_combo12_mean": obs12,
        "non_additive_frac_combo21_mean": obs21,
        "null_frac_combo12_mean": float(null12.mean()),
        "null_frac_combo12_std":  float(null12.std()),
        "null_frac_combo21_mean": float(null21.mean()),
        "null_frac_combo21_std":  float(null21.std()),
        "z_score_combo12": z12, "p_value_combo12": p12,
        "z_score_combo21": z21, "p_value_combo21": p21,
    }
    del data, ind1, ind2, combo12, combo21, base, predicted, shift12, shift21, null12, null21
    gc.collect()
    return row


def _partial_path(model_name: str, tag: str) -> Path:
    return OUT_DIR / f"{model_name}_permutation_null{tag}_partial.csv"


def run_model(model_name: str, n_perm: int, seed: int, tag: str) -> None:
    layers = discover_layers(ACT_DIR, model_name)
    if not layers:
        raise FileNotFoundError(f"No activation shards for '{model_name}' in {ACT_DIR}")

    partial_path = _partial_path(model_name, tag)
    if partial_path.exists():
        prev = pd.read_csv(partial_path)
        rows = prev.to_dict("records")
        done = set(prev["layer"].tolist())
        log.info(f"[{model_name}] resuming -- {len(done)} layers already done: {sorted(done)}")
    else:
        rows, done = [], set()

    log.info(f"[{model_name}] {len(layers)} layers total, n_perm={n_perm}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        if layer in done:
            continue
        t = time.time()
        row = compute_layer(model_name, layer, n_perm, seed)
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        log.info(f"[{model_name}] layer {layer} done in {time.time()-t:.1f}s "
                 f"({len(rows)}/{len(layers)}) "
                 f"obs12={row['non_additive_frac_combo12_mean']:.3f} "
                 f"null12={row['null_frac_combo12_mean']:.3f} "
                 f"z12={row['z_score_combo12']:+.1f} p12={row['p_value_combo12']:.3f}")

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    df["p_value_combo12_fdr"] = bh_fdr(df["p_value_combo12"].to_numpy())
    df["p_value_combo21_fdr"] = bh_fdr(df["p_value_combo21"].to_numpy())
    out_path = OUT_DIR / f"{model_name}_permutation_null{tag}.csv"
    df.to_csv(out_path, index=False)
    partial_path.unlink(missing_ok=True)
    log.info(f"[{model_name}] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="_full")
    args = parser.parse_args()
    for model_name in args.models:
        try:
            run_model(model_name, args.n_perm, args.seed, args.tag)
        except (FileNotFoundError, ValueError) as exc:
            log.error(str(exc))


if __name__ == "__main__":
    main()
