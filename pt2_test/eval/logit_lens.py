"""
Logit-lens analysis: projects the already-extracted residual-stream vectors
(activations_random/*.npz, final-prompt-token position) through the model's
own final norm + lm_head, restricted to a curated set of yes/no token
variants -- turns "how far is this vector from additive prediction"
(the geometric metric used throughout additivity_random.py) into "how far is
the model's own *provisional next-token decision* from additive prediction,"
which is a much more direct proxy for the real behavioral non-additivity
computed by pt2_test/eval/behavioral_additivity.py from real generation.

Caveat, stated once here rather than at every call site: the model's real
answer comes after up to 300 tokens of reasoning (load_models_reasoning.py),
so the first-token yes/no lean this script computes is the model's
*immediate inclination* at the position right before generation starts, not
a guaranteed reproduction of the eventual parsed answer. Still a meaningful
per-layer probe of when that inclination diverges from additive prediction.

Works in logit space (pre-softmax), not probability space: log-odds are the
natural additive unit for combining independent evidence (a log-odds sum is
the Bayesian-style combination of two independent pieces of evidence), so
predicted_logit = logit(ind1) + logit(ind2) - logit(base) is a better-founded
"additive prediction" than trying to add raw probabilities.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from activation_io import discover_layers, load_scenarios

ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"
def lm_head_path(model: str):
    """Per-model head; falls back to the legacy shared filename."""
    p = ROOT.parent / "data" / f"lm_head_yesno_{model}.npz"
    legacy = ROOT.parent / "data" / "lm_head_yesno.npz"
    return p if p.exists() or not legacy.exists() else legacy


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def apply_final_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None,
                      norm_type: str, eps: float) -> np.ndarray:
    """x: (n, d). Replicates the model's actual final-norm formula, whichever
    it is, rather than assuming -- branches on what extract_lm_head_yesno.py
    introspected from the live model."""
    if "RMS" in norm_type:
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
        return (x / rms) * weight
    else:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mu) / np.sqrt(var + eps) * weight
        if bias is not None and bias.size:
            normed = normed + bias
        return normed


def yes_logit(x: np.ndarray, head) -> np.ndarray:
    """x: (n, d) residual-stream vectors -> (n,) log-odds of yes vs no,
    aggregated over all kept yes-like/no-like token variants via logsumexp
    (so e.g. 'Yes' and ' Yes' both contribute rather than picking one)."""
    normed = apply_final_norm(x, head["norm_weight"], head["norm_bias"], head["norm_type"], head["norm_eps"])
    logits = normed @ head["lm_head_rows"].T  # (n, n_tokens)
    yes_mask = head["is_yes"]
    yes_score = np.logaddexp.reduce(logits[:, yes_mask], axis=1)
    no_score = np.logaddexp.reduce(logits[:, ~yes_mask], axis=1)
    return yes_score - no_score  # log-odds(yes) - log-odds(no) == log-odds ratio


def load_head(model: str):
    data = np.load(lm_head_path(model), allow_pickle=True)
    return {
        "lm_head_rows": data["lm_head_rows"],
        "norm_weight": data["norm_weight"],
        "norm_bias": data["norm_bias"] if data["norm_bias"].size else None,
        "norm_type": str(data["norm_type"]),
        "norm_eps": float(data["norm_eps"]),
        "is_yes": data["is_yes"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="granite")
    parser.add_argument("--behavior-csv", default=None,
                         help="default: {model}_behavioral_additivity{tag}.csv, "
                              "as written by pt2_test/eval/behavioral_additivity.py")
    parser.add_argument("--tag", default="", help="output filename suffix, e.g. _full112 -- "
                         "also used to resolve the default --behavior-csv path")
    args = parser.parse_args()
    behavior_csv = args.behavior_csv or str(OUT_DIR / f"{args.model}_behavioral_additivity{args.tag}.csv")

    if not lm_head_path(args.model).exists():
        raise FileNotFoundError(
            f"{lm_head_path(args.model)} not found -- run pt2_test/extract_lm_head_yesno.py "
            f"on the model-serving machine first."
        )
    head = load_head(args.model)
    beh = pd.read_csv(behavior_csv)
    beh_lookup = {tuple(sorted([r.stigma1, r.stigma2])): r.abs_behavioral_residual for _, r in beh.iterrows()}

    layers = discover_layers(ACT_DIR, args.model)
    per_layer_rows = []
    per_pair_by_layer = {}

    for layer in layers:
        data = load_scenarios(ACT_DIR, args.model, layer)
        sids = data["scenario_ids"]
        ind1, ind2, base = data["ind1"], data["ind2"], data["base"]
        combo12, combo21 = data["combo12"], data["combo21"]

        lo_ind1, lo_ind2, lo_base = yes_logit(ind1, head), yes_logit(ind2, head), yes_logit(base, head)
        lo_combo12, lo_combo21 = yes_logit(combo12, head), yes_logit(combo21, head)
        predicted_logit = lo_ind1 + lo_ind2 - lo_base
        resid12 = lo_combo12 - predicted_logit
        resid21 = lo_combo21 - predicted_logit
        resid = (np.abs(resid12) + np.abs(resid21)) / 2

        rows = []
        for i, (s1, s2, pid) in enumerate(sids):
            key = tuple(sorted([s1, s2]))
            if key in beh_lookup:
                rows.append({"pair_key": key, "logit_lens_resid": resid[i]})
        df = pd.DataFrame(rows)
        pair_resid = df.groupby("pair_key")["logit_lens_resid"].mean()
        beh_vals = pd.Series({k: beh_lookup[k] for k in pair_resid.index})
        per_pair_by_layer[layer] = pair_resid

        r, p = stats.pearsonr(pair_resid, beh_vals)
        per_layer_rows.append({"layer": layer, "mean_logit_lens_resid": pair_resid.mean(), "r": r, "p": p})

    pl = pd.DataFrame(per_layer_rows)
    pl["p_fdr"] = bh_fdr(pl["p"].to_numpy())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{args.model}_logit_lens_vs_behavior{args.tag}.csv"
    pl.to_csv(out_csv, index=False)

    pd.set_option("display.width", 140)
    print(pl.to_string(index=False))
    print()
    n_sig = (pl.p_fdr < 0.05).sum()
    print(f"layers surviving FDR<0.05: {n_sig} of {len(pl)}")
    if n_sig:
        print(pl[pl.p_fdr < 0.05][["layer", "r", "p_fdr"]].to_string(index=False))
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()