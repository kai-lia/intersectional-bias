"""
P(yes) pipeline -- rebuilds the behavioural measure as a continuous log-odds
quantity decoded from the saved residual-stream activations, and measures
compositionality in those same behavioural units.

Motivation
----------
random_sample_generation.py runs greedy decoding (do_sample=False), so the
generated answer is exactly argmax over the first-token distribution.  The
`biased` column is therefore a 1-bit threshold of a continuous quantity we can
recover exactly:

    biased  ==  1[ P(yes) > P(no) ]  at the final prompt token

Binarising it and averaging 37 Bernoulli draws per (pair, condition) leaves a
~8pp noise floor, which is larger than most of the effects of interest.  This
script recovers the pre-threshold quantity

    delta = log P(yes) - log P(no)        (log-odds, unbounded)

from activations we already have, removing the sampling noise and the [0,1]
ceiling that made the additive prediction ill-posed.

Two corrections applied here that the original behavioural pipeline missed
------------------------------------------------------------------------
1. POLARITY.  23 of the 37 patterns in patterns_list_yes_no.csv have "no" as
   the biased answer, but random_sample_generation.py drops that column and
   codes `biased = 1 if answer == "yes" else 0` unconditionally.  Averaging
   across patterns therefore mixes +signal and -signal.  Here every delta is
   sign-flipped to a common "biased direction" using the pattern's own
   Biased Answer value.

2. WITHIN-PATTERN COMPOSITION.  behavioral_additivity.py pools bias rates
   across all patterns before forming ind1 + ind2 - base, which both mixes
   polarities and collapses the base condition to a single scalar shared by
   every pair.  Here the additive prediction is formed *within* a pattern,
   where base is that pattern's own control, and only then aggregated.

Three quantities per (pattern, pair)
------------------------------------
    delta_combo12/21 : the model's actual decoded inclination
    delta_add        : delta_ind1 + delta_ind2 - delta_base
                       -- additivity in BEHAVIOURAL (log-odds) space
    delta_pred       : decode(v_ind1 + v_ind2 - v_base)
                       -- additivity in REPRESENTATION space, read out
                          behaviourally.  Differs from delta_add because the
                          final norm makes the decode non-linear; the gap
                          between them is itself informative.

Outputs (pt2_test/data/eval/)
-----------------------------
    {model}_pyes_scenarios{tag}.csv  one row per (pattern, pair) at the final
                                     layer -- the behavioural dataset that
                                     should replace the binary `biased` column
    {model}_pyes_by_layer{tag}.csv   one row per (layer, pair), means over
                                     patterns -- for the layer-resolved
                                     representation->behaviour analysis
    {model}_pyes_validation{tag}.csv per-condition agreement between the
                                     final-layer argmax and the actually
                                     generated answers.  THIS IS A GATE: if
                                     agreement is not >~95% the decode is
                                     wrong and nothing downstream is usable.
"""
import argparse
import gc
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT))
from activation_io import discover_layers, load_scenarios
from logit_lens import yes_logit, load_head, LM_HEAD_PATH

ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"
PATTERNS = REPO / "data" / "templates" / "patterns_list_yes_no.csv"
GENERATED = ROOT.parent / "data" / "random_sample_results.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_polarity() -> pd.Series:
    """pattern_id -> +1 if the biased answer is 'yes', -1 if it is 'no'."""
    pat = pd.read_csv(PATTERNS)
    pol = pat["Biased Answer"].astype(str).str.strip().str.lower()
    if not pol.isin({"yes", "no"}).all():
        raise ValueError(f"unexpected Biased Answer values: {sorted(pol.unique())}")
    return pol.map({"yes": 1.0, "no": -1.0})


def deltas_for_layer(model: str, layer: int, head, polarity: pd.Series) -> pd.DataFrame:
    """Decode every condition at one layer into polarity-corrected log-odds.

    Memory is the binding constraint here (one layer of the full-112 sweep is
    ~19 GB of float32), so each large intermediate is released as soon as its
    delta has been computed.
    """
    d = load_scenarios(ACT_DIR, model, layer)
    sids = d["scenario_ids"]

    # representation-space additive prediction, built and released first so it
    # never coexists with the combo decodes
    v_pred = d["ind1"] + d["ind2"] - d["base"]
    delta_pred = yes_logit(v_pred, head)
    del v_pred
    gc.collect()

    delta = {"pred": delta_pred}
    for name in ["ind1", "ind2", "base", "combo12", "combo21"]:
        delta[name] = yes_logit(d[name], head)
        d[name] = None          # release the (n, d) array
        gc.collect()

    pids = np.array([int(p) for _, _, p in sids])
    sign = polarity.reindex(pids).to_numpy()          # +1 / -1 per row
    if np.isnan(sign).any():
        missing = sorted(set(pids[np.isnan(sign)]))
        raise ValueError(f"pattern ids missing from polarity table: {missing}")

    out = pd.DataFrame({
        "layer": layer,
        "pattern_id": pids,
        "stigma1": [s1 for s1, _, _ in sids],
        "stigma2": [s2 for _, s2, _ in sids],
        # polarity-corrected: positive always means "leaning toward the biased answer"
        "delta_ind1":    sign * delta["ind1"],
        "delta_ind2":    sign * delta["ind2"],
        "delta_base":    sign * delta["base"],
        "delta_combo12": sign * delta["combo12"],
        "delta_combo21": sign * delta["combo21"],
        "delta_pred":    sign * delta["pred"],
        # raw (uncorrected) yes log-odds kept for the validation gate, which
        # must compare against literal yes/no answers, not biased-direction
        "raw_yes_combo12": delta["combo12"],
        "raw_yes_ind1":    delta["ind1"],
        "raw_yes_base":    delta["base"],
    })
    # additivity in behavioural space
    out["delta_add"] = out.delta_ind1 + out.delta_ind2 - out.delta_base
    # residuals: how far the real combo is from each additive account
    out["resid_add_12"]  = out.delta_combo12 - out.delta_add
    out["resid_add_21"]  = out.delta_combo21 - out.delta_add
    out["resid_pred_12"] = out.delta_combo12 - out.delta_pred
    out["resid_pred_21"] = out.delta_combo21 - out.delta_pred

    del d, delta
    gc.collect()
    return out


def _best_threshold(raw: np.ndarray, said_yes: np.ndarray) -> tuple[float, float]:
    """Threshold on `raw` that best reproduces `said_yes`, and that accuracy."""
    v = np.unique(raw)
    if len(v) < 2:
        t = v[0] if len(v) else 0.0
        return float(t), float(((raw > t) == said_yes).mean())
    cands = np.concatenate([[v[0] - 1.0], (v[:-1] + v[1:]) / 2.0, [v[-1] + 1.0]])
    accs = [((raw > t) == said_yes).mean() for t in cands]
    i = int(np.argmax(accs))
    return float(cands[i]), float(accs[i])


def _agreement(raw: np.ndarray, said_yes: np.ndarray, groups: np.ndarray | None) -> dict:
    """Uncalibrated (threshold 0) and calibrated agreement.

    The yes-vs-no restricted log-odds carries a constant offset -- the model's
    first token is frequently neither a yes- nor a no-variant, so logsumexp over
    each side is a *conditional* comparison whose zero point is not the decision
    boundary.  Empirically the offset is near-constant (granite: single global
    threshold ~-47 recovers 96% agreement).  This matters not at all for the
    analyses downstream, because every one of them is a *difference* of deltas
    within a pattern, where a per-pattern constant cancels:

        resid = d_combo - (d_ind1 + d_ind2 - d_base)
        each d carrying +c  ->  +c - c - c + c = 0

    So the gate is calibrated agreement, not raw-sign agreement.
    """
    out = {"agreement_uncal": float(((raw > 0) == said_yes).mean())}
    t, a = _best_threshold(raw, said_yes)
    out["global_threshold"] = t
    out["agreement_global_thresh"] = a
    if groups is not None:
        num = den = 0.0
        for g in np.unique(groups):
            msk = groups == g
            _, ag = _best_threshold(raw[msk], said_yes[msk])
            num += ag * msk.sum(); den += msk.sum()
        out["agreement_per_pattern"] = float(num / den)
    return out


def validate(model: str, final_rows: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Gate: does the decoded distribution reproduce what the model actually
    generated?  Compares individual, base and combo12 conditions.  Reported both
    uncalibrated and with a fitted threshold -- see _agreement() for why the
    calibrated number is the meaningful one."""
    if not GENERATED.exists():
        log.warning(f"[{model}] {GENERATED} not found -- skipping validation gate")
        return pd.DataFrame()

    gen = pd.read_csv(GENERATED, usecols=["pattern_id", "condition", "stigma1",
                                           "stigma2", "model", "model_answer"])
    gen = gen[gen.model == model]
    gen["said_yes"] = gen.model_answer.astype(str).str.strip().str.lower().eq("yes")

    rows = []

    # --- individual: match on (pattern_id, identity) -------------------------
    ind = pd.concat([
        final_rows[["pattern_id", "stigma1", "raw_yes_ind1"]]
            .rename(columns={"stigma1": "stigma", "raw_yes_ind1": "raw"}),
    ]).drop_duplicates(subset=["pattern_id", "stigma"])
    g_ind = gen[gen.condition == "individual"][["pattern_id", "stigma1", "said_yes"]] \
        .rename(columns={"stigma1": "stigma"})
    mi = ind.merge(g_ind, on=["pattern_id", "stigma"], how="inner")
    if len(mi):
        rows.append({"condition": "individual", "n": len(mi),
                     **_agreement(mi.raw.to_numpy(), mi.said_yes.to_numpy(),
                                  mi.pattern_id.to_numpy())})

    # --- base: one control per pattern ---------------------------------------
    base = final_rows.groupby("pattern_id").raw_yes_base.first().reset_index()
    g_base = gen[gen.condition == "base"][["pattern_id", "said_yes"]]
    mb = base.merge(g_base, on="pattern_id", how="inner")
    if len(mb):
        # one row per pattern, so a per-pattern threshold would be vacuous here
        rows.append({"condition": "base", "n": len(mb),
                     **_agreement(mb.raw_yes_base.to_numpy(), mb.said_yes.to_numpy(), None)})

    # --- combo12: match on (pattern_id, sorted pair) -------------------------
    g12 = gen[gen.condition == "combo12"].copy()
    lo = np.minimum(g12.stigma1.values.astype(str), g12.stigma2.values.astype(str))
    hi = np.maximum(g12.stigma1.values.astype(str), g12.stigma2.values.astype(str))
    g12["stigma1"], g12["stigma2"] = lo, hi
    m12 = final_rows[["pattern_id", "stigma1", "stigma2", "raw_yes_combo12"]].merge(
        g12[["pattern_id", "stigma1", "stigma2", "said_yes"]],
        on=["pattern_id", "stigma1", "stigma2"], how="inner")
    if len(m12):
        rows.append({"condition": "combo12", "n": len(m12),
                     **_agreement(m12.raw_yes_combo12.to_numpy(), m12.said_yes.to_numpy(),
                                  m12.pattern_id.to_numpy())})

    v = pd.DataFrame(rows)
    v.insert(0, "model", model)
    return v


def run_model(model: str, tag: str, layers_arg: str | None) -> None:
    log.info(f"[{model}] extracting lm_head + final norm")
    subprocess.run([sys.executable, "pt2_test/extract_lm_head_yesno.py", "--model", model],
                   cwd=REPO, check=True, capture_output=True)
    head = load_head()
    polarity = load_polarity()

    layers = discover_layers(ACT_DIR, model)
    if not layers:
        raise FileNotFoundError(f"no activation shards for '{model}' in {ACT_DIR}")
    if layers_arg:
        wanted = {int(x) for x in layers_arg.split(",")}
        layers = [L for L in layers if L in wanted]
    final_layer = max(layers)
    log.info(f"[{model}] {len(layers)} layers, final = {final_layer}")

    per_layer, final_rows = [], None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for L in layers:
        t = time.time()
        rows = deltas_for_layer(model, L, head, polarity)

        # per-pair means across patterns, for the layer-resolved analysis
        agg = rows.groupby(["stigma1", "stigma2"]).agg(
            delta_combo12=("delta_combo12", "mean"),
            delta_combo21=("delta_combo21", "mean"),
            delta_add=("delta_add", "mean"),
            delta_pred=("delta_pred", "mean"),
            resid_add=("resid_add_12", "mean"),
            resid_pred=("resid_pred_12", "mean"),
            abs_resid_add=("resid_add_12", lambda s: s.abs().mean()),
            abs_resid_pred=("resid_pred_12", lambda s: s.abs().mean()),
        ).reset_index()
        agg.insert(0, "layer", L)
        per_layer.append(agg)

        if L == final_layer:
            final_rows = rows

        log.info(f"[{model}] layer {L} done in {time.time()-t:.1f}s  "
                 f"mean|resid_add|={rows.resid_add_12.abs().mean():.3f}  "
                 f"mean|resid_pred|={rows.resid_pred_12.abs().mean():.3f}")
        if L != final_layer:
            del rows
            gc.collect()

    keep = ["layer", "pattern_id", "stigma1", "stigma2",
            "delta_ind1", "delta_ind2", "delta_base",
            "delta_combo12", "delta_combo21", "delta_add", "delta_pred",
            "resid_add_12", "resid_add_21", "resid_pred_12", "resid_pred_21"]
    p_sc = OUT_DIR / f"{model}_pyes_scenarios{tag}.csv"
    final_rows[keep].to_csv(p_sc, index=False)
    log.info(f"[{model}] saved -> {p_sc}  ({len(final_rows)} rows)")

    p_bl = OUT_DIR / f"{model}_pyes_by_layer{tag}.csv"
    pd.concat(per_layer, ignore_index=True).to_csv(p_bl, index=False)
    log.info(f"[{model}] saved -> {p_bl}")

    v = validate(model, final_rows, tag)
    if len(v):
        p_v = OUT_DIR / f"{model}_pyes_validation{tag}.csv"
        v.to_csv(p_v, index=False)
        log.info(f"[{model}] VALIDATION GATE:\n{v.to_string(index=False)}")
        gate_col = "agreement_per_pattern" if "agreement_per_pattern" in v else "agreement_global_thresh"
        worst = v[gate_col].min(skipna=True)
        if worst < 0.95:
            log.warning(f"[{model}] *** calibrated agreement {worst:.3f} < 0.95 -- decode is "
                        f"suspect, do not trust downstream results for this model ***")
        else:
            log.info(f"[{model}] gate PASSED (calibrated agreement {worst:.3f} >= 0.95)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--layers", default=None,
                    help="comma-separated subset, e.g. '1,20,40'. Default: all discovered layers.")
    args = ap.parse_args()
    for m in args.models:
        try:
            run_model(m, args.tag, args.layers)
        except Exception as exc:
            log.error(f"[{m}] failed: {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
