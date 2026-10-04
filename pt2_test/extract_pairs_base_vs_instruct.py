"""
Pair-condition activations for BASE and INSTRUCT checkpoints, so the
partner-dependence and erasure findings can be attributed to tuning or not.

Why this exists
---------------
The base-vs-instruct probe covered single identities only, and showed the race
gap is already fully present in llama and mistral base checkpoints (tuning shift
-0.006 and +0.005) while granite's tuning quadrupled it (-0.504).  But the two
findings that actually matter -- race being the most partner-dependent category,
and race being heavily erased under composition -- need PAIR representations,
which no base checkpoint has.

Design constraint that doubles the work
---------------------------------------
The existing instruct activations were extracted with the chat template.  Base
checkpoints have none, so comparing base-plain against instruct-chat-template
would confound tuning with prompt format.  Both variants are therefore
re-extracted here with an identical plain-text prompt.  Consequence: these
activations are NOT comparable to activations_random/, only to each other.

Scope
-----
Full-grid extraction was 464k scenarios x every layer, which took ~17h/model and
380 GB.  This needs neither: a stratified subset of pairs and a handful of layers
is enough to compare partner-dependence and erasure between variants.

  112 singles x 6 patterns          =   672
  500 pairs x 6 patterns x 2 orders = 6,000
  6 base-case controls              =     6
                                      -----
                                      6,678 forward passes per variant

Pairs are stratified by displacement gap so dominated and balanced pairs are both
represented, using the same strata as the behavioural run.  Layers are sampled
across depth rather than fixed at the instruct gap-peak, since the peak may sit
elsewhere under plain-text prompts.

Output: pt2_test/data/eval/pairact_{model}_{variant}{tag}.npz
"""
import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
load_dotenv(REPO / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.load_models import detect_device
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from random_sample_activations import load_identities, single_phrase, mirror_phrase

OUT_DIR = ROOT / "data" / "eval"
ACT_DIR = ROOT / "data" / "activations_random"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PAIRS = {
    "granite": ("ibm-granite/granite-3.0-8b-base", "ibm-granite/granite-3.0-8b-instruct"),
    "llama":   ("meta-llama/Llama-3.1-8B",         "meta-llama/Llama-3.1-8B-Instruct"),
    "mistral": ("mistralai/Mistral-7B-v0.1",       "mistralai/Mistral-7B-Instruct-v0.1"),
}
PATTERN_IDS = [0, 4, 2, 33, 26, 30]   # child safety / housing / employment / healthcare / social / legal


def sample_pairs(n_pairs, seed):
    """Stratify by displacement gap, matching the behavioural run's strata."""
    from emergence import collect
    probe = pd.read_csv(OUT_DIR / "granite_identity_probe_full.csv")
    probe["gap"] = probe.ceiling_auc_real - probe.pair_auc_real
    L = int(probe.groupby("layer").gap.mean().idxmax())
    sums, pairs_idx, names, S, B, half, pats = collect("granite", L, np.random.default_rng(0))
    d = np.linalg.norm(S - B[:, None, :], axis=2).mean(0); d /= d.mean()
    gap = pd.Series({(names[a], names[b]): abs(d[a] - d[b]) for a, b in pairs_idx}).sort_values()
    q = pd.qcut(gap, 4, labels=False)
    picked = []
    for k in range(4):
        idx = pd.Series(list(gap.index[q == k]))
        picked += list(idx.sample(min(n_pairs // 4, len(idx)), random_state=seed))
    return picked


def build(pairs, identities, combined, patterns, suffix):
    rows = []
    for pid, prow in patterns.iterrows():
        tmpl = str(prow["Pattern"])
        rows.append((pid, "base", None, None, _apply_swap(str(prow["Base Case"])) + suffix))
        for t in identities:
            rows.append((pid, "single", t, None,
                         _apply_swap(tmpl.replace("{stigma}", single_phrase(combined, t))) + suffix))
        for a, b in pairs:
            p12 = combined[(combined.stigma1 == a) & (combined.stigma2 == b)]
            if p12.empty:
                continue
            rows.append((pid, "combo12", a, b,
                         _apply_swap(tmpl.replace("{stigma}", p12.iloc[0]["With Stigma"])) + suffix))
            rows.append((pid, "combo21", a, b,
                         _apply_swap(tmpl.replace("{stigma}", mirror_phrase(combined, a, b))) + suffix))
    return pd.DataFrame(rows, columns=["pattern_id", "condition", "s1", "s2", "prompt"])


@torch.inference_mode()
def extract(model_id, prompts, layers_frac, device_map, dtype, batch):
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"                     # final real token must be last
    mo = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, dtype=dtype).eval()
    n_layers = mo.config.num_hidden_layers
    layers = sorted({max(1, int(round(f * n_layers))) for f in layers_frac})
    log.info(f"  {model_id}: {n_layers} layers, extracting {layers}")
    out = {L: [] for L in layers}
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tk(chunk, return_tensors="pt", padding=True).to(mo.device)
        hs = mo(**enc, output_hidden_states=True).hidden_states
        for L in layers:
            out[L].append(hs[L][:, -1, :].float().cpu().numpy())
        del hs; gc.collect()
        if (i // batch) % 20 == 0:
            log.info(f"    {i + len(chunk)}/{len(prompts)}")
    del mo; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {L: np.concatenate(v).astype(np.float32) for L, v in out.items()}, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--variants", nargs="+", default=["base", "instruct"])
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--suffix", default="\nAnswer:")
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jobs = [(m, v) for m in args.models for v in args.variants]
    # Loading several checkpoints into one MPS process silently corrupts every
    # model after the first (see base_vs_instruct.py) -- one subprocess per job.
    if len(jobs) > 1:
        import subprocess
        for m, v in jobs:
            log.info(f"=== subprocess: {m}/{v} ===")
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--models", m, "--variants", v,
                            "--n-pairs", str(args.n_pairs), "--suffix", args.suffix,
                            "--tag", args.tag, "--seed", str(args.seed)],
                           cwd=REPO, check=True)
        return

    if (tok := os.getenv("HF_TOKEN")):
        login(tok)
    model, variant = args.models[0], args.variants[0]
    model_id = PAIRS[model][0 if variant == "base" else 1]

    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities("full", combined)
    patterns = load_patterns(PATTERNS_YES_NO).loc[PATTERN_IDS]
    picked = sample_pairs(args.n_pairs, args.seed)
    df = build(picked, identities, combined, patterns, args.suffix)
    log.info(f"[{model}/{variant}] {len(df)} prompts "
             f"({(df.condition=='single').sum()} single, {(df.condition.str.startswith('combo')).sum()} pair)")

    device, device_map, dtype, batch = detect_device()
    acts, layers = extract(model_id, df.prompt.tolist(), [0.25, 0.4, 0.5, 0.6, 0.75, 0.9],
                           device_map, dtype, batch)
    p = OUT_DIR / f"pairact_{model}_{variant}{args.tag}.npz"
    np.savez_compressed(
        p, layers=np.array(layers),
        pattern_id=df.pattern_id.to_numpy(), condition=df.condition.to_numpy(dtype=object),
        s1=df.s1.to_numpy(dtype=object), s2=df.s2.to_numpy(dtype=object),
        **{f"L{L}": acts[L] for L in layers})
    log.info(f"[{model}/{variant}] saved -> {p}  ({p.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
