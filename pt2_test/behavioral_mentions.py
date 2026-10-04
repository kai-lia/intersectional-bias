"""
Does the model's own stated reasoning mention the identity that the geometry
says is flattened?

Everything the erasure claim currently rests on is probe geometry, and every
basis used so far derives from the same difference-in-means directions the
residual was defined against.  This tests the same claim by a route that shares
none of those assumptions: generate an open-ended justification and ask whether
the lower-displacement member of the pair gets mentioned.

Prediction from the geometry: within a pair, the LOWER-displacement identity is
mentioned less often than the higher-displacement one, with the gap widening as
the displacement gap grows (the probe showed 55->86%, 59->91%, 59->96% across
displacement-gap quartiles).  Symmetric mention rates would remove the only
non-geometric support the claim has.

The lexical-detectability confound, and how it is handled
--------------------------------------------------------
Identity labels differ enormously in how easy they are to spot in free text.
"Black" is short and distinctive; "Working In A Service Industry" is long and
endlessly paraphrasable.  Since the claim is precisely that class-type identities
are the erased ones, naive string matching would manufacture the result.

So single-identity prompts are generated too, and each identity's mention rate
THERE is its ceiling -- how often it gets named when it is the only identity
present and unambiguously relevant.  Pair mention rates are then read relative to
that ceiling, the same ceiling-relative construction that made the probe
interpretable.  An identity that is simply hard to detect lexically has a low
ceiling and is not penalised twice.

Output: {model}_mentions{tag}.csv  (one row per generated response)
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval'))
load_dotenv(REPO / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.load_models import detect_device, MODEL_IDS
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from random_sample_activations import load_identities, single_phrase, mirror_phrase

OUT_DIR = ROOT / "data" / "eval"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# one pattern per deployment domain, so the result is not a property of one scenario
PATTERNS = [0, 4, 2, 33, 26, 30]   # child safety / housing / employment / healthcare / social / legal
# The original six are 4 yes-harmful / 2 no-harmful.  That skew is what made
# every pooled behavioural measure in this project reversible: a model with a
# directional answer prior scores differently on the two polarities, and an
# unbalanced pattern set turns that into a fake effect.  EXTRA balances it to
# 6/6 while keeping domain spread (employment, housing, childcare, healthcare).
EXTRA = [31, 34, 3, 15, 20, 36]


def build_prompts(pairs, identities, combined, patterns):
    rows = []
    for pid, prow in patterns.iterrows():
        tmpl = str(prow["Pattern"])
        for t in identities:                      # ceiling condition
            rows.append({"pattern_id": pid, "condition": "single", "s1": t, "s2": None,
                         "prompt": _apply_swap(tmpl.replace("{stigma}", single_phrase(combined, t)))})
        for a, b in pairs:                        # both orderings
            p12 = combined[(combined.stigma1 == a) & (combined.stigma2 == b)]
            if p12.empty:
                continue
            rows.append({"pattern_id": pid, "condition": "combo12", "s1": a, "s2": b,
                         "prompt": _apply_swap(tmpl.replace("{stigma}", p12.iloc[0]["With Stigma"]))})
            rows.append({"pattern_id": pid, "condition": "combo21", "s1": a, "s2": b,
                         "prompt": _apply_swap(tmpl.replace("{stigma}", mirror_phrase(combined, a, b)))})
    return pd.DataFrame(rows)


@torch.inference_mode()
def generate(model_name, prompts, max_new_tokens, batch):
    mid = MODEL_IDS[model_name]
    tk = AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    device, device_map, dtype, _ = detect_device()
    mo = AutoModelForCausalLM.from_pretrained(mid, device_map=device_map, dtype=dtype).eval()
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        chats = [tk.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True) for p in chunk]
        enc = tk(chats, return_tensors="pt", padding=True).to(mo.device)
        n_in = enc["input_ids"].shape[1]
        gen = mo.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                          pad_token_id=tk.pad_token_id)
        out.extend(tk.batch_decode(gen[:, n_in:], skip_special_tokens=True))
        if (i // batch) % 10 == 0:
            log.info(f"    {i + len(chunk)}/{len(prompts)}")
    del mo
    import gc; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patterns", nargs="*", type=int, default=None,
                    help="pattern ids to generate; default the original six. "
                         "Pair sampling is deterministic given --seed and --n-pairs, "
                         "so a run on new patterns extends an existing run and the "
                         "two can simply be concatenated.")
    args = ap.parse_args()

    # One checkpoint per process.  Loading several into a single MPS process
    # silently corrupts every model after the first -- the first run of this
    # script produced fluent text for granite and "the the the aa aa" for llama.
    # Nothing raises; only the output degrades.
    if len(args.models) > 1:
        import subprocess
        for m in args.models:
            log.info(f"=== subprocess: {m} ===")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--models", m,
                   "--n-pairs", str(args.n_pairs), "--seed", str(args.seed),
                   "--tag", args.tag, "--max-new-tokens", str(args.max_new_tokens)]
            if args.patterns:
                cmd += ["--patterns"] + [str(x) for x in args.patterns]
            subprocess.run(cmd, cwd=REPO, check=True)
        return

    if (tok := os.getenv("HF_TOKEN")):
        login(tok)

    # stratify the pair sample by displacement gap so dominated AND balanced
    # pairs are both represented; a uniform sample would be mostly middling
    probe = pd.read_csv(OUT_DIR / "granite_identity_probe_full.csv")
    L = int((probe.assign(g=probe.ceiling_auc_real - probe.pair_auc_real)
             .groupby("layer").g.mean().idxmax()))
    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities("full", combined)
    # displacement gap per pair, from the ranking already computed
    from emergence import collect, unit                      # reuse the shard reader
    sums, pairs_idx, names, S, B, half, pats = collect("granite", L, np.random.default_rng(0))
    dnorm = np.linalg.norm(S - B[:, None, :], axis=2).mean(0); dnorm /= dnorm.mean()
    gap = {(names[a], names[b]): abs(dnorm[a] - dnorm[b]) for a, b in pairs_idx}

    g = pd.Series(gap).sort_values()
    rng = np.random.default_rng(args.seed)
    q = pd.qcut(g, 4, labels=False)
    per = args.n_pairs // 4
    picked = []
    for k in range(4):
        idx = g.index[q == k]
        picked += list(pd.Series(list(idx)).sample(min(per, len(idx)), random_state=args.seed))
    log.info(f"sampled {len(picked)} pairs, stratified into 4 displacement-gap strata")

    patterns = load_patterns(PATTERNS_YES_NO).loc[args.patterns or PATTERNS]
    df = build_prompts(picked, identities, combined, patterns)
    log.info(f"{len(df)} prompts per model "
             f"({(df.condition=='single').sum()} single, {(df.condition!='single').sum()} pair)")

    for m in args.models:
        log.info(f"[{m}] generating with max_new_tokens={args.max_new_tokens}")
        _, _, _, batch = detect_device()
        resp = generate(m, df.prompt.tolist(), args.max_new_tokens, batch)
        o = df.copy(); o["model"] = m; o["response"] = resp
        o["disp_gap"] = [gap.get((a, b), np.nan) if b else np.nan
                         for a, b in zip(o.s1, o.s2)]
        p = OUT_DIR / f"{m}_mentions{args.tag}.csv"
        o.to_csv(p, index=False)
        log.info(f"[{m}] saved -> {p}")


if __name__ == "__main__":
    main()
