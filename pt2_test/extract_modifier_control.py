"""
Is the recovery gap about IDENTITY composition, or about generic modifier
interference?

The confound
------------
Pairing two identities makes each less linearly decodable.  But pairing an
identity with ANY second modifier lengthens the phrase, adds an "and", and
introduces a competing referent.  If "who is Black and is left-handed" attenuates
the Black direction as much as "who is Black and who has complete deafness" does,
then the finding is about attaching a second modifier to a noun phrase, not about
how the model composes social identities -- and the paper is about something
else.

Design
------
The benchmark already ships the ideal control: every identity has a NEGATED form
("without autism", "who does not use crystal meth recreationally").  Building a
half-negated pair gives a minimal pair --

    real     "with autism and who has bipolar disorder"      (already extracted)
    control  "with autism and who does not have bipolar disorder"

-- identical in length, syntax, topic and register, differing only in whether the
second attribute APPLIES.  Both elements are benchmark identities throughout.

The tail is taken by stripping the known solo prefix from the pair string rather
than splitting on " and ", because 111 pair phrases have a compound first member
("with a cleft lip and palate currently") that a naive split truncates.

Only the control arm needs extracting: the real pairs and the singles are already
on disk from the main run, so the comparison is exact rather than approximate.

Output: pt2_test/data/eval/modctl_{model}.npz
"""
import argparse
import gc
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
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
load_dotenv(REPO / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.load_models import detect_device
from pipeline.prompt import PATTERNS_YES_NO, COMBINED_PATH, load_patterns, _apply_swap
from random_sample_activations import load_identities, single_phrase

OUT_DIR = ROOT / "data" / "eval"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINTS = {
    "granite": "ibm-granite/granite-3.0-8b-instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

N_PARTNERS = 12          # partner identities per focal identity

def stratified_identities(model, n, seed):
    """Span the displacement range rather than sampling uniformly."""
    d = pd.read_csv(OUT_DIR / f"{model}_erasure_additive_null_full.csv")
    d = d.sort_values("disp").reset_index(drop=True)
    idx = np.linspace(0, len(d) - 1, n).round().astype(int)
    return d.identity.iloc[idx].tolist()


def build(identities, combined, patterns, n_partners, seed):
    solo = combined[combined.stigma2.isna()].set_index("stigma1")
    pr = combined[combined.stigma2.notna()]
    rng = np.random.default_rng(seed)
    phrases = []
    for t in identities:
        sub = pr[pr.stigma1 == t]
        if not len(sub):
            continue
        sub = sub.iloc[rng.choice(len(sub), min(n_partners, len(sub)), replace=False)]
        for _, r in sub.iterrows():
            tail = negated_tail(t, r["Without Stigma"], solo)
            if tail:
                phrases.append((t, r["stigma2"], f"{solo['With Stigma'][t]} and {tail}"))
    log.info(f"  built {len(phrases)} half-negated pair phrases")
    rows = []
    for pid, prow in patterns.iterrows():
        tmpl = str(prow["Pattern"])
        for a, b, ph in phrases:
            rows.append((pid, "pair_half_neg", a, b,
                         _apply_swap(tmpl.replace("{stigma}", ph))))
    return pd.DataFrame(rows, columns=["pattern_id", "condition", "s1", "s2", "prompt"])


@torch.inference_mode()
def extract(model_id, prompts, layer_frac, device_map, dtype, batch):
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    mo = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, dtype=dtype).eval()
    n = mo.config.num_hidden_layers
    layers = sorted({max(1, int(round(f * n))) for f in layer_frac})
    log.info(f"  {n} layers, extracting {layers}")
    out = {L: [] for L in layers}
    for i in range(0, len(prompts), batch):
        enc = tk(prompts[i:i + batch], return_tensors="pt", padding=True).to(mo.device)
        hs = mo(**enc, output_hidden_states=True).hidden_states
        for L in layers:
            out[L].append(hs[L][:, -1, :].float().cpu().numpy())
        del hs; gc.collect()
        if (i // batch) % 25 == 0:
            log.info(f"    {i}/{len(prompts)}")
    del mo; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {L: np.concatenate(v).astype(np.float32) for L, v in out.items()}, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--n-identities", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negated-singles", action="store_true",
                    help="extract the NEGATED single-identity prompts instead of "
                         "half-negated pairs.  Needed to estimate v_i^neg and test "
                         "whether the affirmative direction encodes assertion or "
                         "merely topic-presence.")
    ap.add_argument("--layer-fracs", nargs="*", type=float, default=None,
                    help="explicit depth fractions; use the SAME values as the "
                         "half-negated pair run (0.4 0.5 0.6 0.65 0.75) so the "
                         "solo negation baseline is measured at the same layers "
                         "as the in-pair one and the two are comparable")
    ap.add_argument("--layer-stride", type=int, default=4,
                    help="with --negated-singles: probe every Nth layer, so the "
                         "affirmative/negated angle can be tracked across depth")
    args = ap.parse_args()

    if len(args.models) > 1:
        import subprocess
        for m in args.models:                        # one checkpoint per process
            log.info(f"=== subprocess: {m} ===")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--models", m,
                            "--n-identities", str(args.n_identities), "--seed", str(args.seed)]
                           + (["--negated-singles"] if args.negated_singles else [])
                           + (["--layer-fracs"] + [str(f) for f in args.layer_fracs]
                              if args.layer_fracs else
                              ["--layer-stride", str(args.layer_stride)]),
                           cwd=REPO, check=True)
        return

    if (tok := os.getenv("HF_TOKEN")):
        login(tok)
    model = args.models[0]
    combined = pd.read_csv(COMBINED_PATH)
    patterns = load_patterns(PATTERNS_YES_NO)
    if args.negated_singles:
        solo = combined[combined.stigma2.isna()].set_index("stigma1")
        rows = []
        for pid, prow in patterns.iterrows():
            tmpl = str(prow["Pattern"])
            for t in solo.index:
                neg = solo["Without Stigma"].get(t)
                if isinstance(neg, str) and neg.strip():
                    rows.append((pid, "single_neg", t, None,
                                 _apply_swap(tmpl.replace("{stigma}", neg))))
        df = pd.DataFrame(rows, columns=["pattern_id", "condition", "s1", "s2", "prompt"])
        frac = None
        out_name = (f"negsingles_{model}_matched.npz" if args.layer_fracs
                    else f"negsingles_{model}.npz")
    else:
        ids = stratified_identities(model, args.n_identities, args.seed)
        df = build(ids, combined, patterns, N_PARTNERS, args.seed)
        frac = [0.4, 0.5, 0.6, 0.65, 0.75]
        out_name = f"modctl_{model}.npz"
    log.info(f"[{model}] {len(df)} prompts over {df.pattern_id.nunique()} templates, "
             f"{df.s1.nunique()} identities")

    device, device_map, dtype, batch = detect_device()
    if frac is None:
        if args.layer_fracs:
            frac = list(args.layer_fracs)
        else:                             # every Nth layer, for the depth curve
            import transformers
            cfg = transformers.AutoConfig.from_pretrained(CHECKPOINTS[model])
            n = cfg.num_hidden_layers
            frac = [L / n for L in range(1, n + 1, args.layer_stride)]
    acts, layers = extract(CHECKPOINTS[model], df.prompt.tolist(),
                           frac, device_map, dtype, batch)
    p = OUT_DIR / out_name
    np.savez_compressed(p, layers=np.array(layers),
                        pattern_id=df.pattern_id.to_numpy(),
                        condition=df.condition.to_numpy(dtype=object),
                        s1=df.s1.to_numpy(dtype=object), s2=df.s2.to_numpy(dtype=object),
                        **{f"L{L}": acts[L] for L in layers})
    log.info(f"[{model}] saved -> {p}  ({p.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
