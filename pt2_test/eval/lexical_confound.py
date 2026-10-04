"""
Is displacement measuring social stigma, or lexical rarity?

The confound
------------
displacement_i = ||ind_i - base|| is the magnitude of the shift an identity term
produces in the residual stream.  Rare, long, or surprising terms produce larger
shifts for reasons that have nothing to do with what the term means socially --
and the identities at the top of the displacement ranking ("Sex Offender",
"Crystal Meth. Use Recreationally") are plainly rarer and more marked than those
at the bottom ("Unemployed", "Old Age").  If displacement is lexical, then the
peril/disgust interpretation, the within-pair asymmetry, and everything built on
the ranking are about word statistics rather than about stigma.

Phrase length is too weak a proxy to settle it (r = -0.07, R2 = 0.005).  The
right measure is the model's OWN surprisal for the phrase, which is what the
representation is actually responding to:

    surprisal_i = -log P(phrase tokens | carrier context)

computed in the same carrier sentence the extraction used, so the quantity is
matched to the thing being explained.  Reported both as a total (confounded with
length) and per-token (length-normalised).

Also reported: token count, character length, and the six human stigma
dimensions, so the regression can ask whether stigma ratings explain displacement
BEYOND what lexical properties already explain.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT))
load_dotenv(REPO / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.load_models import detect_device
from pipeline.prompt import COMBINED_PATH
from random_sample_activations import load_identities, single_phrase

OUT_DIR = ROOT.parent / "data" / "eval"
CHECKPOINTS = {
    "granite": "ibm-granite/granite-3.0-8b-instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}
# The carrier is the frame the extraction actually used, so surprisal is measured
# on the same string whose representation produced the displacement.
CARRIER = "I know someone {phrase}."


@torch.inference_mode()
def surprisals(model_id, phrases, device_map, dtype):
    tk = AutoTokenizer.from_pretrained(model_id)
    mo = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, dtype=dtype).eval()
    out = []
    for ph in phrases:
        full = CARRIER.format(phrase=ph)
        prefix = CARRIER.split("{phrase}")[0]
        ids_full = tk(full, return_tensors="pt").input_ids.to(mo.device)
        n_prefix = tk(prefix, return_tensors="pt").input_ids.shape[1]
        logits = mo(ids_full).logits[0, :-1]
        tgt = ids_full[0, 1:]
        lp = torch.log_softmax(logits.float(), -1).gather(1, tgt[:, None])[:, 0]
        # only the phrase tokens, not the carrier
        span = lp[n_prefix - 1:]
        out.append({"total": float(-span.sum()), "per_token": float(-span.mean()),
                    "n_tok": int(span.numel())})
    del mo
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return pd.DataFrame(out, index=phrases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    args = ap.parse_args()
    if len(args.models) > 1:
        import subprocess
        for m in args.models:                       # one checkpoint per process
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--models", m],
                           cwd=REPO, check=True)
        return
    if (tok := os.getenv("HF_TOKEN")):
        login(tok)
    model = args.models[0]
    combined = pd.read_csv(COMBINED_PATH)
    identities = load_identities("full", combined)
    phr = {t: single_phrase(combined, t) for t in identities}
    device, device_map, dtype, _ = detect_device()
    s = surprisals(CHECKPOINTS[model], list(phr.values()), device_map, dtype)
    s.index = list(phr.keys())
    s.to_csv(OUT_DIR / f"{model}_lexical.csv")
    print(f"[{model}] wrote {len(s)} surprisals -> {model}_lexical.csv")


if __name__ == "__main__":
    main()
