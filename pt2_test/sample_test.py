"""
small test
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from pipeline.prompt import load_patterns, make_prompts
from pipeline.load_models import detect_device, load_model, unload_model
from load_models_reasoning import RUNNERS
from extract_activations import extract_activations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_CSV   = ROOT / "data" / "sample_test_results.csv"
OUTPUT_NPZ   = ROOT / "data" / "sample_test_activations.npz"
PATTERN_ID   = 0          # "playing with kids who live in a household..." pattern
PROMPT_STYLE = "original"

# (label, stigma1, stigma2, stigma_phrase) — phrasing matches how
# pipeline/combined_stigmas.py would render a real "With Stigma" combo
IDENTITIES = [
    ("B",  "Black",  None,     "who is Black"),
    ("F",  "Female", None,     "who is female"),
    ("BF", "Black",  "Female", "who is Black and is female"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", choices=["granite", "llama", "mistral"])
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)
    login(token)

    pattern_row = load_patterns().iloc[PATTERN_ID]
    prompts = [
        make_prompts(pattern_row, phrase)[PROMPT_STYLE]1
        for _, _, _, phrase in IDENTITIES
    ]
    for (label, *_), p in zip(IDENTITIES, prompts):
        log.info(f"[{label}] prompt: {p}")

    device, device_map, dtype, _ = detect_device()
    model, tokenizer = load_model(args.model, device_map, dtype)
    runner = RUNNERS[args.model]

    answers = runner(prompts, model, tokenizer)

    log.info("Extracting residual-stream activations (1 forward pass per prompt)...")
    per_condition_layers = {
        label: extract_activations(prompt, model, tokenizer)
        for (label, *_), prompt in zip(IDENTITIES, prompts)
    }
    layers = sorted(next(iter(per_condition_layers.values())).keys())
    activations = {
        label: np.stack([layer_vecs[l] for l in layers])  # (n_layers, d)
        for label, layer_vecs in per_condition_layers.items()
    }
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_NPZ, layers=np.array(layers), **activations)
    for label, arr in activations.items():
        log.info(f"[{label}] activations shape: {arr.shape}")
    log.info(f"Saved → {OUTPUT_NPZ}")

    rows = []
    for (label, s1, s2, phrase), prompt, (answer, reasoning) in zip(IDENTITIES, prompts, answers):
        rows.append({
            "pattern_id":    PATTERN_ID,
            "condition":     label,
            "stigma1":       s1,
            "stigma2":       s2,
            "stigma_phrase": phrase,
            "prompt_style":  PROMPT_STYLE,
            "prompt":        prompt,
            "model":         args.model,
            "model_answer":  answer,
            "Reasoning":     reasoning,
            "biased":        1 if answer == "yes" else 0,
        })

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.max_colwidth", 80)
    print(df[["condition", "stigma_phrase", "model_answer", "biased", "Reasoning"]])
    log.info(f"Saved → {OUTPUT_CSV}")

    unload_model(args.model, model, tokenizer, device)


if __name__ == "__main__":
    main()