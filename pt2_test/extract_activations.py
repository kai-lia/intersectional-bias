"""
Stage 2 — extract final-token residual-stream activations for the Black/Lesbian
2x2 intersectional design (B, L, BL, LB) from pt2_test/filter_reasoning.py's
degeneracy-filtered output.

Only the *prompt* side of each generation is used — activations come from a single
forward pass with output_hidden_states=True, not from re-running generation. The
chat template is reapplied exactly as it was for generation
(pt2_test/load_models_reasoning.py) so the extracted residual stream matches the
one that actually produced the stored Reasoning/model_answer.

scenario_id = (pattern_id, prompt_style). A scenario only contributes to the
output if all four conditions (B, L, BL, LB) survived degeneracy filtering for
that model — incomplete quadruplets are logged and dropped, since downstream CKA /
permutation-null analysis assumes matched rows across the four condition matrices.

stigma_col is pinned to "With Stigma" (affirmative phrasing, e.g. "who is Black"),
not the negated/plural variants also present in the source data — mixing those in
would turn this into a 16-condition design instead of the intended 4.
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

load_dotenv()

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from pipeline.load_models import detect_device, load_model, unload_model, mem_used

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CLEAN_CSV = ROOT / "data" / "stigma_reasoning_clean.csv"
OUT_DIR   = ROOT / "data" / "activations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STIGMA_COL = "With Stigma"

CONDITION_KEYS = {
    "B":  ("Black", None),
    "L":  ("Lesbian", None),
    "BL": ("Black", "Lesbian"),
    "LB": ("Lesbian", "Black"),
}
CONDITIONS = list(CONDITION_KEYS)


def extract_activations(prompt: str, model, tokenizer) -> dict[int, np.ndarray]:
    """{layer_index: activation_vector} — residual stream at the final prompt
    token, for layers 1..n_layers (skips the embedding layer, index 0)."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    return {
        layer: out.hidden_states[layer][0, -1, :].float().cpu().numpy()
        for layer in range(1, len(out.hidden_states))
    }


def _match(group: pd.DataFrame, s1: str, s2) -> pd.DataFrame:
    if s2 is None:
        return group[(group.stigma1 == s1) & group.stigma2.isna()]
    return group[(group.stigma1 == s1) & (group.stigma2 == s2)]


def run_for_model(model_name: str, df: pd.DataFrame) -> None:
    sub = df[(df["model"] == model_name) & (df["stigma_col"] == STIGMA_COL)]
    if sub.empty:
        log.warning(f"[{model_name}] no rows at stigma_col='{STIGMA_COL}' — skipping.")
        return

    device, device_map, dtype, _ = detect_device()
    model, tokenizer = load_model(model_name, device_map, dtype)
    n_layers = model.config.num_hidden_layers
    log.info(f"[{model_name}] {n_layers} layers  (mem after load: {mem_used(device)})")

    activations = {layer: {c: [] for c in CONDITIONS} for layer in range(1, n_layers + 1)}
    scenario_ids, dropped = [], []

    groups = list(sub.groupby(["pattern_id", "prompt_style"]))
    for scenario_id, group in groups:
        row_by_cond = {}
        for c, (s1, s2) in CONDITION_KEYS.items():
            match = _match(group, s1, s2)
            if len(match):
                row_by_cond[c] = match.iloc[0]

        if len(row_by_cond) < len(CONDITIONS):
            dropped.append((scenario_id, sorted(set(CONDITIONS) - set(row_by_cond))))
            continue

        scenario_ids.append(scenario_id)
        for c, row in row_by_cond.items():
            layer_vecs = extract_activations(row["prompt"], model, tokenizer)
            for layer, vec in layer_vecs.items():
                activations[layer][c].append(vec)

    log.info(
        f"[{model_name}] {len(scenario_ids)}/{len(groups)} complete quadruplets "
        f"({len(dropped)} dropped)"
    )
    if dropped:
        for sid, missing in dropped[:20]:
            log.info(f"[{model_name}]   dropped {sid}: missing {missing}")
        if len(dropped) > 20:
            log.info(f"[{model_name}]   ... and {len(dropped) - 20} more")

    for layer, by_cond in activations.items():
        payload = {c: np.stack(by_cond[c]) for c in CONDITIONS}
        payload["scenario_ids"] = np.array(scenario_ids, dtype=object)
        np.savez(OUT_DIR / f"{model_name}_layer{layer}.npz", **payload)

    log.info(f"[{model_name}] saved {n_layers} layer files → {OUT_DIR}")

    unload_model(model_name, model, tokenizer, device)
    del model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    args = parser.parse_args()

    if not CLEAN_CSV.exists():
        log.error(f"{CLEAN_CSV} not found — run pt2_test/filter_reasoning.py first.")
        sys.exit(1)

    token = os.getenv("HF_TOKEN")
    if token:
        login(token)
        log.info("HuggingFace login successful.")
    else:
        log.error("No HF_TOKEN found in environment.")
        sys.exit(1)

    df = pd.read_csv(CLEAN_CSV)
    for model_name in args.models:
        run_for_model(model_name, df)


if __name__ == "__main__":
    main()