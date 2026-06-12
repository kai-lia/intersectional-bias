"""
Run intersectional bias prompts for Black, Lesbian, and Black Lesbian stigmas only.
Stores the model's full reasoning in a 'Reasoning' column alongside the yes/no answer.
"""
import logging
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

# parent repo on path for shared modules; pt2_test on path for local modules
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from config.settings import MODELS, STIGMA_COLS, PROMPT_STYLES, CSV_FLUSH_EVERY
from pipeline.combined_stigmas import run as build_combined_stigmas
from pipeline.prompt import build_prompt_rows, COMBINED_PATH, PATTERNS_YES_NO
from pipeline.load_models import detect_device, load_model, unload_model, mem_used
from load_models_reasoning import RUNNERS  # reasoning-aware runners

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Target stigmas ────────────────────────────────────────────────────────────
# Includes both orderings of the combined pair since the phrasing differs.
TARGET_STIGMAS = {
    ("Black",   None),
    ("Lesbian", None),
    ("Black",   "Lesbian"),
    ("Lesbian", "Black"),
}


def _s2(val) -> str:
    """Normalize stigma2 to '' for checkpoint keys (guards NaN vs None on resume)."""
    return "" if pd.isna(val) else str(val)


def _s2_norm(val):
    """Normalize stigma2 to None for set membership checks."""
    return None if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)


def _matches_target(row) -> bool:
    return (row["stigma1"], _s2_norm(row.get("stigma2"))) in TARGET_STIGMAS


# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_CSV = ROOT / "data" / "results_pt2.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── Auth ──────────────────────────────────────────────────────────────────────
token = os.getenv("HF_TOKEN")
if token:
    login(token)
    log.info("HuggingFace login successful.")
else:
    log.error("No HF_TOKEN found in environment.")
    sys.exit(1)

# ── Build combined stigmas CSV if missing ─────────────────────────────────────
if not Path(COMBINED_PATH).exists():
    log.info("combined_neostigmas.csv not found — building...")
    build_combined_stigmas()

# ── Active settings ───────────────────────────────────────────────────────────
active_models  = [m for m, on in MODELS.items() if on]
active_cols    = [c for c, on in STIGMA_COLS.items() if on]
active_styles  = [s for s, on in PROMPT_STYLES.items() if on]

log.info(f"Models:        {active_models}")
log.info(f"Stigma cols:   {active_cols}")
log.info(f"Prompt styles: {active_styles}")

# ── Build and filter prompt rows ──────────────────────────────────────────────
all_rows = []
for col in active_cols:
    all_rows.extend(build_prompt_rows(PATTERNS_YES_NO, COMBINED_PATH, col=col))

target_rows = [r for r in all_rows if _matches_target(r)]
log.info(f"All prompt rows: {len(all_rows)}  →  filtered to target stigmas: {len(target_rows)}")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE, DEVICE_MAP, DTYPE, _REC_BATCH = detect_device()
BATCH_SIZE = _REC_BATCH
log.info(f"Device: {DEVICE}  batch size: {BATCH_SIZE}")

# ── Checkpointing ─────────────────────────────────────────────────────────────
completed_keys: set = set()
write_header = not OUTPUT_CSV.exists()

if not write_header:
    existing = pd.read_csv(OUTPUT_CSV)
    for _, r in existing.iterrows():
        completed_keys.add((r["stigma1"], _s2(r["stigma2"]), r["stigma_col"], r["prompt_style"], r["model"]))
    log.info(f"Resuming — {len(completed_keys)} rows already done.")

# ── Work list ─────────────────────────────────────────────────────────────────
work = [
    (row, style)
    for row in target_rows
    for style in active_styles
]
total  = len(work) * len(active_models)
done   = len(completed_keys)
errors = 0
log.info(f"Total inference calls: {total}  |  Already done: {done}  |  Remaining: {total - done}")


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Inference — one model at a time ───────────────────────────────────────────
for model_name in active_models:
    runner = RUNNERS[model_name]

    pending = [
        (row, style) for row, style in work
        if (row["stigma1"], _s2(row["stigma2"]), row["stigma_col"], style, model_name) not in completed_keys
    ]
    pending.sort(key=lambda item: len(item[0][item[1]]))

    if not pending:
        log.info(f"[{model_name}] All prompts done — skipping.")
        continue

    log.info(f"[{model_name}] Loading  (mem before: {mem_used(DEVICE)})")
    model, tokenizer = load_model(model_name, DEVICE_MAP, DTYPE)
    log.info(f"[{model_name}] {len(pending)} prompts to run  (mem after: {mem_used(DEVICE)})")

    csv_buffer: list[dict] = []

    def flush_buffer(f):
        global write_header
        if not csv_buffer:
            return
        pd.DataFrame(csv_buffer).to_csv(f, header=write_header, index=False)
        write_header = False
        f.flush()
        csv_buffer.clear()

    with open(OUTPUT_CSV, "a", newline="") as f:
        for batch in _chunks(pending, BATCH_SIZE):
            texts = [row[style] for row, style in batch]
            metas = [(row, style) for row, style in batch]

            try:
                answers = runner(texts, model, tokenizer)
            except Exception as exc:
                errors += len(batch)
                log.error(
                    f"[{model_name}] Batch error ({len(batch)} prompts): "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
                answers = [("error", "")] * len(batch)

            for (row, style), (answer, reasoning) in zip(metas, answers):
                key = (row["stigma1"], _s2(row["stigma2"]), row["stigma_col"], style, model_name)
                csv_buffer.append({
                    "stigma1":       row["stigma1"],
                    "stigma2":       row["stigma2"],
                    "stigma_col":    row["stigma_col"],
                    "stigma_phrase": row["stigma_phrase"],
                    "prompt_style":  style,
                    "prompt":        row[style],
                    "biased_answer": row["biased_answer"],
                    "model":         model_name,
                    "model_answer":  answer,
                    "Reasoning":     reasoning,
                    "biased":        1 if answer == "yes" else 0,
                })
                completed_keys.add(key)
                done += 1

            if len(csv_buffer) >= CSV_FLUSH_EVERY:
                flush_buffer(f)

            if done % 500 == 0:
                log.info(f"[{model_name}] Progress: {done}/{total} ({100*done/total:.1f}%)  errors={errors}")

        flush_buffer(f)

    unload_model(model_name, model, tokenizer, DEVICE)
    del model, tokenizer
    log.info(f"[{model_name}] Unloaded  (mem after: {mem_used(DEVICE)})")

log.info(f"Done. Results saved to {OUTPUT_CSV}  (total errors: {errors})")
