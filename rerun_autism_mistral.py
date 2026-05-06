"""
Re-run mistral for Autism Or Autism Spectrum Disorder pairs, base style only.
Fills in the 36 missing patterns per pair caused by the checkpoint key bug.
Appends results to the existing output CSV.
"""
import logging
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from huggingface_hub import login

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import BATCH_SIZE, CSV_FLUSH_EVERY
from pipeline.prompt import build_prompt_rows, COMBINED_PATH, PATTERNS_YES_NO
from pipeline.load_models import detect_device, load_model, unload_model, RUNNERS, mem_used

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_STIGMA1 = "Autism Or Autism Spectrum Disorder"
TARGET_STYLE   = "base"
TARGET_MODEL   = "mistral"
ROOT           = Path(__file__).resolve().parent
OUTPUT_CSV     = ROOT / "data" / "output" / "results__mistral__with_single__original_positive_doubt_base.csv"

# ── Auth ──────────────────────────────────────────────────────────────────────
token = os.getenv("HF_TOKEN")
if token:
    login(token)
    log.info("HuggingFace login successful.")
else:
    log.error("No HF_TOKEN found.")
    sys.exit(1)

# ── Build all prompt rows for the target stigma1 ──────────────────────────────
all_rows = build_prompt_rows(PATTERNS_YES_NO, COMBINED_PATH, col="With Stigma")
autism_rows = [r for r in all_rows if r["stigma1"] == TARGET_STIGMA1]
log.info(f"Found {len(autism_rows)} prompt rows for '{TARGET_STIGMA1}'")

# ── Checkpoint — include prompt text in key to avoid the original bug ─────────
def _s2(val) -> str:
    return "" if pd.isna(val) else str(val)

completed_keys: set = set()
if OUTPUT_CSV.exists():
    existing = pd.read_csv(OUTPUT_CSV)
    for _, r in existing.iterrows():
        completed_keys.add((
            r["stigma1"], _s2(r["stigma2"]), r["stigma_col"],
            r["prompt_style"], r["model"], r.get("prompt", ""),
        ))
    log.info(f"Loaded {len(completed_keys)} completed keys from existing CSV.")

# ── Build pending work (base style only, skip already-done prompts) ───────────
work = [
    (row, TARGET_STYLE)
    for row in autism_rows
    if (row["stigma1"], _s2(row["stigma2"]), row["stigma_col"],
        TARGET_STYLE, TARGET_MODEL, row[TARGET_STYLE]) not in completed_keys
]

log.info(f"Pending inference calls: {len(work)}")
if not work:
    log.info("Nothing to do — all prompts already complete.")
    sys.exit(0)

# ── Device + model ────────────────────────────────────────────────────────────
DEVICE, DEVICE_MAP, DTYPE = detect_device()
model, tokenizer = load_model(TARGET_MODEL, DEVICE_MAP, DTYPE)
runner = RUNNERS[TARGET_MODEL]

# ── Inference ─────────────────────────────────────────────────────────────────
def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

csv_buffer: list[dict] = []
done = errors = 0
total = len(work)
write_header = not OUTPUT_CSV.exists()

def flush_buffer(f):
    global write_header
    if not csv_buffer:
        return
    pd.DataFrame(csv_buffer).to_csv(f, header=write_header, index=False)
    write_header = False
    f.flush()
    csv_buffer.clear()

with open(OUTPUT_CSV, "a", newline="") as f:
    for batch in _chunks(work, BATCH_SIZE):
        texts = [row[style] for row, style in batch]
        metas = [(row, style) for row, style in batch]

        try:
            answers = runner(texts, model, tokenizer)
        except Exception as exc:
            errors += len(batch)
            log.error(f"Batch error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            answers = ["error"] * len(batch)

        for (row, style), answer in zip(metas, answers):
            csv_buffer.append({
                "stigma1":       row["stigma1"],
                "stigma2":       row["stigma2"],
                "stigma_col":    row["stigma_col"],
                "stigma_phrase": row["stigma_phrase"],
                "prompt_style":  style,
                "prompt":        row[style],
                "biased_answer": row["biased_answer"],
                "model":         TARGET_MODEL,
                "model_answer":  answer,
                "biased":        1 if answer == "yes" else 0,
            })
            done += 1

        if len(csv_buffer) >= CSV_FLUSH_EVERY:
            flush_buffer(f)

        if done % 200 == 0:
            log.info(f"Progress: {done}/{total} ({100*done/total:.1f}%)  errors={errors}")

    flush_buffer(f)

unload_model(TARGET_MODEL, model, tokenizer, DEVICE)
log.info(f"Done. {done} rows appended to {OUTPUT_CSV}  (errors={errors})")