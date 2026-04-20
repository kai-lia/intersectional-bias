import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from huggingface_hub import login

# ── Config ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import MODELS, STIGMA_COLS, STIGMA_COL_SLUGS, PROMPT_STYLES, BATCH_SIZE, CSV_FLUSH_EVERY

# ── Args (override config/settings.py) ───────────────────────────────────────
_parser = argparse.ArgumentParser(description="Run intersectional bias benchmarking.")
_parser.add_argument("--models",  nargs="+", choices=list(MODELS.keys()),
                     help="Models to run (default: all enabled in settings.py)")
_parser.add_argument("--cols",    nargs="+", choices=list(STIGMA_COLS.keys()),
                     help="Stigma columns to use (default: all enabled in settings.py)")
_parser.add_argument("--styles",  nargs="+", choices=list(PROMPT_STYLES.keys()),
                     help="Prompt styles to use (default: all enabled in settings.py)")
_parser.add_argument("--batch-size", type=int, default=None,
                     help=f"Batch size (default: {BATCH_SIZE} from settings.py)")
_args = _parser.parse_args()

if _args.models:
    MODELS     = {k: (k in _args.models)     for k in MODELS}
if _args.cols:
    STIGMA_COLS = {k: (k in _args.cols)      for k in STIGMA_COLS}
if _args.styles:
    PROMPT_STYLES = {k: (k in _args.styles)  for k in PROMPT_STYLES}
if _args.batch_size:
    BATCH_SIZE = _args.batch_size

# ── Pipeline modules ──────────────────────────────────────────────────────────
from pipeline.combined_stigmas import run as build_combined_stigmas
from pipeline.prompt import build_prompt_rows, COMBINED_PATH, PATTERNS_YES_NO
from pipeline.load_models import detect_device, load_model, unload_model, RUNNERS, mem_used

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

def _build_output_path() -> str:
    def _slug(items: dict) -> str:
        active = [k for k, v in items.items() if v]
        return "_".join(active)

    models_slug  = _slug(MODELS)
    cols_slug    = "_".join(STIGMA_COL_SLUGS[c] for c, on in STIGMA_COLS.items() if on)
    styles_slug  = _slug(PROMPT_STYLES)
    filename = f"results__{models_slug}__{cols_slug}__{styles_slug}.csv"
    return str(ROOT / "data" / filename)

OUTPUT_CSV = _build_output_path()

# ── HuggingFace auth ──────────────────────────────────────────────────────────
token = os.getenv("HF_TOKEN")
if token:
    login(token)
    log.info("HuggingFace login successful.")
else:
    log.error("No HF_TOKEN found — gated models will fail.")
    sys.exit(1)

# ── Step 1: Build combined stigmas CSV if needed ──────────────────────────────
if not Path(COMBINED_PATH).exists():
    log.info("combined_neostigmas.csv not found — generating now.")
    build_combined_stigmas()

# ── Step 2: Resolve active settings from config ───────────────────────────────
active_models  = [m for m, on in MODELS.items() if on]
active_cols    = [c for c, on in STIGMA_COLS.items() if on]
active_styles  = [s for s, on in PROMPT_STYLES.items() if on]

log.info(f"Models:        {active_models}")
log.info(f"Stigma cols:   {active_cols}")
log.info(f"Prompt styles: {active_styles}")
log.info(f"Batch size:    {BATCH_SIZE}")

# ── Step 3: Build all prompt rows across active stigma columns ────────────────
all_rows = []
for col in active_cols:
    all_rows.extend(build_prompt_rows(PATTERNS_YES_NO, COMBINED_PATH, col=col))

log.info(f"Total prompt rows (pre-style filter): {len(all_rows)}")

# ── Step 4: Device detection ──────────────────────────────────────────────────
DEVICE, DEVICE_MAP, DTYPE = detect_device()

# ── Step 5: Checkpointing ─────────────────────────────────────────────────────
completed_keys: set = set()
write_header = not Path(OUTPUT_CSV).exists()

def _s2(val) -> str:
    """Normalize stigma2 to a consistent string — guards against NaN vs None on resume."""
    return "" if pd.isna(val) else str(val)

if not write_header:
    existing = pd.read_csv(OUTPUT_CSV)
    for _, r in existing.iterrows():
        completed_keys.add((r["stigma1"], _s2(r["stigma2"]), r["stigma_col"], r["prompt_style"], r["model"]))
    log.info(f"Resuming — {len(completed_keys)} rows already done.")

# ── Step 6: Build flat work list ──────────────────────────────────────────────
# Each item: (row_dict, style). Skips already-completed keys.
work = [
    (row, style)
    for row in all_rows
    for style in active_styles
]

total  = len(work) * len(active_models)
done   = len(completed_keys)
errors = 0
log.info(f"Total inference calls: {total}  |  Already done: {done}  |  Remaining: {total - done}")

# ── Step 7: Inference — one model at a time, batched ─────────────────────────

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


for model_name in active_models:
    runner = RUNNERS[model_name]

    pending = [
        (row, style) for row, style in work
        if (row["stigma1"], _s2(row["stigma2"]), row["stigma_col"], style, model_name) not in completed_keys
    ]
    if not pending:
        log.info(f"[{model_name}] All prompts done — skipping.")
        continue

    log.info(f"[{model_name}] Loading  (mem before: {mem_used(DEVICE)})")
    model, tokenizer = load_model(model_name, DEVICE_MAP, DTYPE)
    log.info(f"[{model_name}] {len(pending)} prompts to run  (mem after: {mem_used(DEVICE)})")

    csv_buffer: list[dict] = []

    def flush_buffer(f):
        nonlocal write_header
        if not csv_buffer:
            return
        pd.DataFrame(csv_buffer).to_csv(f, header=write_header, index=False)
        write_header = False
        f.flush()
        csv_buffer.clear()

    with open(OUTPUT_CSV, "a", newline="") as f:
        for batch in _chunks(pending, BATCH_SIZE):
            texts  = [row[style] for row, style in batch]
            metas  = [(row, style) for row, style in batch]

            try:
                answers = runner(texts, model, tokenizer)
            except Exception as exc:
                errors += len(batch)
                log.error(
                    f"[{model_name}] Batch error ({len(batch)} prompts): "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
                answers = ["error"] * len(batch)

            for (row, style), answer in zip(metas, answers):
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
                    "biased":        1 if answer == "yes" else 0,
                })
                completed_keys.add(key)
                done += 1

            if len(csv_buffer) >= CSV_FLUSH_EVERY:
                flush_buffer(f)

            if done % 500 == 0:
                log.info(f"[{model_name}] Progress: {done}/{total} ({100*done/total:.1f}%)  errors={errors}")

        flush_buffer(f)  # flush any remaining rows

    unload_model(model_name, model, tokenizer, DEVICE)
    del model, tokenizer
    log.info(f"[{model_name}] Unloaded  (mem after: {mem_used(DEVICE)})")

log.info(f"Done. Results saved to {OUTPUT_CSV}  (total errors: {errors})")
