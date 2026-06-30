"""
Collect next-token logits for target tokens "black", "lesbian", and "black lesbian"
filtered to prompts in the Black / Lesbian stigma subset.

Output CSV columns
------------------
stigma1, stigma2, stigma_col, stigma_phrase, prompt_style, prompt, biased_answer, model,
logit_black,  logprob_black,
logit_lesbian, logprob_lesbian,
logprob_black_lesbian,   # joint: logprob(black) + logprob(lesbian | context + "black")
top1_token, top1_logit   # greedy next token, for sanity-check reference
"""
import argparse
import gc
import logging
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from config.settings import MODELS, STIGMA_COLS, PROMPT_STYLES, CSV_FLUSH_EVERY

_parser = argparse.ArgumentParser()
_parser.add_argument("--models",     nargs="+", choices=list(MODELS.keys()))
_parser.add_argument("--cols",       nargs="+", choices=list(STIGMA_COLS.keys()))
_parser.add_argument("--styles",     nargs="+", choices=list(PROMPT_STYLES.keys()))
_parser.add_argument("--batch-size", type=int, default=None)
_args = _parser.parse_args()

if _args.models:
    MODELS = {k: (k in _args.models) for k in MODELS}
if _args.cols:
    STIGMA_COLS = {k: (k in _args.cols) for k in STIGMA_COLS}
if _args.styles:
    PROMPT_STYLES = {k: (k in _args.styles) for k in PROMPT_STYLES}
from pipeline.combined_stigmas import run as build_combined_stigmas
from pipeline.prompt import build_prompt_rows, COMBINED_PATH, PATTERNS_YES_NO
from pipeline.load_models import detect_device, load_model, unload_model, mem_used

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUTPUT_CSV = ROOT / "data" / "logits_black_lesbian.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Stigma pairs to keep — both orderings of the combined pair are included
TARGET_STIGMAS = {
    ("Black",   None),
    ("Lesbian", None),
    ("Black",   "Lesbian"),
    ("Lesbian", "Black"),
}

# Single-word targets; "black lesbian" joint prob is computed from these two
_TARGET_WORDS = ["black", "lesbian"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _s2(val) -> str:
    return "" if pd.isna(val) else str(val)


def _s2_norm(val):
    return None if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)


def _matches_target(row) -> bool:
    return (row["stigma1"], _s2_norm(row.get("stigma2"))) in TARGET_STIGMAS


def _find_token_id(tokenizer, word: str) -> int:
    """
    Return a single token ID for `word`.
    Most sub-word tokenizers prefix a space for mid-sentence words,
    so we try " {word}" first, then bare `word`.
    Falls back to the first sub-token and logs a warning.
    """
    for surface in (f" {word}", word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            log.info(f"  '{word}' → token id {ids[0]} (surface: '{surface}')")
            return ids[0]
    ids = tokenizer.encode(f" {word}", add_special_tokens=False)
    log.warning(f"  '{word}' encodes to multiple sub-tokens {ids}; using first: {ids[0]}")
    return ids[0]


def _apply_chat(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Core logit extraction ─────────────────────────────────────────────────────

def _batch_extract_logits(
    texts: list[str],
    model,
    tokenizer,
    token_ids: dict[str, int],
) -> list[dict]:
    """
    Two forward passes per batch:
      Pass 1 — get next-token logits for "black" and "lesbian".
      Pass 2 — append the "black" token and get logit for "lesbian"
               to compute P("black lesbian") = P("black") × P("lesbian" | context + "black").
    """
    chats  = [_apply_chat(tokenizer, t) for t in texts]
    inputs = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)

    # ── Pass 1 ────────────────────────────────────────────────────────────────
    with torch.inference_mode():
        out1 = model(**inputs)

    last_logits = out1.logits[:, -1, :]          # [batch, vocab]
    log_probs1  = torch.log_softmax(last_logits, dim=-1)

    results = []
    for i in range(len(texts)):
        top1_id    = int(last_logits[i].argmax())
        rec = {
            "top1_token": tokenizer.decode([top1_id]),
            "top1_logit": float(last_logits[i, top1_id]),
        }
        for word, tid in token_ids.items():
            rec[f"logit_{word}"]   = float(last_logits[i, tid])
            rec[f"logprob_{word}"] = float(log_probs1[i, tid])
        results.append(rec)

    # ── Pass 2 — append "black" token, predict "lesbian" ─────────────────────
    black_tid = token_ids["black"]
    les_tid   = token_ids["lesbian"]

    black_col  = torch.full(
        (inputs["input_ids"].shape[0], 1), black_tid,
        dtype=torch.long, device=model.device,
    )
    ext_ids  = torch.cat([inputs["input_ids"],    black_col],                                             dim=1)
    ext_mask = torch.cat([inputs["attention_mask"],
                          torch.ones((ext_ids.shape[0], 1), dtype=inputs["attention_mask"].dtype,
                                     device=model.device)], dim=1)

    with torch.inference_mode():
        out2 = model(input_ids=ext_ids, attention_mask=ext_mask)

    log_probs2 = torch.log_softmax(out2.logits[:, -1, :], dim=-1)

    for i in range(len(texts)):
        results[i]["logprob_black_lesbian"] = (
            results[i]["logprob_black"] + float(log_probs2[i, les_tid])
        )

    return results


# ── Setup ─────────────────────────────────────────────────────────────────────

token = os.getenv("HF_TOKEN")
if token:
    login(token)
    log.info("HuggingFace login successful.")
else:
    log.error("No HF_TOKEN found in environment.")
    sys.exit(1)

if not Path(COMBINED_PATH).exists():
    log.info("Building combined_neostigmas.csv …")
    build_combined_stigmas()

active_models = [m for m, on in MODELS.items() if on]
active_cols   = [c for c, on in STIGMA_COLS.items() if on]
active_styles = [s for s, on in PROMPT_STYLES.items() if on]

log.info(f"Models: {active_models}")
log.info(f"Cols:   {active_cols}")
log.info(f"Styles: {active_styles}")

all_rows    = []
for col in active_cols:
    all_rows.extend(build_prompt_rows(PATTERNS_YES_NO, COMBINED_PATH, col=col))

target_rows = [r for r in all_rows if _matches_target(r)]
log.info(f"All rows: {len(all_rows)}  →  target subset: {len(target_rows)}")

DEVICE, DEVICE_MAP, DTYPE, _REC_BATCH = detect_device()
BATCH_SIZE = _args.batch_size or _REC_BATCH
log.info(f"Device: {DEVICE}  |  batch size: {BATCH_SIZE}")

# ── Checkpointing ─────────────────────────────────────────────────────────────
completed_keys: set = set()
write_header = not OUTPUT_CSV.exists()

if not write_header:
    existing = pd.read_csv(OUTPUT_CSV)
    for _, r in existing.iterrows():
        completed_keys.add(
            (r["stigma1"], _s2(r["stigma2"]), r["stigma_col"], r["prompt_style"], r["model"])
        )
    log.info(f"Resuming — {len(completed_keys)} rows already done.")

work  = [(row, style) for row in target_rows for style in active_styles]
total = len(work) * len(active_models)
done  = len(completed_keys)
errors = 0
log.info(f"Total calls: {total}  |  Done: {done}  |  Remaining: {total - done}")


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Inference loop ────────────────────────────────────────────────────────────

for model_name in active_models:
    pending = [
        (row, style) for row, style in work
        if (row["stigma1"], _s2(row["stigma2"]), row["stigma_col"], style, model_name)
        not in completed_keys
    ]
    pending.sort(key=lambda item: len(item[0][item[1]]))

    if not pending:
        log.info(f"[{model_name}] All done — skipping.")
        continue

    log.info(f"[{model_name}] Loading  (mem before: {mem_used(DEVICE)})")
    model, tokenizer = load_model(model_name, DEVICE_MAP, DTYPE)
    log.info(f"[{model_name}] Loaded   (mem after:  {mem_used(DEVICE)})")

    token_ids = {w: _find_token_id(tokenizer, w) for w in _TARGET_WORDS}
    log.info(f"[{model_name}] Token IDs → {token_ids}")

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
                batch_results = _batch_extract_logits(texts, model, tokenizer, token_ids)
            except Exception as exc:
                errors += len(batch)
                log.error(
                    f"[{model_name}] Batch error ({len(batch)} prompts): "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
                batch_results = [{}] * len(batch)

            for (row, style), logit_rec in zip(metas, batch_results):
                csv_buffer.append({
                    "stigma1":       row["stigma1"],
                    "stigma2":       row["stigma2"],
                    "stigma_col":    row["stigma_col"],
                    "stigma_phrase": row["stigma_phrase"],
                    "prompt_style":  style,
                    "prompt":        row[style],
                    "biased_answer": row["biased_answer"],
                    "model":         model_name,
                    **logit_rec,
                })
                completed_keys.add(
                    (row["stigma1"], _s2(row["stigma2"]), row["stigma_col"], style, model_name)
                )
                done += 1

            if len(csv_buffer) >= CSV_FLUSH_EVERY:
                flush_buffer(f)

            if done % 200 == 0:
                log.info(f"[{model_name}] {done}/{total} ({100*done/total:.1f}%)  errors={errors}")

        flush_buffer(f)

    unload_model(model_name, model, tokenizer, DEVICE)
    del model, tokenizer
    gc.collect()
    log.info(f"[{model_name}] Unloaded  (mem after: {mem_used(DEVICE)})")

log.info(f"Done. Logits saved → {OUTPUT_CSV}  (total errors: {errors})")
