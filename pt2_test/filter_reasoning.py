"""
Stage 1 — flag and drop degenerate reasoning generations from pt2_test/main.py's
output before activation extraction.

Degenerate = the model's `Reasoning` text doesn't reflect real reasoning:
  - improper_output : no yes/no found in the raw generation (model_answer == "improper output")
  - empty_reasoning  : nothing left after stripping the yes/no answer
  - repetition_loop  : degenerate token/phrase repetition (low type-token ratio),
                        a common failure mode when generation runs past a natural
                        stopping point

Writes:
  pt2_test/data/stigma_reasoning_clean.csv   — rows that passed all checks
  pt2_test/data/degeneracy_report.csv        — drop counts broken out by
                                                model x prompt_style x condition,
                                                so degeneracy-rate differences across
                                                conditions are visible, not just
                                                totaled away
"""
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

INPUT_CSV  = ROOT / "data" / "results_pt2.csv"
CLEAN_CSV  = ROOT / "data" / "stigma_reasoning_clean.csv"
REPORT_CSV = ROOT / "data" / "degeneracy_report.csv"

_MIN_WORDS_FOR_REPETITION_CHECK = 12
_TYPE_TOKEN_RATIO_FLOOR = 0.35  # below this, treat as a repetition loop


def _condition(row) -> str:
    """B / L / BL / LB label for a stigma1/stigma2 pair; '' outside the target set."""
    s1 = row["stigma1"]
    s2 = row["stigma2"]
    s2 = None if pd.isna(s2) else s2
    if s1 == "Black" and s2 is None:
        return "B"
    if s1 == "Lesbian" and s2 is None:
        return "L"
    if s1 == "Black" and s2 == "Lesbian":
        return "BL"
    if s1 == "Lesbian" and s2 == "Black":
        return "LB"
    return ""


def classify(answer: str, reasoning) -> str:
    """Return a degeneracy reason, or '' if the row is clean."""
    if answer == "improper output":
        return "improper_output"

    text = "" if pd.isna(reasoning) else str(reasoning).strip()
    if not text:
        return "empty_reasoning"

    words = text.lower().split()
    if len(words) >= _MIN_WORDS_FOR_REPETITION_CHECK:
        ttr = len(set(words)) / len(words)
        if ttr < _TYPE_TOKEN_RATIO_FLOOR:
            return "repetition_loop"

    return ""


def main():
    if not INPUT_CSV.exists():
        log.error(f"{INPUT_CSV} not found — run pt2_test/main.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    df["degeneracy_reason"] = [
        classify(a, r) for a, r in zip(df["model_answer"], df["Reasoning"])
    ]
    df["condition"] = df.apply(_condition, axis=1)

    dropped = df[df["degeneracy_reason"] != ""]
    clean   = df[df["degeneracy_reason"] == ""]

    log.info(
        f"{len(df)} rows total — {len(dropped)} degenerate "
        f"({100 * len(dropped) / len(df):.1f}%), {len(clean)} clean"
    )

    report = (
        dropped.groupby(["model", "prompt_style", "condition", "degeneracy_reason"])
        .size()
        .reset_index(name="n_dropped")
        .sort_values("n_dropped", ascending=False)
    )
    report.to_csv(REPORT_CSV, index=False)
    log.info(f"Degeneracy breakdown saved → {REPORT_CSV}")

    clean.drop(columns=["degeneracy_reason", "condition"]).to_csv(CLEAN_CSV, index=False)
    log.info(f"Clean rows saved → {CLEAN_CSV}")


if __name__ == "__main__":
    main()