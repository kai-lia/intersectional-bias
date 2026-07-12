from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
PATTERNS_YES_NO = str(_ROOT / "data/templates/patterns_list_yes_no.csv")
COMBINED_PATH   = str(_ROOT / "data/templates/combined_neostigmas.csv")

_ANSWER_SWAP = ("Answer with yes/no/can't tell", "Answer first only with yes or no, then provide your reasoning")


#help ers

def _apply_swap(text: str) -> str:
    return text.replace(_ANSWER_SWAP[0], _ANSWER_SWAP[1])


def load_patterns(path: str = PATTERNS_YES_NO) -> pd.DataFrame:
    return pd.read_csv(path)


def load_stigmas(path: str = COMBINED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# prompt struct

def make_prompts(pattern_row: pd.Series, stigma_phrase: str) -> dict:
    """
    fill a pattern template with a stigma phrase and return all prompt variants

    return a dict with: original, positive, doubt, base
    all keys share the same structure as intersectional_benchmarking.make_prompts
    """
    def _fill(template: str) -> str:
        return _apply_swap(str(template).replace("{stigma}", stigma_phrase))

    return {
        "original": _fill(pattern_row["Pattern"]),
        "positive":  _fill(pattern_row["Positively Biasing Version"]),
        "doubt":     _fill(pattern_row["Doubt Biasing Version"]),
        "base":      _apply_swap(str(pattern_row["Base Case"])),
    }


def build_prompt_rows(
    patterns_path: str = PATTERNS_YES_NO,
    stigmas_path: str  = COMBINED_PATH,
    col: str = "With Stigma",
) -> list[dict]:
    """
    cross patterns, stigma rows and return a flat list of dicts
    each dict carries all four prompt variants + other data
    """
    patterns = load_patterns(patterns_path)
    stigmas  = load_stigmas(stigmas_path)

    rows = []
    for _, stigma_row in stigmas.iterrows():
        stigma_phrase = str(stigma_row[col])
        for pat_idx, pat_row in patterns.iterrows():
            prompts = make_prompts(pat_row, stigma_phrase)
            rows.append({
                "pattern_id":   pat_idx,
                "stigma1":      stigma_row.get("stigma1", stigma_row["Stigma"]),
                "stigma2":      stigma_row.get("stigma2"),
                "stigma_col":   col,
                "stigma_phrase": stigma_phrase,
                "biased_answer": pat_row.get("Biased Answer"),
                **prompts,
            })
    return rows

