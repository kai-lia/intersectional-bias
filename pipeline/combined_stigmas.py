import re
from pathlib import Path
import pandas as pd

COLS = ["Stigma", "With Stigma", "Without Stigma", "With Stigma - plural", "Without Stigma - plural"]
OUT_COLS = ["stigma1", "stigma2"] + COLS
_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = str(_ROOT / "data/templates/neostigmas.csv")
OUTPUT_PATH = str(_ROOT / "data/templates/combined_neostigmas.csv")

_PREFIX = re.compile(r"^(who |with |without )", re.IGNORECASE)
_WITH = re.compile(r"^with ", re.IGNORECASE)
_WITHOUT = re.compile(r"^without ", re.IGNORECASE)

_WITH_REPLACEMENT = {
    "With Stigma": "who has ",
    "Without Stigma": "who does not have ",
    "With Stigma - plural": "who have ",
    "Without Stigma - plural": "who do not have ",
    "Stigma": "",
}


def _normalize_row2(val: str, col: str) -> str:
    if _WITH.match(val):
        remainder = _WITH.sub("", val)
        return _WITH_REPLACEMENT[col] + remainder
    if _WITHOUT.match(val):
        remainder = _WITHOUT.sub("", val)
        return _WITH_REPLACEMENT[col] + remainder
    return _PREFIX.sub("", val)


def combine_stigma(row1: dict, row2: dict) -> dict:
    combined = {"stigma1": row1["Stigma"], "stigma2": row2["Stigma"]}
    for col in COLS:
        val2 = _normalize_row2(str(row2[col]), col)
        combined[col] = f"{row1[col]} and {val2}"
    return combined


def save_output(rows: list, path: str = OUTPUT_PATH) -> None:
    pd.DataFrame(rows, columns=OUT_COLS).to_csv(path, index=False)
    print(f"Saved {len(rows)} rows to {path}")


def run(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> None:
    df = pd.read_csv(input_path, usecols=COLS).dropna(subset=["Stigma"])
    records = df.reset_index(drop=True).to_dict("records")

    original_rows = [{"stigma1": r["Stigma"], "stigma2": None, **r} for r in records]

    combined_rows = []
    for i, stigma1 in enumerate(records):
        for j, stigma2 in enumerate(records):
            if i == j:
                continue
            combined_rows.append(combine_stigma(stigma1, stigma2))

    save_output(original_rows + combined_rows, output_path)


def main():
    run()


if __name__ == "__main__":
    main()