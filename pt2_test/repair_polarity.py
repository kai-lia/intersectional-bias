"""
Repair the polarity coding in results files written before pipeline/polarity.py.

Every generation script originally wrote `biased = 1 if answer == "yes" else 0`,
ignoring the scenario's own `Biased Answer`.  23 of 37 yes/no patterns are
"no"-biased, so 62% of rows carry an inverted label.  Regenerating is not an
option -- random_sample_results.csv alone is 1.39M rows and took ~3 days -- so
the column is recomputed in place from the answer text.

What this writes
    biased_answer   the scenario's polarity, added where it was dropped
    biased          recomputed correctly
    biased_naive    the original value, preserved

biased_naive is kept deliberately: audit_simulation.py's arm A reproduces the
naive audit to show what a practitioner following the documented format would
conclude, and that arm needs the uncorrected labels to be honest.

The repaired copy is built alongside, then the original is renamed to
<name>.pre_polarity_fix and the new file takes its place -- nothing is
overwritten, and a failure part-way leaves the original untouched.
"""
import argparse

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from pipeline.polarity import polarity_by_pattern, is_biased

TARGETS = [
    ROOT / "data" / "random_sample_results.csv",
    ROOT / "data" / "random_sample_results_max300_partial.csv",
    ROOT / "data" / "results_pt2.csv",
    ROOT / "data" / "results_pt2_old_schema.csv",
    ROOT / "data" / "stigma_reasoning_clean.csv",
    REPO / "data" / "output" / "results__granite__with_single__original_positive_doubt_base.csv",
    REPO / "data" / "output" / "results__llama__with_single__original_positive_doubt_base.csv",
    REPO / "data" / "output" / "results__mistral__with_single__original_positive_doubt_base.csv",
]
CHUNK = 100_000


def _polarity(chunk: pd.DataFrame):
    """Recover each row's biased answer, preferring the column if it survived."""
    if "biased_answer" in chunk.columns and chunk.biased_answer.notna().all():
        return chunk.biased_answer, "existing biased_answer column"
    if "pattern_id" in chunk.columns:
        return chunk.pattern_id.map(polarity_by_pattern()), "pattern_id -> template lookup"
    return None, None


def repair(path: Path, dry_run: bool) -> None:
    if not path.exists():
        print(f"  {path.name}: missing, skipped")
        return
    head = pd.read_csv(path, nrows=1)
    if "biased" not in head.columns:
        print(f"  {path.name}: no `biased` column, skipped")
        return
    if "biased_naive" in head.columns:
        print(f"  {path.name}: already repaired, skipped")
        return
    if _polarity(head)[0] is None:
        print(f"  {path.name}: no biased_answer and no pattern_id, CANNOT repair")
        return

    # Streamed: these files run to 750 MB and the text columns balloon in memory.
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = n_changed = old_sum = new_sum = 0
    src = None
    try:
        for k, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK)):
            pol, src = _polarity(chunk)
            if pol.isna().any():
                raise ValueError(f"{pol.isna().sum()} rows in chunk {k} have no resolvable polarity")
            fixed = pd.Series([is_biased(a, b) for a, b in zip(chunk.model_answer, pol)],
                              index=chunk.index)
            n += len(chunk)
            n_changed += int((fixed != chunk["biased"]).sum())
            old_sum += int(chunk["biased"].sum())
            new_sum += int(fixed.sum())
            if not dry_run:
                chunk["biased_naive"] = chunk["biased"]
                chunk["biased_answer"] = pol
                chunk["biased"] = fixed
                chunk.to_csv(tmp, index=False, mode="w" if k == 0 else "a", header=(k == 0))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"  {path.name}: FAILED, left untouched -- {e}")
        return

    print(f"  {path.name}: {n:,} rows, polarity from {src}")
    print(f"      {n_changed/n:.1%} of labels change   "
          f"mean biased {old_sum/n:.3f} -> {new_sum/n:.3f}")
    if dry_run:
        tmp.unlink(missing_ok=True)
        return
    # rename rather than copy: the original becomes the backup, no second write
    path.rename(path.with_suffix(path.suffix + ".pre_polarity_fix"))
    tmp.rename(path)
    print(f"      written; original at {path.name}.pre_polarity_fix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()
    targets = [Path(f) for f in args.files] if args.files else TARGETS
    print(f"{'DRY RUN -- ' if args.dry_run else ''}repairing polarity in {len(targets)} files\n")
    for t in targets:
        repair(t, args.dry_run)


if __name__ == "__main__":
    main()
