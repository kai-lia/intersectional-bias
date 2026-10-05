"""
Post-run checks for extract.py + generate.py.

  completeness  done markers and file counts per model (activations, readout,
                generations) against what the inputs imply
  inputs        run_info hashes equal the frozen inputs/ hashes
  agreement     generated answer vs argmax(P(yes), P(no)) from the activation
                pass, row by row; disagreeing rows are written out so they can
                be excluded or inspected
  generation    share of answers that hit the cap, have no reasoning, or have
                no parsable yes/no

Runs on a local directory.  After a --remote run the big files live in the
bucket; the activations are only counted (via `rclone lsf`), and the small
readout + generations folders are pulled first:
    rclone copy REMOTE:bucket/final_run outputs --include "readout/**" \
        --include "generations/**" --include "run_info/**" --include "done*/**"
    python check.py --remote REMOTE:bucket/final_run
"""
import argparse
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import extract as fr

N_LAYERS = {"granite": 40, "llama": 32, "mistral": 32}


def count(root: Path, remote: str | None, sub: str, suffix: str) -> int:
    if remote:
        r = subprocess.run(["rclone", "lsf", "-R", "--files-only", f"{remote}/{sub}"],
                           capture_output=True, text=True)
        return sum(1 for line in r.stdout.splitlines() if line.endswith(suffix))
    return sum(1 for _ in (root / sub).rglob(f"*{suffix}")) if (root / sub).exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=fr.HERE / "outputs")
    ap.add_argument("--remote", help="count activations in the bucket instead of locally")
    ap.add_argument("--models", nargs="+", default=list(fr.MODEL_IDS))
    args = ap.parse_args()

    tpl, identities, _, pairs, _ = fr.load_inputs()
    n_groups = len(tpl) * 4
    n_prompts = 1 + len(identities) + len(pairs)
    want_hash = {p.name: fr.sha256(p) for p in (fr.IDENTITIES_CSV, fr.TEMPLATES_CSV)}
    checks_dir = args.out / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    for m in args.models:
        print(f"\n=== {m}")
        c = {
            "done (extract)": (count(args.out, None, f"done/model={m}", ".done"), n_groups),
            "activation files": (count(args.out, args.remote, f"activations/model={m}", ".npz"), n_groups * N_LAYERS[m]),
            "readout files": (count(args.out, None, f"readout/model={m}", ".npz"), n_groups),
            "done (generate)": (count(args.out, None, f"done_generate/model={m}", ".done"), n_groups),
            "generation files": (count(args.out, None, f"generations/model={m}", ".jsonl.gz"), n_groups),
        }
        for sub in ("done", "done_generate"):
            short = [f.name for f in (args.out / sub / f"model={m}").glob("*.done")
                     if json.loads(f.read_text()).get("n_prompts") != n_prompts]
            if short:
                ok = False
                print(f"  {sub}: {len(short)} marker(s) from a smaller run (e.g. a smoke test): {short[:3]}")
        markers = {sub: {f.stem: json.loads(f.read_text())
                         for f in (args.out / sub / f"model={m}").glob("*.done")}
                   for sub in ("done", "done_generate")}
        both = sorted(markers["done"].keys() & markers["done_generate"].keys())
        mismatch = [s for s in both if markers["done"][s].get("token_ids_sha256") is None
                    or markers["done"][s].get("token_ids_sha256") != markers["done_generate"][s].get("token_ids_sha256")]
        ok &= not mismatch
        print(f"  token ids extract = generate: {len(both) - len(mismatch)} / {len(both)} groups"
              + (f"  MISMATCH OR MISSING: {mismatch[:3]}" if mismatch else "  ok"))
        for sub, step in (("done", "extract"), ("done_generate", "generate")):
            setups = sorted({(d.get("gpu"), d.get("driver")) for d in markers[sub].values()}, key=str)
            note = "" if len(setups) <= 1 else "  NOTE: more than one setup -- report it in the methods"
            print(f"  {step} ran on (gpu, driver): {setups}{note}")
        for k, (have, want) in c.items():
            flag = "ok" if have == want else "MISSING"
            ok &= have == want
            print(f"  {k:18} {have:6d} / {want:6d}  {flag}")
        for step in ("model", "generate_model"):
            p = args.out / "run_info" / f"{step}={m}.json"
            if p.exists():
                same = json.loads(p.read_text()).get("inputs_sha256") == want_hash
                ok &= same
                print(f"  run_info {step:15} inputs {'match' if same else 'DO NOT MATCH'}")

        rows, bad = 0, []
        stats = {"hit_cap": 0, "no_reasoning": 0, "no_answer": 0, "answer_not_at_start": 0}
        for gen in sorted((args.out / f"generations/model={m}").glob("*.jsonl.gz")):
            ro = args.out / f"readout/model={m}" / gen.name.replace(".jsonl.gz", ".npz")
            with gzip.open(gen, "rt") as f:
                recs = [json.loads(x) for x in f]
            rows += len(recs)
            stats["hit_cap"] += sum(r["finish"] == "length" for r in recs)
            stats["no_reasoning"] += sum(not r["has_reasoning"] for r in recs)
            stats["no_answer"] += sum(r["answer"] == "none" for r in recs)
            stats["answer_not_at_start"] += sum(not r.get("answer_at_start", True) for r in recs)
            if not ro.exists():
                continue
            z = np.load(ro)
            if len(z["p_yes"]) != len(recs) or any(
                    r["stigma1"] != z["stigma1"][r["row"]] or r["stigma2"] != z["stigma2"][r["row"]] for r in recs):
                ok = False
                print(f"  ROWS DO NOT ALIGN between {gen.name} and its readout -- skipping its agreement check")
                continue
            pred = np.where(z["p_yes"] >= z["p_no"], "yes", "no")
            for r, pr, py, pn in zip(recs, pred, z["p_yes"], z["p_no"]):
                if r["answer"] != pr:
                    bad.append({"file": gen.name, "row": r["row"], "kind": r["kind"], "stigma1": r["stigma1"],
                                "stigma2": r["stigma2"], "p_yes": float(py), "p_no": float(pn),
                                "generated": r["answer"], "text_start": r["text"][:120]})
        if rows:
            print(f"  generated rows {rows:,}: hit cap {stats['hit_cap'] / rows:.3%}, "
                  f"no reasoning {stats['no_reasoning'] / rows:.1%}, no yes/no {stats['no_answer'] / rows:.2%}, "
                  f"yes/no not at start (review) {stats['answer_not_at_start'] / rows:.2%}")
            print(f"  answer vs P(yes) disagreement: {len(bad):,} rows ({len(bad) / rows:.2%})")
            pd.DataFrame(bad).to_csv(checks_dir / f"disagreements_model={m}.csv", index=False)
            if stats["hit_cap"] / rows > 0.005:
                print("  NOTE: >0.5% of answers hit the cap -- consider rerunning those rows with a higher cap")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
