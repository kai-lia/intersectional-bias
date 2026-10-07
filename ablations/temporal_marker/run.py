"""
Ablation: does the time word ("previously", "currently", "recently") matter for
the 8 identities where it does not separate two identities?

Each of the 8 target identities is run twice -- as worded in the final run
(marked) and with only the time word deleted (unmarked) -- alone and paired with
14 fixed partners in both orders, across all 37 templates x 4 wordings.  The
other 30 time-word identities (current vs remitted pairs) are not touched: the
word is what tells them apart.

Prompts are built with final_run/extract.py's own functions, and unmarked
pairs use the same composition rule that produced identities.csv (verified to
reproduce all 12,142 pairs), so marked vs unmarked differ by the deleted word
and nothing else.

Per (template, wording): 1 base + 16 target singles + 14 partner singles
+ 8 x 2 variants x 14 partners x 2 orders = 479 prompts; 70,892 per model.

Outputs (this folder only):
    outputs/{model}/p{PP}_w{W}.npz   rows, readout, and last-token activations
                                     at 3 key layers per model
    run.py --dry-run prints every target phrase pair and sample prompts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "final_run"))
import extract as fr  # noqa: E402  final_run/extract.py

OUT = HERE / "outputs"

# final-run wording -> time word deleted; nothing else changes
TARGETS = {
    "Had An Abortion Previously":   "who had an abortion",
    "Was Raped Previously":         "who was raped",
    "Divorced Previously":          "who was divorced",
    "Gang Member Currently":        "who is a gang member",
    "On Parole Currently":          "who is on parole",
    "Cleft Lip And Palate Current": "with a cleft lip and palate",
    "Stroke Recent Avg. Impairment":       "who had a stroke",
    "Heart Attack Recent Avg. Impairment": "who had a heart attack",
}
# pilot's 15 stratified identities minus Gang Member (a target)
PARTNERS = [
    "Sex Offender", "Crystal Meth. Use Recreationally", "Transgender",
    "Lesbian/Gay/Bisexual/Non-Heterosexual", "Muslim", "Fundamentalist Christian",
    "Depression Remitted", "Breast Cancer Current Avg. Symptoms",
    "Overweight Current Avg. Severity", "Black", "Asian", "Latina", "Old Age", "Homeless",
]
LAYERS = {"granite": [7, 24, 40], "llama": [8, 16, 32], "mistral": [8, 19, 32]}


def _tail(b):
    return b[4:] if b.startswith("who ") else "has " + b[5:]


def compose(a, b):
    """Same rule that built identities.csv pairs (incl. the 'who has X and' fix)."""
    if a.startswith("with ") and b.startswith("with "):
        return f"{a} and {b[5:]}"
    if a.startswith("with "):
        return f"who has {a[5:]} and {_tail(b)}"
    return f"{a} and {_tail(b)}"


def build_rows(single_phrase):
    """(kind, target, variant, partner, order, phrase); row 0 is the base."""
    rows = [("base", "", "", "", "", None)]
    for t, unmarked in TARGETS.items():
        for v, ph in (("marked", single_phrase[t]), ("unmarked", unmarked)):
            rows.append(("target", t, v, "", "", ph))
    for p in PARTNERS:
        rows.append(("partner", "", "", p, "", single_phrase[p]))
    for t, unmarked in TARGETS.items():
        for v, ph in (("marked", single_phrase[t]), ("unmarked", unmarked)):
            for p in PARTNERS:
                rows.append(("pair", t, v, p, "target_first", compose(ph, single_phrase[p])))
                rows.append(("pair", t, v, p, "partner_first", compose(single_phrase[p], ph)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    ap.add_argument("--patterns", nargs="+", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tpl, _, single_phrase, _, combo_phrase = fr.load_inputs()
    rows = build_rows(single_phrase)
    # marked rows must equal the final run's own phrases
    for kind, t, v, p, order, ph in rows:
        if kind == "pair" and v == "marked":
            key = (t, p) if order == "target_first" else (p, t)
            assert combo_phrase[key] == ph, (key, ph, combo_phrase[key])
    if args.patterns:
        tpl = tpl[tpl.pattern_id.isin(args.patterns)]
    groups = [(int(r.pattern_id), w, str(r[fr.WORDING_COLS[w]])) for _, r in tpl.iterrows() for w in range(4)]
    print(f"{len(rows)} prompts/group, {len(groups)} groups, {len(rows) * len(groups):,} prompts/model")

    if args.dry_run:
        for t, u in TARGETS.items():
            print(f"  {single_phrase[t]:45} -> {u}")
        pid, w, template = groups[0]
        for i in [1, 2, 31, 32, 31 + 28, 32 + 28]:
            print(f"  [{rows[i][2] or rows[i][0]:8} {rows[i][4]:13}] {fr.fill(template, rows[i][5])}")
        return

    cols = {k: np.array([r[i] for r in rows], dtype=str)
            for i, k in enumerate(["kind", "target", "variant", "partner", "order"])}
    for name in args.models:
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [g for g in groups if not (out_dir / f"p{g[0]:02d}_w{g[1]}.npz").exists()]
        print(f"[{name}] {len(groups) - len(todo)} done, {len(todo)} to go")
        if not todo:
            continue
        model, tok, device, _, auto_batch = fr.load_model(name)
        yes_ids, no_ids = fr.answer_token_ids(tok)
        t_start = time.time()
        for k, (pid, w, template) in enumerate(todo, 1):
            prompts = [fr.base_prompt(template)] + [fr.fill(template, r[5]) for r in rows[1:]]
            hidden, logp, _ = fr.run_group(prompts, model, tok, args.batch_size or auto_batch, yes_ids, no_ids)
            p = np.exp(logp)
            fr.atomic_savez(out_dir / f"p{pid:02d}_w{w}.npz", pattern_id=pid, wording_id=w, **cols,
                            layers=np.array(LAYERS[name]), hidden=hidden[[l - 1 for l in LAYERS[name]]],
                            p_yes=p[:, :len(yes_ids)].sum(1), p_no=p[:, len(yes_ids):].sum(1))
            eta = (time.time() - t_start) / k * (len(todo) - k) / 60
            print(f"[{name}] p{pid:02d}_w{w} [{k}/{len(todo)}] ETA {eta:.0f} min", flush=True)
        del model, tok
        (out_dir / "info.json").write_text(json.dumps({"yes_ids": yes_ids, "no_ids": no_ids,
                                                       "layers": LAYERS[name]}))


if __name__ == "__main__":
    main()
