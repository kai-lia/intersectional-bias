# final_run

The full run, kept apart from `pt2_test/` so nothing gets mixed up. It has two steps over the same prompts:

1. `extract.py`: residual-stream activations (every layer) plus P(yes)/P(no), from one forward pass per prompt.
2. `generate.py`: the generated answer and reasoning (greedy, capped at 512 tokens).

`check.py` verifies both afterwards. All three scripts read only `inputs/` and write only `outputs/` (or `--out`).
They import nothing from the rest of the repo, so you can copy this folder to a cloud GPU machine on its own.

## Scope

| | |
|---|---|
| Models | granite-3.0-8b-instruct (40 layers), Llama-3.1-8B-Instruct (32), Mistral-7B-Instruct-v0.1 (32) |
| Templates | 37, each in 4 wordings (original + 3 paraphrases, gender-neutral) |
| Identities | 111 singles + 12,142 ordered pairs + 1 no-identity base = 12,254 prompts per template-wording |
| Layers | all, last prompt token, stored fp16 |
| Also saved | next-token P(yes) / P(no) at the answer position (from the same forward pass) |
| Reasoning | full generated answer per prompt, greedy, cap 512 tokens for all models, with finish status and a has-reasoning flag |
| Not included | negative / positive / doubt framings |

1.81M prompts per model, 5.44M in total, about **1.55 TB** of activations.

## Inputs (frozen)

`inputs/identities.csv` is a copy of `data/templates/final_clean/clean_combined_neostigmas.csv`.
`inputs/templates.csv` is a copy of `data/templates/final_clean/gn_paraphrases_review.csv`, which includes the
2026-10-02 edits: template 34 has "might" and "likely" removed, and template 11 paraphrase 1 now asks
"go with this realtor". Checksums are in `inputs/SHA256SUMS`; each model's `run_info` records them too.
To change an input, edit the copy here and regenerate `SHA256SUMS`. Don't point the script back at `data/`.

`inputs/identities.csv` has one edit relative to its source (2026-10-02): 10 identities are worded
"with X" (autism, schizophrenia, complete blindness, ...). In the 1,010 pairs where one of them comes
first and the partner is a "who ..." phrase, the pair read "someone with autism and is short". These are
now "someone who has autism and is short". Single prompts and all other pairs are unchanged. The
pilot and `pt2_test` data used the old wording for these pairs.

The 68 ordered pairs missing from 111×110 are deliberate near-synonym exclusions in the identity file
(e.g. Black + African American, Gay + Lesbian).

Prompts are built exactly as in `pt2_test/paraphrase_pilot.py`, so final-run and pilot vectors are directly
comparable: smoke-test cosine ≥ 0.9996 against the pilot at every layer. One deliberate difference: the
no-identity base prompt for template 8 no longer contains a stray `" ."`.

## Output layout

```
outputs/
  activations/model={m}/layer={LL}/p{PP}_w{W}.npz   one layer of one template-wording, ~100 MB
  readout/model={m}/p{PP}_w{W}.npz                  P(yes)/P(no) per prompt
  done/model={m}/p{PP}_w{W}.done                    written last; used to resume
  run_info/model={m}.json                           versions, model revision, input hashes, timing
  generations/model={m}/p{PP}_w{W}.jsonl.gz         answer + reasoning per prompt (generate.py)
  done_generate/model={m}/p{PP}_w{W}.done           generate.py resume marker
  run_info/generate_model={m}.json                  backend + version, cap, input hashes
  checks/disagreements_model={m}.csv                rows where the generated yes/no disagrees with P(yes)
```

Each done marker records the prompt count, a SHA-256 of the group's prompt token IDs, and the hostname, GPU and
driver it ran on. `check.py` requires the extract and generate token digests to match for every group, and lists
the (GPU, driver) setups each step used. Models load from pinned Hugging Face commits (`MODEL_REVISIONS` in
`extract.py`, the snapshots the pilot used); `run_info` records the pinned and the actually loaded revision.

All per-prompt files list rows in the same order: `[base, 111 singles, 12,142 pairs]`. Row *i* of a
generation file is row *i* of the matching readout file and of every activation file for that template-wording.
Each generation row holds `row, kind, stigma1, stigma2, text, answer, answer_at_start, has_reasoning, n_tokens, finish`.
Read them with `pd.read_json(path, lines=True)`.

### Generation notes (from a 256-prompt uncapped benchmark per model, 2026-10-03)

| Model | Mean answer | Longest | Bare "Yes."/"No." with no reasoning |
|---|---|---|---|
| Llama 3.1 8B | 191 tokens | 460 | 0% |
| Granite 3.0 8B | 97 | 340 | 0% |
| Mistral 7B v0.1 | 86 | 347 | ~30% |

- The 512-token cap fits every answer in that sample. `check.py` flags a model if more than 0.5% of its answers hit the cap.
- Mistral often answers with only "Yes."/"No.". Those rows have `has_reasoning = false`. They still have a yes/no answer but nothing to tag.
- Generation (vLLM) is a separate step from the activation pass (transformers). Both get identical token IDs, but
  different software can make the reasoning text drift slightly. `check.py` compares each generated yes/no with
  P(yes) and writes out the rows that disagree, so you can exclude them.

Activation `.npz` keys match the pilot's: `singles_vec`, `singles_stigma`, `base_vec`, `combo_vec`,
`combo_stigma1`, `combo_stigma2`, `pattern_id`, `wording_id`, `layer`. Labels are plain strings, so no
`allow_pickle` is needed. One layer of one model is a single prefix (`activations/model=llama/layer=16/`,
148 files, ~15 GB), which makes it easy to sync or read just the layers you need.

## Running

Use two separate environments: vLLM pins its own torch version.

```bash
python -m venv .venv-extract && .venv-extract/bin/pip install -r requirements.txt
python -m venv .venv-generate && .venv-generate/bin/pip install -r requirements-generate.txt
export HF_TOKEN=...            # Llama is gated

# step 1: activations (in .venv-extract)

python extract.py --dry-run                                   # counts + sample prompts
python extract.py --models granite --patterns 0 --wordings 0 \
    --identities Black Asian Latina --out outputs_smoke        # ~1 minute smoke test
python extract.py --remote REMOTE:bucket/final_run            # full run

# step 2: reasoning (in .venv-generate)
python generate.py --sample 1000                              # uncapped length check first
python generate.py --remote REMOTE:bucket/final_run           # full run, cap 512

# step 3: verify
rclone copy REMOTE:bucket/final_run outputs --include "readout/**" --include "generations/**" \
    --include "run_info/**" --include "done*/**"
python check.py --remote REMOTE:bucket/final_run
```

`generate.py --backend hf` runs the same thing with transformers. It's slow, but it works on a Mac for checking the pipeline.

- `--remote` uses rclone (works with R2, B2 or GCS). Each finished template-wording is moved to the bucket in the
  background, so local disk holds only a few groups at a time.
  Set the remote up first with `rclone config`.
- **Resuming:** rerun the same command. Finished groups are skipped (with `--remote`, done markers are pulled
  from the bucket first), so a stopped or preempted VM loses at most one group (~2–5 min of GPU time).
- Run it inside `tmux` or `nohup` so an SSH disconnect doesn't kill it.
- `--batch-size` overrides the automatic choice (128 on 80 GB GPUs, 64 on 40 GB). Batches are sorted
  longest-first, so an out-of-memory error shows up in the first batch, not hours in.
