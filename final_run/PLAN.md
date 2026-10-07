# Final Run Plan: Activations, Reasoning, Tagging, Analysis

*Last updated 2026-10-04. A living plan: when a decision is made or a number is measured, edit it here.*

---

## 1. Why this run exists

This paper studies intersectional bias in LLMs **mechanistically**: how the internal
representations of two identities combine when a prompt mentions both ("someone who is X and Y"),
and how that composition relates to the model's answer. Every prompt needs three things, row-aligned:

| Output | Why it's needed | Produced by |
|---|---|---|
| Residual-stream activations, last prompt token, every layer | The mechanistic object of study (additivity, pair-specific interaction, order effects) | `extract.py` |
| P(yes) / P(no) for the first answer token | A cheap, exact behavioral readout from the same forward pass | `extract.py` |
| Generated answer + reasoning text | Outcome tagging (answer, identity mentions/negations), linked row by row to the activations | `generate.py` |

The paraphrase pilot (Section 3.2) showed that the pair-specific signal is real and survives wording
changes, and that noise correction with paraphrases is required. That justifies the 4-wording design at full scale.

---

## 2. Run scope (frozen)

| | |
|---|---|
| Models | `ibm-granite/granite-3.0-8b-instruct` (40 layers) · `meta-llama/Llama-3.1-8B-Instruct` (32, gated) · `mistralai/Mistral-7B-Instruct-v0.1` (32). Hidden size 4096. |
| Templates | 37, each in 4 wordings: original + 3 gender-neutral paraphrases |
| Identities | 111 singles + 12,142 ordered pairs + 1 no-identity base = **12,254 prompts per template-wording** |
| Phrasing | "With Stigma" only. No negative, positive or doubt framings. |
| Totals | **1.81M prompts per model, 5.44M overall** |
| Activations | fp16, ~100 MB per (template-wording, layer) file, **~1.55 TB**: 15,392 activation files + 444 readout files |
| Reasoning | Greedy, **512-token cap** for all models; ~0.7B generated tokens; a few GB of gzipped JSONL |

Inputs are frozen in `final_run/inputs/` with checksums in `SHA256SUMS`. The README documents
every edit made to them.

---

## 3. What we've learned

### 3.1 Infrastructure

- **Home internet is the binding constraint.** Upload is ~50 Mbps (1.4 TB takes ~2.5 days);
  download is ~800 Mbps. TB-scale data must never cross the home connection. Small results
  (CSVs, tags, single layers of a few GB) can come down.
- **Decision: everything runs in the cloud.** Extraction, generation, tagging, storage and analysis.
- **The local Mac is too slow for reasoning.** Benchmarked on an M4 Max (64 GB) with MLX
  continuous batching in bf16: about 19–20 days nonstop for the full run. MLX was ~5.5× faster
  than the repo's PyTorch-on-Mac approach, and still not fast enough.
- **The Mac's disk is 91% full** (~83 GB free). It holds ~380 GB of older `activations_random` and ~27 GB of pilot data.
- **Analysis machines need 64–128 GB RAM.** The permutation and bootstrap scripts hit ~40 GB and were
  OOM-killed on the Mac. Analysis is mostly CPU work (NumPy, PyMC): cents to a few dollars per sweep on spot machines.

### 3.2 Paraphrase pilot (15 identities × 37 templates × 4 wordings × 3 models)

- **The pair-specific interaction is real.** At middle and late layers, 75–83% of it reproduces
  across wordings, and it's ~15–21% of the pair effect (noise-corrected) at Granite L24, Llama L16 and Mistral L19.
- **Early layers are mostly wording noise.** Only 13–26% reproduces at L1–4; at Mistral L2, nothing does.
- **The raw non-additive fraction (~0.85) is mostly a prompt-length offset.** The pair-specific part is ~0.2.
- **Word order is a real effect** (reproducibility 0.70–0.77, size 0.10–0.18). Report it separately.
- **Paraphrases agree well** (~0.84 within a template vs ~0.49 across templates). Two outliers were fixed (Section 3.3).

### 3.3 Input fixes (applied)

- **Template 34:** removed "might"/"likely", so all four wordings ask the same factual question.
- **Template 11, paraphrase 1:** "hire them" became "go with this realtor"; models had treated it as a hiring template.
- **1,010 identity pairs:** "someone with autism and is short" became "someone who has autism and is short". Applied to the final-run inputs and the clean data file; backups kept.
- **Known, intentionally unchanged:**
  - The legacy `data/templates/combined_neostigmas.csv`, used by all earlier results, has the same bug in 1,111 pairs. → Limitations section.
  - Time words ("previously"/"currently") kept: they distinguish 30 of 38 identities.
  - No-identity base prompt kept as the pilot built it (decided 2026-10-06), although 145 of 148 template-wordings read e.g. "a person who is someone." → note in the methods.
  - 68 missing ordered pairs are deliberate synonym exclusions.

### 3.4 Reasoning generation (256 prompts per model, uncapped)

| | Llama | Granite | Mistral |
|---|---|---|---|
| Mean / longest answer (tokens) | 191 / 460 | 97 / 340 | 86 / 347 |
| Bare "Yes."/"No." with no reasoning | 0% | 0% | **~30%** |
| Answer opens with yes/no | 100% | 100% | 100% |

- 300 tokens would cut off 13% of Llama's answers. **512 fits every answer in the sample.**
- Different software gives the same yes/no ~99.6% of the time but **different wording** (only 20% identical
  texts). So **all generation must run on one fixed setup**, and reruns aren't bit-identical.
- **Mistral's 30% bare answers** have an answer but nothing to tag. Check whether they cluster by identity or template.

### 3.5 Pipeline verification (done locally)

- `extract.py` activations match the pilot for all three models (cosine ≥ 0.9996).
- P(yes)+P(no) = 0.96–1.00, so the yes/no token sets capture nearly all first-token probability.
- The full chain `extract.py` → `generate.py --backend hf` → `check.py` passed end to end.
- Bug pass fixed six issues:
  - smoke-test markers could make the full run skip groups
  - vLLM would crash when switching models
  - failed uploads were never retried
  - ~5 GB per batch of unused next-token scores
  - the answer parser could misread
  - generation metadata wasn't uploaded
- **Never tested:** CUDA, vLLM, rclone upload and resume. These run for the first time in the smoke test.

---

## 4. Architecture

### Option A (primary): all on Google Cloud

```
GCE GPU VM (spot, A100 80GB or H100) ──rclone──▶ GCS bucket (single region)
                                                   │
                     ┌─────────────────────────────┼──────────────────────────┐
                     ▼                             ▼                          ▼
            BigQuery external tables     Vertex Workbench / GCE CPU VM   small results ──▶ Mac
            (P(yes), tags, metadata)     (64–128 GB RAM, Jupyter,        (CSVs, figures)
                                          vector math per layer)
```

- **Why:** analysis is in the cloud and BigQuery external tables need GCS. Everything stays in one region, so no download fees. One account and one bill. Spot pricing is close to Vast.ai's.
- **Costs:** GCS ~$31/month for 1.55 TB. Downloading to the internet costs ~$0.12/GB, so only small results come down.
- **Blocker:** new projects usually have **zero GPU quota**. Request it now; approval can take days.

### Option B (fallback): Vast.ai GPUs + Cloudflare R2 + a cloud analysis VM

- Cheapest GPUs, and R2 has no download fees at all.
- Costs: R2 ~$23/month; Backblaze B2 ~$9/month as an alternative (free downloads up to 3× stored per month).
- Trade-offs:
  - three services to manage
  - no native BigQuery (convert tables, or use DuckDB on the analysis VM)
  - Vast hosts vary in reliability and upload bandwidth
  - credentials sit on third-party machines

**Choose B if** GPU quota on Google Cloud doesn't come through in time, or BigQuery isn't needed.

---

## 5. Budget (estimates, to be replaced by smoke-test measurements)

| Item | GPU-hours | Option A (GCE spot + GCS) | Option B (Vast + R2) |
|---|---|---|---|
| Smoke test + length sample | 2–5 | $5–15 | $5–10 |
| Activation pass | 10–30 (A100) | $20–80 | $15–55 |
| Reasoning (vLLM) | 40–75 (A100) | $70–200 | $50–135 |
| Outcome tagging (see Phase 5) | 0–60 | $0–150 | $0–110 |
| Upload / bandwidth fees | | $0 (ingress free) | $0–30 |
| Buffer (~25%) | | $25–110 | $20–85 |
| **One-time total** | | **~$120–555** | **~$90–425** |
| Storage | | ~$31/mo | ~$23/mo (R2) or ~$9/mo (B2) |
| Analysis compute | | ~$5–20/mo | ~$5–20/mo |
| **12-month total** | | **~$550–1,160** | **~$260–940** |

Notes:
- H100 instead of A100 is ~2× faster at ~1.5–2× the price: similar cost, half the wall-clock time.
- On-demand Google Cloud GPUs (~$5/hr for an A100) would roughly triple Option A's GPU lines.
- Keeping only key layers after analysis settles (Phase 7) cuts storage roughly in proportion.

---

## 6. Phases

### Phase 0: Decisions (you)

- [x] **Storage and compute:** Option A (Google Cloud) — decided 2026-10-04, because the $300 trial credit only applies to Google. Vast.ai is the fallback if A100 spot capacity fails. Bucket `gs://intersectionality-data` in project `intersectionality-510622`; GPUs in project `intersectionality-compute`.
- [x] **GPU type:** 1× A100 80 GB spot in us-central1 (quota approved 2026-10-04). One GPU type for the whole run; extraction and generation run one after the other.
- [x] **Extra start-of-text token for Llama/Mistral:** keep (comparable with the pilot and all earlier data); note it in the methods.
- [ ] **Tagging method** (Phase 5). *Recommendation: hybrid.*
- [ ] **Budget ceiling and deadline,** so the plan can be checked against them.

### Phase 1: Code hardening (me, before any cloud spend)

- [x] Pin Hugging Face model revisions (commit hashes) in `extract.py` and `generate.py`, and record them in `run_info`. *(2026-10-04: pinned to the pilot's snapshots, which equal current `main`.)*
- [x] Save a checksum of the prompt token IDs in each done marker, and have `check.py` confirm extract and generate match.
- [x] Record hostname, GPU and driver in every done marker and `run_info`; `check.py` lists the setups each step ran on.
- [ ] Pin the vLLM version in `requirements-generate.txt` after the smoke test confirms it works.
- [x] Apply the extra-token decision (keep: no code change).
- [x] Commit `final_run/` to git, so the VM clones a fixed version and `run_info` records the commit hash.

### Phase 2: Accounts and setup (you, with a runbook from me)

- [ ] **Option A:**
  - Create a GCP project.
  - **Request GPU quota now**, in one region.
  - Create a GCS bucket in that region with **soft delete / object versioning on**.
- [ ] **Option B:**
  - Vast.ai account with ~$50 credit.
  - R2 or B2 bucket with versioning.
- [ ] **Billing alerts and caps:** e.g. alerts at $100 / $250 / $400.
- [ ] **Credentials:**
  - Accept the Llama 3.1 license on Hugging Face.
  - Create a **read-only** HF token.
  - Create a bucket key **scoped to this bucket only**.
- [ ] **rclone remote** configured. Test with `rclone lsd`.

### Phase 3: Cloud smoke test (~$5–15)

On one GPU VM with ~200 GB disk, in the bucket's region, with **two Python environments** (`.venv-extract`, `.venv-generate`), inside `tmux`:

- [ ] **Activations:**
  - Run `extract.py --out outputs_smoke` on 1 template × 1 wording × all identities for one model.
  - Record prompts/s and peak GPU memory at the auto batch size.
- [ ] **Matches the Mac:** a few vectors line up with the local smoke test (cosine ≥ 0.999).
- [ ] **Generation:**
  - Run `generate.py --sample 1000` per model, uncapped.
  - Record length distribution, tokens/s and has-reasoning rate.
  - Confirm 512 still fits ≥ 99.5%.
- [ ] **Upload:** `--remote` moves files to the bucket and deletes them locally, then `rclone check` passes.
- [ ] **Resume:** kill a run mid-group, restart it, and confirm it continues correctly.
- [ ] **Agreement:** `check.py` on the smoke output shows ≥ 99% agreement between generated yes/no and P(yes).
- [ ] **Update Section 5** with measured numbers, and decide go or no-go.

### Phase 4: Full run

- [ ] **Activations:** `extract.py --remote …` for all three models. Watch the ETA and the bucket growth.
- [ ] **Reasoning:** `generate.py --remote …`, one process per model, run automatically.
  - It can run on a second VM in parallel once extraction is underway.
- [ ] **Verify:** pull `readout/`, `generations/`, `run_info/` and the done markers, then run `check.py --remote …`. Expect:
  - **Completeness:** 148 groups per model and 15,392 activation files.
  - **Inputs and rows:** input hashes match, and rows align across files.
  - **Agreement:** ≥ 99% between generated yes/no and P(yes).
  - **Cap hits:** < 0.5% of answers per model. If higher, rerun those rows with a higher cap.
- [ ] **Shut down every GPU VM** and confirm in the console that nothing is still billing.

### Phase 4b: Base models, activations only (after the instruct run)

Decided 2026-10-04: base-vs-instruct at full scale runs as a separate step once Phase 4 is verified.

- [ ] Models: `ibm-granite/granite-3.0-8b-base`, `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.1`. Request Hugging Face access to the gated ones.
- [ ] Activations + P(yes)/P(no) only; no reasoning generation (base models don't reliably follow the answer format).
- [ ] Prompt format: match `pt2_test/extract_pairs_base_vs_instruct.py` (no chat template), so results are comparable with the pilot.
- [ ] Estimate: +10–30 A100-hours (~$25–75 spot), +0.5–1.5 days on one GPU, +1.55 TB storage.
- [ ] Same VM setup, bucket and checks; output under separate `model=` keys (e.g. `granite_base`).

### Phase 5: Outcome tagging (5.44M texts)

Manual tagging is impossible at this scale. The options:

| Method | Cost | Quality |
|---|---|---|
| Rules (extend `score_mentions.py`: answer, strict/loose identity mention, negation) | ~$0, CPU | High precision; misses paraphrased mentions |
| Open-weight LLM judge on rented GPUs (`mention_judge.py` approach, via vLLM) | ~$50–150 | Better recall; needs a validation set |
| Paid API judge | Likely hundreds to thousands of dollars at 5.4M texts | Highest convenience |
| **Hybrid (recommended)** | ~$0 + a small judge run | Rules on everything; judge plus human labels on a stratified sample to measure rule accuracy |

- [ ] Freeze the tag schema before running: answer, has reasoning, mentions of identity A and B, negation, refusal or hedging.
- [ ] Hand-label a stratified validation set, reusing the annotation tool in `pt2_test/annotation/`.
- [ ] Run tagging in the cloud, next to the generations. Write the tags as a table that BigQuery or DuckDB can query.
- [ ] Report Mistral's bare-answer rate, and check whether it clusters by identity or template.

### Phase 6: Analysis setup

- [ ] **Loader:** write one for the new layout (`model=/layer=/p{PP}_w{W}.npz` plus wordings), modeled on `pt2_test/eval/activation_io.py`. It should read from the bucket with a local cache.
- [ ] **Analysis VM:** 64–128 GB RAM, spot, with idle auto-shutdown.
- [ ] **Tables:** BigQuery external tables (Option A) or DuckDB (Option B) over the readout, generation and tag tables.
- [ ] **Re-run the key analyses at full scale:**
  - pair-specific residual with paraphrase noise correction
  - order effects
  - MAIHDA
  - the residual-structure and erasure analyses
  - the link between behavior and representation, using the tags

### Phase 7: Long-term data management

- [ ] **Storage:** once analysis settles on key layers, move the other layers to archival storage (GCS Archive or B2), or delete them if regeneration (~$100–300) is acceptable.
- [ ] **Local Mac data:** decide keep, prune or slow-archive for the ~400 GB. Uploading 380 GB takes ~17 hours at 50 Mbps.
- [ ] **Deposit after publication** (e.g. Hugging Face Datasets).
  - Llama 3.1 outputs carry Meta's license terms ("Built with Llama" attribution and use conditions).
  - Granite and Mistral are Apache 2.0.
  - Consider whether stigma-related generations need a content note or gated access.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| GPU quota delay (Option A) | Request it now; Option B is the fallback |
| Spot preemption | Both scripts resume from done markers; at most one group is lost |
| Slow upload from a GPU host | Option A: same-region GCS is fast. Option B: pick hosts with ≥ 1 Gbps upload; the uploader applies backpressure |
| Forgotten VMs / cost overrun | Billing alerts and caps, idle auto-shutdown, an explicit teardown checklist |
| Accidental deletion | Bucket versioning or soft delete; `rclone check` after upload |
| Model repo changes mid-run | Pin revisions (Phase 1); `run_info` records them |
| extract and generate seeing different inputs | Token-ID checksums per group (Phase 1); `check.py` alignment checks |
| vLLM output not bit-reproducible | Pin the version, use one setup for the whole run, document it in the methods |
| Credential exposure on third-party hosts | Read-only HF token, bucket-scoped keys, revoke after the run |
| Answers cut off at 512 | Smoke-test sample first; `check.py` flags > 0.5% per model; rerun only those rows |
| Mistral bare answers biasing the tag sample | `has_reasoning` flag; test clustering; report the rate |

---

## 8. Open items carried from earlier work

- Legacy `combined_neostigmas.csv` grammar bug (1,111 pairs). *Recommendation: limitations note, no edit.*
- `Without Stigma` and plural columns in the clean identity file still have the "with X and is Y" bug. They're unused by this run; fix them only if those framings are ever used.
- Time-word test (`ablations/temporal_marker/`) is set up but not run. Time words are kept as they are.
- `pt2_test/data/paraphrases_review.md` is out of date relative to the CSV (80 of 148 wordings differ). Regenerate it if it's still used for review.

---

## 9. File reference

| Path | Purpose |
|---|---|
| `final_run/extract.py` | Activations + P(yes)/P(no) |
| `final_run/generate.py` | Reasoning (vLLM; `--backend hf` for local checks; `--sample` for length checks) |
| `final_run/check.py` | Completeness, hashes, alignment, agreement, cap hits |
| `final_run/inputs/` | Frozen identities and templates + `SHA256SUMS` |
| `final_run/README.md` | Scope, output layout, run commands, generation notes |
| `final_run/requirements.txt` / `requirements-generate.txt` | The two separate environments |
| `ablations/temporal_marker/` | Optional time-word test (not run) |
| `data/templates/final_clean/*.bak_*` | Backups from before the input edits |
