# Runbook: running final_run on Google Cloud

Step-by-step commands for the smoke test and the full run. You type everything in **Cloud Shell**
(the terminal built into the Google Cloud console) or, after `ssh`, on the VM. Nothing needs
installing on the Mac.

| | |
|---|---|
| GPU project | `intersectionality-compute` (1× A100 80GB **spot**, us-central1) |
| Bucket | `gs://intersectionality-data` (project `intersectionality-510622`) |
| VM | `final-run-a100`, `a2-ultragpu-1g`, 250 GB disk, runs as `code-runner` (bucket access only, no key file) |
| Cost | ~$2.50/h while the VM is RUNNING; ~$0.85/day for the disk while it exists |

**Safety nets built in:**
- **48 h limit:** each start of the VM may run at most 48 h, then it stops itself.
- **Idle shutdown:** the VM shuts down after ~30 min with no GPU job and nobody logged in (paused while a run is active).
- **Full run:** shuts the VM down when finished or if a step fails.
- **Stop alert:** an email ~10 min after the VM stops for any reason (section B.2), so an interrupted run is never left idle for long.
- **Preflight:** the full run and the smoke test refuse to start unless the GPU, both environments, the bucket, the model cache, the Hugging Face token, the inputs and the disk all check out.
- **Budget alerts:** at $125 / $250 / $375 / $500, arriving a few hours late.

Scripts are in `final_run/vm/`:

| Script | Runs in | Purpose |
|---|---|---|
| `ctl.sh` | Cloud Shell | dryrun / create / ssh / status / start / stop / log / smokelog / bucket / alert / delete |
| `preflight.sh` | VM | sanity checks before any GPU time: GPU, both environments, bucket, model cache, Hugging Face, inputs, disk |
| `setup_vm.sh` | VM | Python, both environments, rclone, idle shutdown, Hugging Face login |
| `smoke_test.sh` | VM | the Phase 3 checks + a measured time and cost estimate |
| `run_all.sh` | VM | the full run; starts on every boot and resumes |
| `log_sync.sh` | VM | copies a running log to the bucket every 2 minutes (used by the two above) |
| `reference.py` + `reference_*.npz` | VM | compare cloud vectors with the Mac's |

---

## Before you start: two quota rows to check (1 min)

In **Intersectionality-compute → IAM & Admin → Quotas & System Limits**, filter by `us-central1` and confirm:
- **Preemptible NVIDIA A100 80GB GPUs** = 1 (done)
- **Persistent Disk SSD (GB)** ≥ 250 (the boot disk; new projects usually get 500+). If it's lower, request 500.
- **Local SSD (GB)** and, if listed, **Preemptible Local SSD (GB)** ≥ 375 (the A100 machine comes with one; the dry run uses one too).

A2 machines need no CPU quota.

## A. Open Cloud Shell and get the code (once)

1. Open console.cloud.google.com and select project **Intersectionality-compute**.
2. Click the **`>_`** icon (top right). A terminal opens at the bottom. Click **Authorize** if asked.
3. Run:
   ```bash
   git clone -b final-run https://github.com/kai-lia/intersectional-bias.git
   cd ~/intersectional-bias/final_run/vm
   ```
   If you've cloned before, update instead: `cd ~/intersectional-bias && git pull && cd final_run/vm`.

## B0. Dry run first (~10 min, a few cents)

Before any GPU is touched, this proves the Google Cloud side works: it creates a tiny CPU spot VM with
**the same flags, image and service account** as the A100 VM, runs the real `setup_vm.sh` on it (minus the
GPU parts), checks the bucket with the VM's own identity, creates the stop alert, and stops the VM with the
real stop flag. Nothing from this run goes near the A100 or the real data.

```bash
bash ctl.sh dryrun you@example.com
```
- Success ends with **`DRY RUN PASSED`**. Within ~15 minutes an email titled *final-run VM stopped: final-run-dryrun*
  should arrive. That email is the proof that the stop alert works.
- Then: `bash ctl.sh dryrun-cleanup` (deletes the dry-run VM and its alert).
- Any `DRY RUN FAILED` or error: paste the output to Claude. Rerun after `bash ctl.sh dryrun-cleanup`.

## B. Create the VM

```bash
bash ctl.sh create
```
- It uses Google's Deep Learning VM image (NVIDIA driver 580 preinstalled) and tries zones a and c; b and f have no A100 80GB.
- Success ends with `Created final-run-a100 in us-central1-x`. **The GPU is now billing.**
- `No zone had a spot A100 80GB available`: no capacity right now. Wait 15–30 min and run it again.
- Any other error: paste it to Claude.

2. **Create the stop alert** (once; same address as the dry run):
   ```bash
   bash ctl.sh alert you@example.com
   ```
   From now on you get an email about 10 minutes after the A100 VM stops, whatever the reason, and another when it
   runs again. A stop you did yourself also emails you; that's a useful confirmation that the GPU is off.

## C. Set up the VM (~15 min, once)

1. Log in to the VM from Cloud Shell:
   ```bash
   bash ctl.sh ssh
   ```
   The first time, it asks to create an SSH key. Press **Enter** at every question (no passphrase).
   If it asks *"install Nvidia driver?"*, answer **y**.
   **Pasting:** one line at a time, and only after the prompt shows `@final-run-a100` (Cloud Shell has its
   own tmux, so `tmux new` typed there fails with *nested with care*). A long line can wrap into two
   commands; type it instead. Never paste a transcript: every line of it runs as a command.
2. On the VM (the prompt changes to `…@final-run-a100`), inside `tmux` so a dropped connection doesn't kill the setup:
   ```bash
   tmux new -s setup
   git clone -b final-run https://github.com/kai-lia/intersectional-bias.git
   cd ~/intersectional-bias/final_run
   bash vm/setup_vm.sh
   ```
   If the connection drops: `bash ctl.sh ssh`, then `tmux attach`.
3. At step 7/7 it asks for the Hugging Face token. Paste the **final-run-vm** token from your
   password manager (nothing shows while you paste), press Enter, and answer **n** to the git question.
4. It must end with three `access ok:` lines and **`SETUP COMPLETE`**. If not, paste the error to Claude.

## D. Smoke test (~1.5–2 h, ~$4–5)

1. On the VM, inside `tmux` so the test survives a dropped connection (skip the first line if you are
   still in the `tmux` session from C):
   ```bash
   tmux new -s smoke
   cd ~/intersectional-bias/final_run
   bash vm/smoke_test.sh
   ```
   It keeps its output in `~/smoke.log` and copies that to the bucket every 2 minutes, so from Cloud Shell
   `bash ctl.sh smokelog` shows how far it got, whether the VM is running, stopped or preempted.
   It writes to `outputs_smoke/` on the VM and `gs://intersectionality-data/smoke/` in the bucket, never to the real
   `final_run/` prefix. Delete the smoke data afterwards from Cloud Shell: `gcloud storage rm -r gs://intersectionality-data/smoke`.
2. You can watch it, or detach with **Ctrl-b, then d**, and close Cloud Shell. While the GPU is
   busy, the idle shutdown won't fire. To come back: `bash ctl.sh ssh`, then `tmux attach`.
3. When it prints **`DONE`**, print the summary and paste it to Claude:
   ```bash
   grep -E "====|PASS|FAIL|preflight passed|min cosine|prompts in|done in|hit cap|token ids|disagreement|already done|activation files|GPU-hours|FULL RUN|vLLM version|Error|Traceback" ~/smoke.log
   ```
4. **Stop the VM** so the GPU stops billing (it would also shut itself down 30 min after you log out):
   ```bash
   exit                 # leave tmux
   exit                 # leave the VM
   bash ctl.sh stop     # in Cloud Shell
   ```

Claude then reviews the numbers, pins the vLLM version, and updates the budget for a go/no-go decision.

## E. Full run (~46 GPU-hours ≈ 2 days measured in the smoke test; longer with preemptions)

Only after the smoke test is approved.

1. Start the VM and log in:
   ```bash
   bash ctl.sh start
   bash ctl.sh ssh
   ```
2. On the VM, get the approved code, clear the smoke files, and start:
   ```bash
   cd ~/intersectional-bias && git pull && cd final_run
   rm -rf outputs_smoke
   bash vm/run_all.sh --install
   sudo systemctl start final-run
   exit
   ```
3. It now runs on its own. It writes only to `gs://intersectionality-data/final_run/`.

**Where everything is stored.** Each finished template-wording ("group": one template × one wording × one model,
12,254 prompts) is moved to the bucket as soon as it is done, then its `.done` marker is written; a restart skips every
group that has a marker. `m` is `granite`, `llama` or `mistral`; `PP` is the template 00–36; `W` the wording 0–3.

| Path under `gs://intersectionality-data/final_run/` | What | Count at the end |
|---|---|---|
| `activations/model={m}/layer={LL}/p{PP}_w{W}.npz` | last-token activations, one layer of one group, ~100 MB | 15,392 files, ~1.55 TB |
| `readout/model={m}/p{PP}_w{W}.npz` | P(yes) / P(no) per prompt | 444 |
| `done/model={m}/p{PP}_w{W}.done` | extraction resume marker (prompt count, token-id digest, timing, host) | 444 |
| `generations/model={m}/p{PP}_w{W}.jsonl.gz` | answer + reasoning per prompt | 444 |
| `done_generate/model={m}/p{PP}_w{W}.done` | generation resume marker | 444 |
| `run_info/model={m}.json`, `run_info/generate_model={m}.json` | versions, model revisions, input hashes | 6 |
| `checks/disagreements_model={m}.csv` | rows where the generated yes/no disagrees with P(yes) | up to 3 |
| `logs/run_*.log`, `logs/run_*.log.check` | the full-run log of each boot, uploaded when the run ends or fails | 1 per boot |
| `_preflight/` | probe files written and deleted by every preflight; normally empty | 0 |

Progress at any time: `bash ctl.sh bucket` (total size), or count markers from Cloud Shell:
```bash
gcloud storage ls "gs://intersectionality-data/final_run/done/**" | wc -l
gcloud storage ls "gs://intersectionality-data/final_run/done_generate/**" | wc -l
```
Both reach 444 when the run is complete.

The smoke test uses the same layout under `gs://intersectionality-data/smoke/` (only template 00 wording 0, plus template 01
wordings 0–1 of granite for the resume test). On the VM, the full run works in `~/intersectional-bias/final_run/outputs/`
(files leave for the bucket as they finish), the smoke test in `outputs_smoke/`, logs in `~/run_logs/` and `~/smoke.log`,
and the downloaded models in `~/.cache/huggingface/`. Nothing on the VM is needed once it is in the bucket.

**When the stop-alert email arrives (or once a day anyway), from Cloud Shell** (`cd ~/intersectional-bias/final_run/vm` first):

| You see | Do |
|---|---|
| `bash ctl.sh status` shows `RUNNING` | fine. `bash ctl.sh log` shows progress and the ETA |
| the stop-alert email arrived, or `status` shows `TERMINATED`, and the run isn't finished | `bash ctl.sh log` to see where it stopped (works while stopped), then `bash ctl.sh start`. It resumes by itself |
| `start` fails with a capacity error | try again later. Nothing is lost |
| the log ends with `=== FINISHED` | done. Go to F |
| the log ends with `exit status` ≠ 0 and no FINISHED | **don't restart.** Send Claude the log: `bash ctl.sh log` |

`bash ctl.sh log` reads the bucket copy of the log, which the VM refreshes every 2 minutes, so it works whether the
VM is running or stopped. `bash ctl.sh bucket` shows how much has been uploaded (~1.55 TB at the end).

## F. After the run

1. In Cloud Shell, check the final log:
   ```bash
   gcloud storage cat "$(gcloud storage ls gs://intersectionality-data/final_run/logs/ | grep '\.check$' | tail -1)"
   ```
   It should end with `ALL CHECKS PASSED`.
2. **Keep the VM stopped, don't delete it yet.** The base-model step (Phase 4b) reuses its setup.
   A stopped VM costs ~$0.85/day for the disk.
3. When all GPU work is over:
   ```bash
   bash ctl.sh delete    # removes the VM, its disk and the stop alert; the bucket is untouched
   bash ctl.sh status    # both lists must be empty
   ```
4. Then revoke the VM's token on Hugging Face (Settings → Access Tokens; the one entered during setup is named `Intersectionality-Run-VM`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Quota 'GPUS_ALL_REGIONS' exceeded` | another GPU VM exists: `bash ctl.sh status`, delete the extra one |
| `PERMISSION_DENIED` / 403 on the bucket in setup step 5 | the `code-runner` grant on `intersectionality-data` is missing (Storage Object Admin) |
| Hugging Face `401` / `403` in setup step 7 | token wrong or access not accepted; rerun `~/venvs/extract/bin/hf auth login` |
| `CUDA out of memory` during extraction | send Claude the log; the fix is `--batch-size 64` |
| The VM shut down while you were working on it | the idle shutdown: you were logged out and no GPU job ran for ~30 min. `bash ctl.sh start` |
| SSH hangs right after `create` / `start` | the VM is still booting; wait 1–2 min and retry |
| `create` fails with an error about `discard-local-ssds` or `max-run-duration` | paste it to Claude; the flags for spot + time limit + local SSD changed and need adjusting |
| `create` says the image family was not found and the fallback also fails | paste it to Claude; the VM can be built from a plain Ubuntu image with the driver installed by `setup_vm.sh` |
| `setup_vm.sh` prints `WARNING: transformers … vs …` | the pins in `requirements-generate.txt` did not take; paste the setup output to Claude |
| `smoke_test.sh` stops at step 2 with a vLLM error | paste it to Claude; vLLM runs for the first time here |
| vLLM fails at start with `No such file or directory: 'ninja'` | FlashInfer trying to compile a sampling kernel that greedy decoding never uses. The scripts set `VLLM_USE_FLASHINFER_SAMPLER=0`; when running `generate.py` by hand, `export` it first |
| after a preemption something is oddly broken: `pip: No such file`, `uv` prints nothing, `JSONDecodeError: Expecting value` when a model loads, vLLM cannot load a compiled graph | a file that was being written when the power was cut is empty or cut short. `setup_vm.sh` and `preflight.sh` catch the common cases and say what to delete; otherwise delete what was being loaded (`rm -rf ~/venvs` and rerun setup; the model's `~/.cache/huggingface/hub/models--*` folder; `~/.cache/vllm`) and rerun |
| `PREFLIGHT FAILED: …` | the message names the broken piece (GPU, environment, bucket, token, inputs, disk); fix that and rerun. Nothing was spent |
| `generate.py` exits with `prompt token ids differ from extract.py's` | the two environments tokenize differently; nothing was generated. Paste it to Claude |
