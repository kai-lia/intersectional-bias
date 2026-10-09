#!/usr/bin/env bash
# Cloud smoke test: ~1.5-2 GPU-hours (~$4-5 on spot).  Run on the VM inside tmux:
#   cd ~/intersectional-bias/final_run
#   bash vm/smoke_test.sh
# It keeps its output in ~/smoke.log and copies that to the bucket every 2 minutes
# (gs://intersectionality-data/smoke/logs/), so `bash ctl.sh smokelog` in Cloud Shell
# shows how far it got, even after a preemption.
#
#   1  each model reproduces the Mac's vectors (cosine >= 0.999)
#   2  uncapped length sample, 1000 prompts per model (does 512 tokens fit?)
#   3  one full template-wording per model: extraction + upload to the bucket
#   4  the same template-wording: generation + upload
#   5  check.py on it (token ids match, generated yes/no agrees with P(yes))
#   6  resume: hard-kill an extraction mid-run, restart, confirm it continues
#   7  measured speed -> full-run time and cost estimate
# Writes to outputs_smoke/ locally and gs://intersectionality-data/smoke/
# (never to the real final_run/ prefix).
set -euo pipefail
SELF=$(realpath "$0")
cd "$(dirname "$0")/.."
if [[ -t 1 ]]; then                                  # started from a terminal: keep a copy of everything in ~/smoke.log
  [[ -f ~/smoke.log ]] && mv ~/smoke.log ~/smoke_"$(date -u +%Y%m%d_%H%M%S)".log
  bash "$SELF" "$@" 2>&1 | tee ~/smoke.log; exit "${PIPESTATUS[0]}"
fi
EX=~/venvs/extract/bin/python
GEN=~/venvs/generate/bin/python
OUT=outputs_smoke
REMOTE=gcs:intersectionality-data/smoke
MODELS=(granite llama mistral)
export VLLM_USE_FLASHINFER_SAMPLER=0                 # see run_all.sh: no on-the-fly kernel compile for a kernel greedy decoding never uses
LOG=~/smoke.log                                      # written by the tee above (or by `... | tee ~/smoke.log`)
LOG_REMOTE="$REMOTE/logs/smoke_$(date -u +%Y%m%d_%H%M%S).log"
sudo touch /run/final_run_busy                       # idle shutdown stays off while this runs
bash vm/log_sync.sh "$LOG" "$LOG_REMOTE" &           # bucket copy of the log every 2 min -> `bash ctl.sh smokelog`
SYNC=$!
trap 'kill $SYNC 2>/dev/null; sleep 2; bash vm/log_sync.sh "$LOG" "$LOG_REMOTE" --once; sudo rm -f /run/final_run_busy' EXIT
step() { echo; echo "==================== $* ($(date -u +%H:%M) UTC)"; }

step "0  preflight"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
git log --oneline -1
bash vm/preflight.sh "$REMOTE"

step "1  matches the Mac"
for m in "${MODELS[@]}"; do
  $EX extract.py --models "$m" --patterns 0 --wordings 0 --identities Black Asian Latina --out "$OUT/tiny"
  $EX vm/reference.py compare --out "$OUT/tiny" --model "$m" || echo "[$m] REFERENCE CHECK FAILED (continuing; Claude decides)"
done

step "2  length sample (1000 prompts per model, uncapped)"
$GEN generate.py --sample 1000 --out "$OUT/sample"

step "3  extraction: pattern 0, wording 0, all 12,254 prompts, every model -> bucket"
$EX extract.py --patterns 0 --wordings 0 --out "$OUT/full" --remote "$REMOTE"

step "4  generation: same template-wording, every model -> bucket"
$GEN generate.py --patterns 0 --wordings 0 --out "$OUT/full" --remote "$REMOTE"

step "5  checks (expect MISSING counts -- this is 1 of 148 groups; look at the token-id and agreement lines)"
rclone copy "$REMOTE" "$OUT/pulled" --include "readout/**" --include "generations/**" \
  --include "run_info/**" --include "done*/**"
$EX check.py --out "$OUT/pulled" --remote "$REMOTE" || true

step "6  resume after a hard kill (granite, pattern 1, wordings 0-1)"
RES=(--models granite --patterns 1 --wordings 0 1 --out "$OUT/resume" --remote "$REMOTE")
$EX extract.py "${RES[@]}" > "$OUT/resume_run1.log" 2>&1 &
pid=$!
until [[ -f "$OUT/resume/done/model=granite/p01_w0.done" ]]; do
  kill -0 "$pid" 2>/dev/null || { echo "first run exited early:"; tail "$OUT/resume_run1.log"; exit 1; }
  sleep 10
done
sleep 15                                   # first group's upload is in flight, second is computing
kill -9 "$pid" 2>/dev/null || echo "(nothing to kill: both groups were already in the bucket from an earlier run, so this is only a re-check)"
wait "$pid" 2>/dev/null || true
pkill -9 -f "rclone move" || true; sleep 2            # a real preemption kills the upload too
echo "killed mid-run (like a spot interruption); files left locally: $(find "$OUT/resume" -name '*.npz' | wc -l)"
echo "restarting the same command"
$EX extract.py "${RES[@]}" > "$OUT/resume_run2.log" 2>&1
grep -E "already done|done in" "$OUT/resume_run2.log"
n=$(rclone lsf -R "$REMOTE/activations/model=granite" | grep -c "p01_w[01].npz" || true)
echo "activation files in bucket for the 2 groups: $n / 80  $([[ $n == 80 ]] && echo PASS || echo FAIL)"

step "7  measured speed -> full-run estimate"
rclone copy "$REMOTE" "$OUT/pulled" --include "done*/**"
$EX - "$OUT/pulled" <<'EOF'
import json, sys
from pathlib import Path
root, groups, price = Path(sys.argv[1]), 148, 2.50
total = 0.0
for m in ("granite", "llama", "mistral"):
    ex = json.loads((root / f"done/model={m}/p00_w0.done").read_text())
    ge = json.loads((root / f"done_generate/model={m}/p00_w0.done").read_text())
    h = (ex["seconds"] + ge["seconds"]) * groups / 3600
    total += h
    print(f"{m:8} extract {ex['seconds']:6.0f}s/group  generate {ge['seconds']:6.0f}s/group  "
          f"hit cap {ge['hit_cap']}/{ge['n_prompts']}  -> {h:5.1f} GPU-hours for 148 groups")
print(f"\nFULL RUN (instruct): ~{total:.0f} GPU-hours = ~{total / 24:.1f} days on 1 GPU, "
      f"~${total * price:.0f} at ${price:.2f}/h spot (plus restarts)")
EOF

step "DONE"
~/venvs/generate/bin/python -c "import vllm; print('vLLM version to pin:', vllm.__version__)"
echo "Send the whole output (~/smoke.log) to Claude.  Smoke data in the bucket can be deleted"
echo "afterwards with:  rclone purge $REMOTE"
