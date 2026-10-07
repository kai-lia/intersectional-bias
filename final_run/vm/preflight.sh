#!/usr/bin/env bash
# Sanity checks before any GPU time is spent.  Exits non-zero on the first failure.
#   bash vm/preflight.sh gcs:intersectionality-data/final_run
# run_all.sh and smoke_test.sh call this first.
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE="${1:?usage: preflight.sh REMOTE}"
EX=~/venvs/extract/bin/python
GEN=~/venvs/generate/bin/python
fail() { echo "PREFLIGHT FAILED: $*"; exit 1; }
ok() { echo "  ok  $*"; }
echo "== preflight ($(date -u +%FT%TZ)) for $REMOTE"

# 1. the GPU we planned for
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || fail "nvidia-smi not working"
[[ -n "$gpu" ]] || fail "no GPU visible"
mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[[ "$gpu" == *A100* && "$mem_mib" -ge 79000 ]] || echo "  WARNING: GPU is '$gpu' (${mem_mib} MiB), not the A100 80GB the run was planned on (consistency!)"
ok "GPU: $gpu, ${mem_mib} MiB"

# 2. both environments can use it
$EX -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null || fail "extract env cannot use the GPU"
$GEN -c "import torch, vllm; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null || fail "generate env cannot use the GPU (or vllm does not import)"
tf_a=$($EX -c "import transformers; print(transformers.__version__)"); tf_b=$($GEN -c "import transformers; print(transformers.__version__)")
[[ "$tf_a" == "$tf_b" ]] || fail "transformers differs between environments ($tf_a vs $tf_b); token ids would not match"
ok "CUDA usable in both environments; transformers $tf_a in both"

# 3. the bucket: write, read back, delete
probe="$REMOTE/_preflight/$(hostname)_$(date +%s).txt"
echo "preflight $(date -u)" > /tmp/preflight.txt
rclone copyto /tmp/preflight.txt "$probe" 2>/dev/null || fail "cannot write to $REMOTE (rclone remote 'gcs' configured? code-runner has Storage Object Admin on the bucket?)"
rclone cat "$probe" >/dev/null 2>&1 || fail "cannot read back from $REMOTE"
rclone deletefile "$probe" 2>/dev/null || fail "cannot delete in $REMOTE"
ok "bucket write/read/delete"

# 4. Hugging Face: logged in, pinned models reachable (cache first, network only if needed)
$EX - <<'EOF' || fail "Hugging Face token or model access"
from huggingface_hub import whoami, hf_hub_download
import extract as fr
whoami()
for m, repo in fr.MODEL_IDS.items():
    try:
        hf_hub_download(repo, "config.json", revision=fr.MODEL_REVISIONS[m], local_files_only=True)
    except Exception:
        hf_hub_download(repo, "config.json", revision=fr.MODEL_REVISIONS[m])
EOF
ok "Hugging Face token valid; all three pinned models reachable"

# 5. inputs untouched, code committed and current
(cd inputs && sha256sum -c --quiet SHA256SUMS) || fail "inputs/ do not match SHA256SUMS"
[[ -z "$(git status --porcelain -- . ':!outputs*')" ]] || fail "uncommitted changes in final_run/ (run_info would record the wrong commit)"
git fetch -q origin final-run 2>/dev/null && [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/final-run)" ]] \
  || echo "  WARNING: HEAD is not origin/final-run (git pull?)"
ok "inputs match checksums; working tree clean at $(git rev-parse --short HEAD)"

# 6. room on the boot disk
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[[ "$free_gb" -ge 60 ]] || fail "only ${free_gb} GB free on the boot disk (need >= 60 for models and in-flight groups)"
ok "${free_gb} GB free on disk"

echo "== preflight passed"
