#!/usr/bin/env bash
# One-time setup on the GPU VM.  Run from the repo:
#   cd ~/intersectional-bias/final_run && bash vm/setup_vm.sh
# Safe to rerun.  Takes ~10-15 minutes (mostly pip installing torch and vLLM).
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE_TEST="gcs:intersectionality-data/smoke/_setup_test"

echo "== 1/7 GPU driver"
for i in $(seq 1 30); do nvidia-smi >/dev/null 2>&1 && break; echo "waiting for the NVIDIA driver ($i/30)..."; sleep 20; done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== 2/7 system tools"
# first boot: wait for the image's own package updates to release the apt lock
for i in $(seq 1 60); do
  sudo fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock >/dev/null 2>&1 || break
  [[ $i == 1 ]] && echo "waiting for the first-boot package updates to finish..."; sleep 10
done
sudo apt-get update -qq && sudo apt-get install -y -qq tmux unzip curl >/dev/null

echo "== 3/7 Python 3.11 (same as the Mac) via uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11

echo "== 4/7 environments (~10 min)"
[[ -x ~/venvs/extract/bin/python ]] || uv venv --python 3.11 --seed ~/venvs/extract
~/venvs/extract/bin/pip install -q -r requirements.txt
[[ -x ~/venvs/generate/bin/python ]] || uv venv --python 3.11 --seed ~/venvs/generate
~/venvs/generate/bin/pip install -q -r requirements-generate.txt
~/venvs/extract/bin/python -c "import torch, transformers; print('extract  env: torch', torch.__version__, '| transformers', transformers.__version__); assert torch.cuda.is_available(), 'extract env cannot see the GPU'"
~/venvs/generate/bin/python -c "import vllm, torch, transformers; print('generate env: vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__); assert torch.cuda.is_available(), 'generate env cannot see the GPU'"
TF_EXTRACT=$(~/venvs/extract/bin/python -c "import transformers; print(transformers.__version__)")
TF_GENERATE=$(~/venvs/generate/bin/python -c "import transformers; print(transformers.__version__)")
if [[ "$TF_EXTRACT" == "$TF_GENERATE" ]]; then
  echo "same transformers in both environments ($TF_EXTRACT): identical tokenization"
else
  echo "WARNING: transformers $TF_EXTRACT (extract) vs $TF_GENERATE (generate) -- the two steps tokenize"
  echo "         independently; the smoke test's check.py must show 'token ids extract = generate ... ok'"
fi

echo "== 5/7 rclone -> bucket (uses the VM's code-runner account, no key file)"
command -v rclone >/dev/null || curl -s https://rclone.org/install.sh | sudo bash >/dev/null
rclone listremotes | grep -qx "gcs:" || \
  rclone config create gcs "google cloud storage" env_auth=true bucket_policy_only=true no_check_bucket=true >/dev/null
echo "rclone test $(date)" > /tmp/rclone_test.txt
rclone copy /tmp/rclone_test.txt "$REMOTE_TEST/"
rclone ls "$REMOTE_TEST/"
rclone purge "$REMOTE_TEST"
echo "bucket upload/read/delete: ok"

echo "== 6/7 idle auto-shutdown (30 min with no GPU job and nobody logged in)"
sudo install -m 755 vm/idle_shutdown.sh /usr/local/bin/idle_shutdown.sh
echo "*/10 * * * * root /usr/local/bin/idle_shutdown.sh" | sudo tee /etc/cron.d/idle-shutdown >/dev/null
echo "installed"

echo "== 7/7 Hugging Face login"
if ~/venvs/extract/bin/python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
  echo "already logged in as $(~/venvs/extract/bin/python -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
else
  echo "Paste the final-run-vm READ token (it will not show while you paste). Answer n to the git-credential question."
  ~/venvs/extract/bin/hf auth login
fi
~/venvs/extract/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
import extract as fr
for m, repo in fr.MODEL_IDS.items():
    # downloading a small file fails on a gated repo without access
    hf_hub_download(repo, "config.json", revision=fr.MODEL_REVISIONS[m])
    print(f"access ok: {repo}@{fr.MODEL_REVISIONS[m][:8]}")
EOF

echo; echo "SETUP COMPLETE. Next: bash vm/smoke_test.sh (inside tmux)"
