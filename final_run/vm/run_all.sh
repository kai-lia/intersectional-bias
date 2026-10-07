#!/usr/bin/env bash
# Full instruct run: extract.py -> generate.py -> check.py, all to the bucket.
#
#   bash vm/run_all.sh --install   once: register it to start on every boot
#   sudo systemctl start final-run start it now (then you can log out)
#
# Every boot (after a spot interruption or the 48 h limit) it resumes where it
# stopped.  When it finishes, or if a step fails, it uploads its log to
# gs://intersectionality-data/final_run/logs/ and shuts the VM down.
#   touch ~/HOLD    boots do nothing (for maintenance); rm ~/HOLD to resume
#   ~/FINISHED      written when every check passed; later boots do nothing
set -uo pipefail
cd "$(dirname "$0")/.."
FINAL_RUN="$PWD"
REMOTE="gcs:intersectionality-data/final_run"

if [[ "${1:-}" == "--install" ]]; then
  sudo tee /etc/systemd/system/final-run.service >/dev/null <<EOF
[Unit]
Description=final_run full pipeline (extract -> generate -> check)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$USER
Environment=HOME=$HOME
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/bin/bash $FINAL_RUN/vm/run_all.sh

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable final-run
  echo "Installed. Start now with: sudo systemctl start final-run"
  exit 0
fi

sudo rm -f /var/tmp/final_run_busy
[[ -f ~/FINISHED ]] && { echo "~/FINISHED exists: nothing to do"; exit 0; }
[[ -f ~/HOLD ]] && { echo "~/HOLD exists: not starting"; exit 0; }
sudo touch /var/tmp/final_run_busy                   # idle shutdown stays off while the run is active
mkdir -p ~/run_logs
LOG=~/run_logs/run_$(date -u +%Y%m%d_%H%M%S).log
exec >>"$LOG" 2>&1
echo "=== boot $(date -u) | $(git -C "$FINAL_RUN" log --oneline -1)"
for i in $(seq 1 30); do nvidia-smi >/dev/null 2>&1 && break; sleep 20; done

status=0
echo "=== preflight $(date -u)"
bash vm/preflight.sh "$REMOTE" || status=$?
if [[ $status -eq 0 ]]; then
  echo "=== extract $(date -u)"
  ~/venvs/extract/bin/python extract.py --remote "$REMOTE" || status=$?
fi
if [[ $status -eq 0 ]]; then
  echo "=== generate $(date -u)"
  ~/venvs/generate/bin/python generate.py --remote "$REMOTE" || status=$?
fi
if [[ $status -eq 0 ]]; then
  echo "=== check $(date -u)"
  rclone copy "$REMOTE" outputs --include "readout/**" --include "generations/**" \
    --include "run_info/**" --include "done*/**"
  ~/venvs/extract/bin/python check.py --remote "$REMOTE" | tee "$LOG.check"
  grep -q "ALL CHECKS PASSED" "$LOG.check" || status=1
  rclone copy outputs/checks "$REMOTE/checks"
fi

echo "=== exit status $status $(date -u)"
[[ $status -eq 0 ]] && touch ~/FINISHED && echo "=== FINISHED"
rclone copy ~/run_logs "$REMOTE/logs"
sudo rm -f /var/tmp/final_run_busy
sudo shutdown -h now
