#!/usr/bin/env bash
# Copy a growing log file to the bucket every 2 minutes, so its progress can be read from
# Cloud Shell (bash ctl.sh log / smokelog) while the VM runs and after it stops or is preempted.
#   bash vm/log_sync.sh LOCAL gcs:bucket/path/name.log &      loop; kill it when the job ends
#   bash vm/log_sync.sh LOCAL gcs:bucket/path/name.log --once  one copy, now
# The file is snapshotted first so rclone never uploads a file that changes under it.
local_log=$1 remote=$2
snap=$(mktemp)
while true; do
  cp "$local_log" "$snap" 2>/dev/null && rclone copyto "$snap" "$remote" 2>/dev/null
  [[ "${3:-}" == --once ]] && break
  sleep 120
done
rm -f "$snap"
