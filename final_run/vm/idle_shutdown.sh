#!/usr/bin/env bash
# Run by cron every 10 minutes (installed by setup_vm.sh).  Shuts the VM down
# after 3 checks in a row (~30 min) with no process on the GPU and nobody
# logged in over SSH, so a finished or forgotten VM stops billing for the GPU.
# Never fires in the first 30 minutes after boot, and never while
# /run/final_run_busy exists (run_all.sh and smoke_test.sh create it; /run is cleared at boot).
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
STATE=/var/tmp/idle_count
BUSY=/run/final_run_busy
uptime_s=$(cut -d. -f1 /proc/uptime)
if [ "$uptime_s" -lt 1800 ] || [ -f "$BUSY" ]; then echo 0 > "$STATE"; exit 0; fi
gpu_jobs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
logged_in=$(who | grep -c . || true)
if [ "$gpu_jobs" -eq 0 ] && [ "$logged_in" -eq 0 ]; then
  n=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
else
  n=0
fi
echo "$n" > "$STATE"
if [ "$n" -ge 3 ]; then
  logger "idle_shutdown: no GPU job and no SSH session for ~30 min, shutting down"
  /sbin/shutdown -h now
fi
