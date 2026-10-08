#!/usr/bin/env bash
# Control the final-run GPU VM from Cloud Shell.
#
#   bash ctl.sh dryrun EMAIL   ~15 min, a few cents: a small CPU spot VM created with the SAME flags, image
#                              and service account as the A100 VM runs the real setup script (minus GPU),
#                              the bucket checks and the stop alert, then is stopped with the real stop flag;
#                              rerunning it resumes (restarts the VM if it was preempted)
#   bash ctl.sh dryrun-cleanup delete the dry-run VM and its alert (after the alert email arrived)
#   bash ctl.sh create         create the spot A100 VM (tries each us-central1 zone that has A100 80GB)
#   bash ctl.sh alert EMAIL    email EMAIL whenever the A100 VM stops (preemption, 48 h limit, finished, by hand)
#   bash ctl.sh ssh            open a terminal on the VM
#   bash ctl.sh status         VM state + what is still billing
#   bash ctl.sh start          start it again (after a spot interruption or the 48 h limit)
#   bash ctl.sh stop           stop it (GPU billing stops; the boot disk is kept, ~$0.85/day)
#   bash ctl.sh log            last lines of the full-run log
#   bash ctl.sh bucket         how much is in the bucket
#   bash ctl.sh delete         delete the A100 VM, its disk and its stop alert (end of the run)
set -euo pipefail

PROJECT="intersectionality-compute"
NAME="final-run-a100"
DRY_NAME="final-run-dryrun"
SA="code-runner@intersectionality-compute.iam.gserviceaccount.com"
BUCKET="gs://intersectionality-data"
REPO="https://github.com/kai-lia/intersectional-bias.git"
MAX_RUN="48h"          # each start of the A100 VM may run at most this long, then it stops itself
ZONES=(us-central1-a us-central1-b us-central1-c)     # the us-central1 zones with A2 Ultra (A100 80GB)
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"    # Google Deep Learning VM: driver 580 preinstalled

gcloud config set project "$PROJECT" --quiet >/dev/null 2>&1

zone_of() {            # VMNAME -> zone, or exit
  local z
  z=$(gcloud compute instances list --filter="name=$1" --format="value(zone.basename())")
  [[ -n "$z" ]] || { echo "No VM named $1 exists." >&2; exit 1; }
  echo "$z"
}
exists() { gcloud compute instances list --filter="name=$1" --format="value(name)" | grep -q .; }

create_vm() {          # VMNAME MACHINE-TYPE DISK-GB MAX-RUN [extra gcloud flags]  -- same flags for the dry run and the A100
  local name=$1 machine=$2 disk=$3 maxrun=$4; shift 4
  if ! gcloud compute images describe-from-family "$IMAGE_FAMILY" --project="$IMAGE_PROJECT" >/dev/null 2>&1; then
    echo "Image family $IMAGE_FAMILY not found; picking the newest common-cu* family instead"
    IMAGE_FAMILY=$(gcloud compute images list --project="$IMAGE_PROJECT" --no-standard-images \
                     --filter="family~^common-cu1" --format="value(family)" | grep -v arm | sort -uV | tail -1)
    [[ -n "$IMAGE_FAMILY" ]] || { echo "No Deep Learning VM image family found. Ask Claude."; return 1; }
  fi
  echo "Image family: $IMAGE_FAMILY"
  for Z in "${ZONES[@]}"; do
    echo "--- trying $Z"
    if gcloud compute instances create "$name" --zone="$Z" --machine-type="$machine" \
         --provisioning-model=SPOT --instance-termination-action=STOP \
         --max-run-duration="$maxrun" --discard-local-ssds-at-termination-timestamp=true \
         --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
         --boot-disk-size="${disk}GB" --boot-disk-type=pd-balanced \
         --service-account="$SA" --scopes=cloud-platform "$@"; then
      return 0
    fi
  done
  return 1
}

wait_ssh() {           # VMNAME ZONE: wait until SSH works (first call also creates your SSH key)
  local i
  for i in $(seq 1 30); do
    gcloud compute ssh "$1" --zone="$2" --quiet --command=true >/dev/null 2>&1 && return 0
    sleep 10
  done
  echo "SSH to $1 did not come up in 5 minutes" >&2; return 1
}

alert_title() { echo "final-run VM stopped: $1"; }

make_alert() {         # VMNAME EMAIL: email when the VM reports no uptime for 10 minutes
  local vm=$1 email=$2 title channel
  title=$(alert_title "$vm")
  gcloud services enable monitoring.googleapis.com --quiet
  if gcloud monitoring policies list --filter="displayName='$title'" --format="value(name)" | grep -q .; then
    echo "alert '$title' already exists"; return 0
  fi
  channel=$(gcloud beta monitoring channels list --filter="type=email AND labels.email_address=$email" \
              --format="value(name)" | head -1)
  [[ -n "$channel" ]] || channel=$(gcloud beta monitoring channels create --display-name="final-run email" \
      --type=email --channel-labels="email_address=$email" --format="value(name)")
  cat > /tmp/final_run_alert.json <<JSONEOF
{
  "displayName": "$title",
  "combiner": "OR",
  "conditions": [{
    "displayName": "$vm reported no uptime for 10 minutes",
    "conditionAbsent": {
      "filter": "resource.type = \"gce_instance\" AND metric.type = \"compute.googleapis.com/instance/uptime\" AND metric.labels.instance_name = \"$vm\"",
      "duration": "600s",
      "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_RATE"}]
    }
  }],
  "notificationChannels": ["$channel"],
  "documentation": {
    "mimeType": "text/markdown",
    "content": "The VM **$vm** has stopped (spot interruption, the run-time limit, the run finished, or you stopped it).\n\nIn Cloud Shell: cd ~/intersectional-bias/final_run/vm && bash ctl.sh status, then bash ctl.sh log. If the log does not end with === FINISHED and you did not stop it yourself: bash ctl.sh start (the run resumes by itself)."
  }
}
JSONEOF
  gcloud monitoring policies create --policy-from-file=/tmp/final_run_alert.json >/dev/null
  echo "Alert created: $email gets an email ~10 minutes after $vm stops, for any reason (and one when it runs again)."
}

delete_alert() {       # VMNAME
  local p
  for p in $(gcloud monitoring policies list --filter="displayName='$(alert_title "$1")'" --format="value(name)"); do
    gcloud monitoring policies delete "$p" --quiet && echo "deleted the stop alert for $1"
  done
}

case "${1:-}" in
dryrun)
  EMAIL="${2:?usage: bash ctl.sh dryrun you@example.com}"
  if exists "$DRY_NAME"; then
    Z=$(zone_of "$DRY_NAME")
    ST=$(gcloud compute instances describe "$DRY_NAME" --zone="$Z" --format="value(status)")
    echo "== 1/5 $DRY_NAME already exists ($ST): resuming. What happened to it last:"
    gcloud compute operations list --filter="targetLink~/$DRY_NAME" --sort-by=~insertTime --limit=4 \
      --format="table(operationType,insertTime.date('%H:%M:%S'),status)"
    [[ "$ST" == RUNNING ]] || gcloud compute instances start "$DRY_NAME" --zone="$Z"
  else
    echo "== 1/5 creating a CPU spot VM with the A100 VM's flags (n2-standard-4 + 1 local SSD, 1 h limit)"
    create_vm "$DRY_NAME" n2-standard-4 100 1h --local-ssd=interface=NVME \
      || { echo "DRY RUN FAILED at VM creation. Paste the error above to Claude."; exit 1; }
    Z=$(zone_of "$DRY_NAME")
  fi
  echo "== 2/5 waiting for SSH"
  wait_ssh "$DRY_NAME" "$Z"
  echo "== 3/5 stop alert for $DRY_NAME (created now so it sees the VM running before the stop)"
  make_alert "$DRY_NAME" "$EMAIL"
  echo "== 4/5 on the VM: identity, sudo, clone, real setup script (no GPU), preflight (bucket, inputs, disk)"
  gcloud compute ssh "$DRY_NAME" --zone="$Z" --quiet --command='
    set -e
    echo "service account on the VM: $(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email)"
    sudo -n true && echo "passwordless sudo: ok"
    [[ -d intersectional-bias ]] || git clone -q -b final-run '"$REPO"'
    cd intersectional-bias/final_run && git pull -q
    echo "memory: $(free -g | awk "/Mem:/ {print \$2}") GB; last kernel OOM/shutdown lines, if any:"; sudo dmesg 2>/dev/null | grep -iE "out of memory|killed process|shutdown" | tail -3 || true
    DRYRUN=1 bash vm/setup_vm.sh
    DRYRUN=1 bash vm/preflight.sh gcs:intersectionality-data/smoke/_dryrun
    ls -la /usr/local/bin/idle_shutdown.sh /etc/cron.d/idle-shutdown /sbin/shutdown
    echo "ON-VM DRY RUN PASSED"
  ' || { gcloud compute instances stop "$DRY_NAME" --zone="$Z" --discard-local-ssd=true --quiet >/dev/null 2>&1 || true
         echo "DRY RUN FAILED on the VM (VM stopped). Paste the output above to Claude. (bash ctl.sh dryrun EMAIL again resumes where it stopped.)"; exit 1; }
  echo "== 5/5 stopping the VM with the real stop flag"
  gcloud compute instances stop "$DRY_NAME" --zone="$Z" --discard-local-ssd=true
  echo
  echo "DRY RUN PASSED. Within ~15 minutes, $EMAIL should get an email from Google Cloud Monitoring:"
  echo "  '$(alert_title "$DRY_NAME")'"
  echo "When it has arrived (or after 20 minutes if it has not -- tell Claude), run: bash ctl.sh dryrun-cleanup"
  ;;
dryrun-cleanup)
  if exists "$DRY_NAME"; then
    Z=$(zone_of "$DRY_NAME"); gcloud compute instances delete "$DRY_NAME" --zone="$Z" --quiet
  fi
  delete_alert "$DRY_NAME"
  echo "== anything left?"; gcloud compute instances list; gcloud compute disks list
  ;;
create)
  exists "$NAME" && { echo "$NAME already exists (zone $(zone_of "$NAME")). Use 'start', or 'delete' first."; exit 1; }
  if create_vm "$NAME" a2-ultragpu-1g 250 "$MAX_RUN" --metadata=install-nvidia-driver=True; then
    echo; echo "Created $NAME in $(zone_of "$NAME"). The GPU is billing from now on. Wait ~2 minutes, then: bash ctl.sh ssh"
    exit 0
  fi
  echo; echo "No zone could create the VM. If every error above says ZONE_RESOURCE_POOL_EXHAUSTED or"
  echo "'does not have enough resources', no spot A100 80GB is free right now: try again in 15-30 min."
  echo "Any other error: paste it to Claude."; exit 1
  ;;
alert)  make_alert "$NAME" "${2:?usage: bash ctl.sh alert you@example.com}" ;;
ssh)    Z=$(zone_of "$NAME"); gcloud compute ssh "$NAME" --zone="$Z" ;;
start)  Z=$(zone_of "$NAME"); gcloud compute instances start "$NAME" --zone="$Z" ;;
stop)   Z=$(zone_of "$NAME"); gcloud compute instances stop "$NAME" --zone="$Z" --discard-local-ssd=true ;;
log)    Z=$(zone_of "$NAME"); gcloud compute ssh "$NAME" --zone="$Z" --command='tail -n 40 "$(ls -t ~/run_logs/*.log | head -1)"' ;;
bucket) gcloud storage du -s "$BUCKET/final_run" --readable-sizes ;;
status)
  echo "== VMs (RUNNING = billing)"; gcloud compute instances list --format="table(name,zone.basename(),status)"
  echo "== disks (billed while they exist)"; gcloud compute disks list --format="table(name,zone.basename(),sizeGb,status)"
  ;;
delete)
  read -r -p "Delete $NAME and its disk (models, venvs, local logs)? Bucket data is NOT touched. [y/N] " a
  [[ "$a" == y ]] || exit 0
  Z=$(zone_of "$NAME"); gcloud compute instances delete "$NAME" --zone="$Z" --quiet
  delete_alert "$NAME"
  echo "== anything left?"; gcloud compute instances list; gcloud compute disks list
  ;;
*) sed -n '2,16p' "$0"; exit 1 ;;
esac
