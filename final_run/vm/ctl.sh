#!/usr/bin/env bash
# Control the final-run GPU VM from Cloud Shell.
#
#   bash ctl.sh create     create the spot A100 VM (tries each us-central1 zone that has A100 80GB)
#   bash ctl.sh ssh        open a terminal on the VM
#   bash ctl.sh status     VM state + what is still billing
#   bash ctl.sh start      start it again (after a spot interruption or the 48 h limit)
#   bash ctl.sh stop       stop it (GPU billing stops; the boot disk is kept, ~$0.85/day)
#   bash ctl.sh log        last lines of the full-run log
#   bash ctl.sh bucket     how much is in the bucket
#   bash ctl.sh alert EMAIL  email EMAIL whenever the VM stops (preemption, 48 h limit, finished, or by hand)
#   bash ctl.sh delete     delete the VM AND its disk and the stop alert (end of the run)
set -euo pipefail

PROJECT="intersectionality-compute"
NAME="final-run-a100"
SA="code-runner@intersectionality-compute.iam.gserviceaccount.com"
BUCKET="gs://intersectionality-data"
MAX_RUN="48h"          # each start may run at most this long, then the VM stops itself
ZONES=(us-central1-a us-central1-b us-central1-c)     # the us-central1 zones with A2 Ultra (A100 80GB)
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"    # Google Deep Learning VM: driver 580 preinstalled
ALERT_NAME="final-run VM stopped"

gcloud config set project "$PROJECT" --quiet >/dev/null 2>&1
zone() {
  local z
  z=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
  [[ -n "$z" ]] || { echo "No VM named $NAME exists. Run: bash ctl.sh create" >&2; exit 1; }
  echo "$z"
}

case "${1:-}" in
create)
  if gcloud compute instances list --filter="name=$NAME" --format="value(name)" | grep -q .; then
    echo "$NAME already exists (zone $(zone)). Use 'start', or 'delete' first."; exit 1
  fi
  if ! gcloud compute images describe-from-family "$IMAGE_FAMILY" --project="$IMAGE_PROJECT" >/dev/null 2>&1; then
    echo "Image family $IMAGE_FAMILY not found; picking the newest common-cu* family instead"
    IMAGE_FAMILY=$(gcloud compute images list --project="$IMAGE_PROJECT" --no-standard-images \
                     --filter="family~^common-cu1" --format="value(family)" | grep -v arm | sort -uV | tail -1)
    [[ -n "$IMAGE_FAMILY" ]] || { echo "No Deep Learning VM image family found. Ask Claude."; exit 1; }
  fi
  echo "Image family: $IMAGE_FAMILY"
  for Z in "${ZONES[@]}"; do
    echo "--- trying $Z"
    if gcloud compute instances create "$NAME" --zone="$Z" \
         --machine-type=a2-ultragpu-1g \
         --provisioning-model=SPOT --instance-termination-action=STOP \
         --max-run-duration="$MAX_RUN" --discard-local-ssds-at-termination-timestamp=true \
         --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
         --boot-disk-size=250GB --boot-disk-type=pd-balanced \
         --service-account="$SA" --scopes=cloud-platform \
         --metadata=install-nvidia-driver=True; then
      echo; echo "Created $NAME in $Z. The GPU is billing from now on. Wait ~2 minutes, then: bash ctl.sh ssh"
      exit 0
    fi
  done
  echo; echo "No zone could create the VM. If every error above says ZONE_RESOURCE_POOL_EXHAUSTED or"
  echo "'does not have enough resources', no spot A100 80GB is free right now: try again in 15-30 min."
  echo "Any other error: paste it to Claude."; exit 1
  ;;
ssh)    gcloud compute ssh "$NAME" --zone="$(zone)" ;;
start)  gcloud compute instances start "$NAME" --zone="$(zone)" ;;
stop)   gcloud compute instances stop "$NAME" --zone="$(zone)" --discard-local-ssd=true ;;
log)    gcloud compute ssh "$NAME" --zone="$(zone)" --command='tail -n 40 "$(ls -t ~/run_logs/*.log | head -1)"' ;;
bucket) gcloud storage du -s "$BUCKET/final_run" --readable-sizes ;;
status)
  echo "== VMs (RUNNING = GPU billing)"; gcloud compute instances list --format="table(name,zone.basename(),status)"
  echo "== disks (billed while they exist)"; gcloud compute disks list --format="table(name,zone.basename(),sizeGb,status)"
  ;;
delete)
  read -r -p "Delete $NAME and its disk (models, venvs, local logs)? Bucket data is NOT touched. [y/N] " a
  [[ "$a" == y ]] || exit 0
  gcloud compute instances delete "$NAME" --zone="$(zone)" --quiet
  for p in $(gcloud monitoring policies list --filter="displayName='$ALERT_NAME'" --format="value(name)"); do
    gcloud monitoring policies delete "$p" --quiet && echo "deleted the stop alert"
  done
  echo "== anything left?"; gcloud compute instances list; gcloud compute disks list
  ;;
alert)
  EMAIL="${2:?usage: bash ctl.sh alert you@example.com}"
  gcloud services enable monitoring.googleapis.com --quiet
  if gcloud monitoring policies list --filter="displayName='$ALERT_NAME'" --format="value(name)" | grep -q .; then
    echo "alert '$ALERT_NAME' already exists"; exit 0
  fi
  CHANNEL=$(gcloud beta monitoring channels list --filter="type=email AND labels.email_address=$EMAIL" --format="value(name)" | head -1)
  [[ -n "$CHANNEL" ]] || CHANNEL=$(gcloud beta monitoring channels create --display-name="final-run email" \
      --type=email --channel-labels="email_address=$EMAIL" --format="value(name)")
  cat > /tmp/final_run_alert.json <<JSONEOF
{
  "displayName": "$ALERT_NAME",
  "combiner": "OR",
  "conditions": [{
    "displayName": "$NAME reported no uptime for 10 minutes",
    "conditionAbsent": {
      "filter": "resource.type = \"gce_instance\" AND metric.type = \"compute.googleapis.com/instance/uptime\" AND metric.labels.instance_name = \"$NAME\"",
      "duration": "600s",
      "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_RATE"}]
    }
  }],
  "notificationChannels": ["$CHANNEL"],
  "documentation": {
    "mimeType": "text/markdown",
    "content": "The GPU VM **$NAME** has stopped (spot interruption, the 48 h limit, the run finished, or you stopped it).\n\nIn Cloud Shell: cd ~/intersectional-bias/final_run/vm && bash ctl.sh status, then bash ctl.sh log. If the log does not end with === FINISHED and you did not stop it yourself: bash ctl.sh start (the run resumes by itself)."
  }
}
JSONEOF
  gcloud monitoring policies create --policy-from-file=/tmp/final_run_alert.json >/dev/null
  echo "Alert created: $EMAIL gets an email ~10 minutes after $NAME stops, for any reason (and one when it is running again)."
  ;;
*) sed -n '2,13p' "$0"; exit 1 ;;
esac
