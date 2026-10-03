#!/usr/bin/env bash
# Mirror the campaign's outputs between a data directory and a bucket prefix.
#
#   sync.sh up   DATA_DIR gs://BUCKET/PREFIX   # sim_* directories and logs -> bucket
#   sync.sh down DATA_DIR gs://BUCKET/PREFIX   # bucket -> DATA_DIR (resume after preemption)
#
# Only campaign outputs travel: the sim_* directories, campaign_log.json and
# campaign_stdout.log -- never the tracked files of paper/data, so syncing the
# bucket back down cannot touch anything git tracks.  A record being written
# right now (*.tmp; save_record renames it into place when complete) is never
# uploaded.  Nothing is deleted on either side.
#
# run_campaign.py's upload() does the same "up" from Python after every stage;
# this copy is what the VM's startup and shutdown scripts use.
set -euo pipefail

direction=${1:?usage: sync.sh up|down DATA_DIR gs://BUCKET/PREFIX}
data=${2:?data directory}
url=${3:?gs://BUCKET/PREFIX}
url=${url%/}

case "$direction" in
  up)
    for dir in "$data"/sim_*/; do
      [ -d "$dir" ] || continue
      name=$(basename "$dir")
      gcloud storage rsync --recursive --exclude='.*\.tmp$' "$dir" "$url/$name"
    done
    for file in campaign_log.json campaign_stdout.log; do
      if [ -f "$data/$file" ]; then
        gcloud storage cp "$data/$file" "$url/$file"
      fi
    done
    ;;
  down)
    mkdir -p "$data"
    if gcloud storage ls "$url/" >/dev/null 2>&1; then
      gcloud storage rsync --recursive "$url" "$data"
    else
      echo "sync.sh: nothing at $url yet -- a first run"
    fi
    ;;
  *)
    echo "usage: sync.sh up|down DATA_DIR gs://BUCKET/PREFIX" >&2
    exit 2
    ;;
esac
