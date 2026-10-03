#!/bin/bash
# GCE startup script for the WP5 campaign VM.
#
#   1. check out the pinned commit (never "latest main");
#   2. build and verify the environment (bootstrap.sh);
#   3. run the test suite, and stop if anything fails;
#   4. restore whatever a preempted predecessor left in the bucket;
#   5. run the whole campaign, syncing to the bucket after every stage;
#   6. verify the dataset on the machine that produced it, sync, power off.
#
# Instance metadata (the gcloud command in paper/cloud/README.md sets these):
#   dgssp-commit    full 40-character SHA to check out                (required)
#   dgssp-bucket    gs://BUCKET/PREFIX: data -> PREFIX/data, logs -> PREFIX/vm
#                                                                     (required)
#   dgssp-repo      clone URL (default: the public GitHub repository)
#   dgssp-args      extra run_campaign.py arguments, e.g. "--max-threads 30"
#   dgssp-tests     "yes" (default) runs pytest first; "no" skips it
#   dgssp-shutdown  "yes" (default) powers off at the end; "no" keeps the VM up
#
# Idempotent: on a fresh VM after a preemption it restores the bucket and
# resumes -- run_campaign.py skips every complete stage and every record on
# disk, and re-runs whole only the chunks a preemption cut, so the counts are
# the ones an uninterrupted run would have drawn.
#
# Exit status (also in the log): 0 complete and verified; 1 a step failed;
# 2 metadata missing; 3 a sync failed; 4 the campaign ran but did not verify.
set -uo pipefail

LOG=/var/log/dgssp-startup.log
exec > >(tee -a "$LOG") 2>&1

WORK=/opt/dgssp
VENV=/opt/dgssp-venv
DATA=$WORK/paper/data

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] dgssp: $*"; }

COMMIT=$(meta dgssp-commit || true)
BUCKET=$(meta dgssp-bucket || true)
BUCKET=${BUCKET%/}
REPO_URL=$(meta dgssp-repo || echo https://github.com/arvizu-god/dgssp.git)
EXTRA_ARGS=$(meta dgssp-args || true)
RUN_TESTS=$(meta dgssp-tests || echo yes)
POWER_OFF=$(meta dgssp-shutdown || echo yes)

finish() {
  local status=$1
  say "finished with status $status"
  if [ -n "$BUCKET" ] && command -v gcloud >/dev/null; then
    gcloud storage cp "$LOG" \
      "$BUCKET/vm/startup-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).log" || true
  fi
  if [ "$POWER_OFF" = yes ]; then
    say "powering off"
    shutdown -h now
  fi
  exit "$status"
}

install_gcloud() {
  apt-get update -qq
  apt-get install -y -qq apt-transport-https ca-certificates gnupg curl >/dev/null
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -qq
  apt-get install -y -qq google-cloud-cli >/dev/null
}

say "boot on $(hostname): commit=$COMMIT bucket=$BUCKET args='$EXTRA_ARGS' tests=$RUN_TESTS"
if [ -z "$COMMIT" ] || [ -z "$BUCKET" ]; then
  say "FAIL: the dgssp-commit and dgssp-bucket metadata are required"
  finish 2
fi
if ! [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  say "FAIL: dgssp-commit must be a full 40-character SHA, got '$COMMIT'"
  finish 2
fi

export DEBIAN_FRONTEND=noninteractive
if ! command -v git >/dev/null; then
  apt-get update -qq && apt-get install -y -qq git >/dev/null
fi
if ! command -v gcloud >/dev/null; then
  say "gcloud not on this image; installing google-cloud-cli"
  install_gcloud || { say "FAIL: could not install gcloud"; finish 1; }
fi

# 1. The pinned commit.
if [ ! -d "$WORK/.git" ]; then
  git clone --quiet --no-checkout "$REPO_URL" "$WORK" || { say "FAIL: clone"; finish 1; }
fi
git -C "$WORK" fetch --quiet origin || true
if ! git -C "$WORK" checkout --quiet --detach "$COMMIT"; then
  say "FAIL: $COMMIT is not on $REPO_URL -- push it first"
  finish 1
fi
say "checked out $(git -C "$WORK" log -1 --format='%h %s')"
if ! cmp -s "$0" "$WORK/paper/cloud/startup.sh"; then
  say "warning: this startup script differs from paper/cloud/startup.sh at $COMMIT"
fi

# 2. The environment.
bash "$WORK/paper/cloud/bootstrap.sh" "$WORK" || { say "FAIL: bootstrap"; finish 1; }
PY=$VENV/bin/python

# 3. The tests, before any data is on disk.
if [ "$RUN_TESTS" = yes ]; then
  say "running the test suite"
  (cd "$WORK" && "$PY" -m pytest -q -x -p no:cacheprovider) || { say "FAIL: tests"; finish 1; }
fi

# 4. What a preempted predecessor left.
bash "$WORK/paper/cloud/sync.sh" down "$DATA" "$BUCKET/data" || { say "FAIL: restore"; finish 3; }

# 5. The campaign.  run_campaign.py syncs before the first stage and after every
#    stage; it refuses to start on a dirty tree, a drifted environment, without
#    Mitiq, or on top of records from another commit.
cd "$WORK" || finish 1
say "starting run_campaign.py"
# shellcheck disable=SC2086  # EXTRA_ARGS is a list of arguments on purpose
"$PY" paper/scripts/run_campaign.py --upload "$BUCKET/data" --keep-going $EXTRA_ARGS \
  2>&1 | tee -a "$DATA/campaign_stdout.log"
status=${PIPESTATUS[0]}
say "run_campaign.py exited $status"

# 6. Verify where it was produced, then the last sync.
"$PY" paper/scripts/verify_campaign.py --commit "$COMMIT" 2>&1 | tee -a "$DATA/campaign_stdout.log"
verified=${PIPESTATUS[0]}
bash "$WORK/paper/cloud/sync.sh" up "$DATA" "$BUCKET/data" || status=3
if [ "$status" = 0 ] && [ "$verified" != 0 ]; then
  status=4
fi
finish "$status"
