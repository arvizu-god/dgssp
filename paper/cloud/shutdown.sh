#!/bin/bash
# GCE shutdown script for the WP5 campaign VM: a best-effort last sync.
#
# A Spot preemption gives the guest about 30 seconds, so this may not finish.
# It does not need to: run_campaign.py has synced after every completed stage,
# and this only rescues records of the stage in progress.  Whatever subset of
# that stage reaches the bucket is fine -- records are renamed into place
# whole (a .tmp is never uploaded), and the next VM re-runs whole every chunk
# with a record missing, so the resumed counts are the uninterrupted ones.
BUCKET=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/dgssp-bucket" || true)
SYNC=/opt/dgssp/paper/cloud/sync.sh
if [ -n "$BUCKET" ] && [ -f "$SYNC" ]; then
  timeout 25 bash "$SYNC" up /opt/dgssp/paper/data "${BUCKET%/}/data" || true
fi
