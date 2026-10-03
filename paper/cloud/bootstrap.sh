#!/usr/bin/env bash
# Build the campaign environment on a clean Debian 12 VM, and prove it is the
# environment of record before anything runs in it.
#
#   sudo bash paper/cloud/bootstrap.sh [REPO_DIR]
#
# 1. apt: python3 (Debian 12 ships 3.11), python3-venv, git.
# 2. A venv at $DGSSP_VENV (default /opt/dgssp-venv), outside the checkout:
#    an untracked file outside paper's output directories would make the tree
#    dirty, and every record would say so.
# 3. pip at the version the freeze was taken with; every distribution in
#    requirements-cloud.txt, from wheels only; then dgssp itself, editable,
#    with the extras the environment of record has (data is pandas: unused on
#    the VM, but in the freeze, so the VM's freeze stays identical).
# 4. freeze_versions.py --verify: the venv's `pip freeze` equals
#    requirements-cloud.txt exactly (nothing missing, extra or different), the
#    seven provenance packages equal requirements-lock.txt, the Python minor
#    version matches the freeze, and Mitiq imports.  Any difference fails the
#    bootstrap and the campaign never starts -- in particular it cannot run
#    with mitiq=off.
#
# Idempotent: on a venv that is already built it re-checks and changes nothing.
set -euo pipefail

repo=$(cd "${1:-$(dirname "$0")/../..}" && pwd)
venv=${DGSSP_VENV:-/opt/dgssp-venv}
pins="$repo/paper/cloud/requirements-cloud.txt"

want_python=$(sed -n 's/^# python: *\([0-9]*\.[0-9]*\)\..*/\1/p' "$pins")
want_pip=$(sed -n 's/^# pip: *\([^ ]*\).*/\1/p' "$pins")
echo "bootstrap: repo=$repo venv=$venv python=$want_python.x pip=$want_pip"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv git >/dev/null

have_python=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$have_python" != "$want_python" ]; then
  echo "bootstrap: FAIL: python3 is $have_python; the pins were frozen under $want_python" >&2
  exit 1
fi

[ -x "$venv/bin/python" ] || python3 -m venv "$venv"
py="$venv/bin/python"
"$py" -m pip install --quiet --disable-pip-version-check "pip==$want_pip"
"$py" -m pip install --quiet --only-binary=:all: -r "$pins"
(cd "$repo" && "$py" -m pip install --quiet -e ".[mitiq,viz,dev,data]")

(cd "$repo" && "$py" paper/scripts/freeze_versions.py --verify)
echo "bootstrap: OK -- $("$py" --version), $py"
