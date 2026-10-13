"""
Tests for ``paper/cloud``: the VM scripts and the README that launches them.

They cannot run a VM.  What they pin is what would fail silently, or an hour
in: CRLF endings on a script bash will execute, the bootstrap installing
different extras from the ones the Linux check resolved, the shell sync and
the Python sync disagreeing about what goes to the bucket, and the launch
command drifting from the machine shape and metadata the scripts expect.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD = REPO_ROOT / "paper" / "cloud"
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SHELL_SCRIPTS = sorted(CLOUD.glob("*.sh"))


def test_the_four_scripts_exist():
    assert {p.name for p in SHELL_SCRIPTS} == {
        "bootstrap.sh",
        "shutdown.sh",
        "startup.sh",
        "sync.sh",
    }


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_have_unix_line_endings(script):
    """
    ``startup.sh`` is uploaded as metadata from a Windows working tree; with
    CRLF endings bash on the VM fails on the first line.  ``.gitattributes``
    forces LF; this checks the file actually on disk.
    """
    assert b"\r" not in script.read_bytes()


def test_gitattributes_forces_lf_on_shell_scripts():
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"^\*\.sh\s+text\s+eol=lf\s*$", text, re.MULTILINE)


def test_the_campaign_data_is_committed_as_the_vm_wrote_it():
    """
    Decision F: the records go into git as they are.  Without ``eol=lf`` a
    Windows checkout would rewrite every line ending, and the files here would
    no longer be the VM's files byte for byte.
    """
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in (
        "paper/data/sim_*/*.json",
        "paper/data/campaign_log.json",
        "paper/data/campaign_stdout.log",
    ):
        assert re.search(rf"^{re.escape(pattern)}\s+text\s+eol=lf\s*$", text, re.MULTILINE), pattern
    readme = (CLOUD / "README.md").read_text(encoding="utf-8")
    assert "git add paper/data/sim_* paper/data/campaign_log.json paper/data/campaign_stdout.log" in readme


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse(script):
    """``bash -n``, where a working bash is available."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    try:
        probe = subprocess.run([bash, "-c", "echo ok"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("bash not runnable")
    if probe.stdout.strip() != "ok":
        pytest.skip("bash on PATH is not usable (e.g. a WSL stub)")
    done = subprocess.run(
        [bash, "-n", script.name], cwd=script.parent, capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr


def test_bootstrap_installs_the_extras_the_linux_check_resolves():
    """``bootstrap.sh`` and ``freeze_versions.BUILD_EXTRAS`` must not drift."""
    freeze_versions = pytest.importorskip("freeze_versions")
    script = (CLOUD / "bootstrap.sh").read_text(encoding="utf-8")
    assert f'.[{",".join(freeze_versions.BUILD_EXTRAS)}]' in script
    assert "requirements-cloud.txt" in script
    assert "--only-binary=:all:" in script
    assert "freeze_versions.py --verify" in script


def test_bootstrap_reads_the_header_the_pin_file_has():
    """The sed patterns in bootstrap.sh match the header write_cloud_requirements writes."""
    header = (CLOUD / "requirements-cloud.txt").read_text(encoding="utf-8")
    assert re.search(r"^# python: *\d+\.\d+\.", header, re.MULTILINE)
    assert re.search(r"^# pip: *\S+", header, re.MULTILINE)


def test_the_shell_sync_and_the_python_sync_move_the_same_files():
    """What the shutdown script rescues is what run_campaign.py uploads."""
    run_campaign = pytest.importorskip("run_campaign")
    sync = (CLOUD / "sync.sh").read_text(encoding="utf-8")
    assert "--exclude='.*\\.tmp$'" in sync
    assert '"$data"/sim_*/' in sync
    for name in (run_campaign.LOG_NAME, run_campaign.STDOUT_LOG_NAME):
        assert name in sync
    assert "delete-unmatched" not in sync


def test_startup_runs_the_campaign_the_way_the_brief_says():
    script = (CLOUD / "startup.sh").read_text(encoding="utf-8")
    for fragment in (
        "meta dgssp-commit",
        "meta dgssp-bucket",
        "^[0-9a-f]{40}$",  # a pinned full SHA, never a branch
        'checkout --quiet --detach "$COMMIT"',
        "bootstrap.sh",
        "sync.sh\" down",
        'run_campaign.py --upload "$BUCKET/data"',
        "verify_campaign.py --commit",
        "shutdown -h now",
    ):
        assert fragment in script, fragment
    # Restore happens after the tests, so the suite never sees campaign data.
    assert script.index("pytest") < script.index('sync.sh" down')


def test_the_launch_command_is_the_planned_shape():
    readme = (CLOUD / "README.md").read_text(encoding="utf-8")
    for fragment in (
        "--machine-type=c2d-highcpu-32",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--max-run-duration=14h",  # a full run measured ~9-10 h; 8 h would cut it
        '$Prefix = "$Bucket/wp5-run2"',  # run 1's folder holds data from another commit
        "--image-family=debian-12",
        "--image-project=debian-cloud",
        "--boot-disk-size=50GB",
        "--scopes=storage-rw",
        "dgssp-commit=$Commit",
        "dgssp-bucket=$Prefix",
        "startup-script=paper/cloud/startup.sh",
        "shutdown-script=paper/cloud/shutdown.sh",
        'gcloud storage rsync --recursive "$Prefix/data" paper/data',
        "verify_campaign.py --commit $Commit",
    ):
        assert fragment in readme, fragment
