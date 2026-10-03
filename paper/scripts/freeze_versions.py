"""
freeze_versions.py

Write ``paper/requirements-lock.txt`` from the currently installed versions of
:data:`_common.PINNED_PACKAGES`, and ``paper/cloud/requirements-cloud.txt``
from the full ``pip freeze`` -- or check an environment against both.

The same list drives the provenance block stamped into every results file, so
the lock file and the results can never describe different environments.  The
lock is the provenance record (seven packages); the cloud file is the build
input a VM installs to reproduce the environment exactly (every distribution).
Both are written by one command from one interpreter, so they cannot drift
apart.

Run::

    python paper/scripts/freeze_versions.py               # write both files
    python paper/scripts/freeze_versions.py --verify      # check this interpreter
    python paper/scripts/freeze_versions.py --check-linux # will a VM resolve it?

``--verify`` is what ``paper/cloud/bootstrap.sh`` runs after installing: it
fails on any missing, extra or different distribution, on a Python minor
version other than the one of record, and when Mitiq cannot be imported --
without Mitiq the campaign would run to completion and silently omit the
Mitiq baseline.

``--check-linux`` asks pip (network, nothing installed) to resolve the cloud
file plus dgssp's own requirements for CPython 3.11 on manylinux x86-64, from
binary wheels only.  It fails if any pin has no Linux wheel, or if Linux would
pull in a distribution this Windows environment does not have -- pip would
install that one unpinned.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import (
    CLOUD_REQUIREMENTS_FILE,
    LOCK_FILE,
    PINNED_PACKAGES,
    environment_drift,
    git_state,
    normalise_name,
    package_versions,
    pin_drift,
    read_pins,
    write_cloud_requirements,
    write_requirements_lock,
)

#: The extras ``paper/cloud/bootstrap.sh`` installs dgssp with.  ``data``
#: (pandas) is not needed on the VM but is in the environment of record, so
#: installing it keeps the VM's ``pip freeze`` identical to the pinned one.
BUILD_EXTRAS: tuple[str, ...] = ("mitiq", "viz", "dev", "data")

#: Wheel tags a Debian 12 VM (glibc 2.36, x86-64) accepts.
LINUX_PLATFORMS: tuple[str, ...] = (
    "manylinux_2_36_x86_64",
    "manylinux_2_28_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
)


def read_header(path: Path) -> dict[str, str]:
    """
    The ``# key: value`` comment lines at the top of a pin file.

    Parameters
    ----------
    path:
        A file written by ``write_requirements_lock`` or
        ``write_cloud_requirements``.

    Returns
    -------
    dict[str, str]
        e.g. ``{"python": "3.11.16 (Windows-...)", "pip": "26.2.1", ...}``.
    """
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        key, sep, value = line[1:].partition(":")
        if sep and " " not in key.strip():
            header[key.strip()] = value.strip()
    return header


def dgssp_requirements(extras: tuple[str, ...] = BUILD_EXTRAS) -> list[str]:
    """
    dgssp's own requirements with ``extras``, as pip requirement strings.

    Read from the installed distribution's metadata, i.e. from
    ``pyproject.toml`` as of the last ``pip install -e``.

    Parameters
    ----------
    extras:
        The extras the VM installs.

    Returns
    -------
    list[str]
        Requirement strings with their markers stripped.
    """
    from importlib.metadata import requires

    from packaging.requirements import Requirement

    out = []
    for text in requires("dgssp") or []:
        req = Requirement(text)
        if req.marker is None or any(req.marker.evaluate({"extra": e}) for e in extras):
            req.marker = None
            out.append(str(req))
    return out


def write(lock: Path, cloud: Path) -> int:
    """
    Regenerate both files and print what went into them.

    Returns
    -------
    int
        ``0`` on success, ``1`` if a required (non-optional) package is
        missing, so the command is usable as an environment check in CI.
    """
    versions = package_versions()
    git = git_state()
    lock_path = write_requirements_lock(lock)
    cloud_path = write_cloud_requirements(cloud)

    print(f"commit : {git['short_commit']}{' (dirty)' if git['dirty'] else ''}")
    print(f"wrote  : {lock_path}")
    print(f"wrote  : {cloud_path} ({len(read_pins(cloud_path))} distributions)")
    if git["dirty"]:
        print(
            "\n  ! the tree is dirty, and both files now say so. Commit (or stash) "
            f"{git['dirty_files'][:3]} and regenerate before committing these."
        )
    print()

    missing = []
    for name in PINNED_PACKAGES:
        version = versions[name]
        if version is None:
            # Mitiq is an optional extra of the library -- but not of the
            # campaign, which `--verify` enforces separately.
            optional = name == "mitiq"
            print(f"  {name:<20} NOT INSTALLED{' (optional)' if optional else ''}")
            if not optional:
                missing.append(name)
        else:
            print(f"  {name:<20} {version}")

    if missing:
        print(f"\nMissing required packages: {', '.join(missing)}")
        return 1
    return 0


def verify(lock: Path, cloud: Path) -> int:
    """
    Check the running interpreter against both files, and that Mitiq imports.

    Returns
    -------
    int
        ``0`` when the environment is exactly the one of record, else ``1``.
    """
    from dgssp.mitigation import mitiq_is_available

    problems = environment_drift(lock, cloud)

    recorded = read_header(cloud).get("python", "").split(" ", 1)[0]
    running = platform.python_version()
    if recorded.split(".")[:2] != running.split(".")[:2]:
        problems.append(
            f"python: {running} running, {recorded or 'unknown'} recorded "
            "(the minor version must match: wheels are built per minor version)"
        )
    if not mitiq_is_available():
        problems.append(
            "mitiq: not importable -- the campaign would run with mitiq=off and "
            "the Mitiq baseline (contribution C2) would be missing from every record"
        )

    if problems:
        print(f"environment does NOT match ({len(problems)} problem(s)):")
        for line in problems:
            print(f"  {line}")
        return 1
    print(
        f"environment matches {lock.name} and {cloud.name} "
        f"({len(read_pins(cloud))} distributions, python {running}, mitiq importable)"
    )
    return 0


def check_linux(cloud: Path) -> int:
    """
    Resolve the cloud pins for CPython 3.11 on manylinux, without installing.

    Returns
    -------
    int
        ``0`` when Linux resolves to exactly the pinned set from wheels.
    """
    pins = read_pins(cloud)
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "report.json"
        command = [
            sys.executable, "-m", "pip", "install",
            "--dry-run", "--ignore-installed", "--quiet",
            "--report", str(report),
            "--target", str(Path(scratch) / "target"),
            *(arg for tag in LINUX_PLATFORMS for arg in ("--platform", tag)),
            "--python-version", "3.11", "--implementation", "cp", "--abi", "cp311",
            "--only-binary=:all:",
            "-r", str(cloud),
            *dgssp_requirements(),
        ]  # fmt: skip
        print("resolving for linux / cp311 (network; nothing is installed) ...")
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout[-2000:] + done.stderr[-4000:])
            print("FAIL: pip could not resolve the cloud pins for Linux.")
            return 1
        install = json.loads(report.read_text(encoding="utf-8"))["install"]

    resolved = {item["metadata"]["name"]: item["metadata"]["version"] for item in install}
    problems = pin_drift(pins, dict(resolved))
    sdists = [
        normalise_name(item["metadata"]["name"])
        for item in install
        if not item["download_info"]["url"].endswith(".whl")
    ]
    problems += [f"{name}: no wheel, would build from source" for name in sdists]
    if problems:
        print(f"FAIL: Linux resolves differently ({len(problems)} problem(s)):")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"OK: Linux cp311 resolves to exactly the {len(pins)} pinned distributions, all wheels.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Write the two pin files, or check an environment against them.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any failure.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Check this interpreter.")
    mode.add_argument("--check-linux", action="store_true", help="Resolve for a VM.")
    parser.add_argument("--lock", type=Path, default=LOCK_FILE)
    parser.add_argument("--cloud", type=Path, default=CLOUD_REQUIREMENTS_FILE)
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.lock, args.cloud)
    if args.check_linux:
        return check_linux(args.cloud)
    return write(args.lock, args.cloud)


if __name__ == "__main__":
    raise SystemExit(main())
