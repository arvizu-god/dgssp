"""
freeze_versions.py

Write ``paper/requirements-lock.txt`` from the currently installed versions of
:data:`_common.PINNED_PACKAGES`.

The same list drives the provenance block stamped into every results file, so
the lock file and the results can never describe different environments.

Run::

    python paper/scripts/freeze_versions.py
"""

from __future__ import annotations

from _common import (
    LOCK_FILE,
    PINNED_PACKAGES,
    git_state,
    package_versions,
    write_requirements_lock,
)


def main() -> int:
    """
    Regenerate the lock file and print what went into it.

    Returns
    -------
    int
        ``0`` on success, ``1`` if a required (non-optional) package is
        missing, so the command is usable as an environment check in CI.
    """
    versions = package_versions()
    path = write_requirements_lock(LOCK_FILE)

    git = git_state()
    print(f"commit : {git['short_commit']}{' (dirty)' if git['dirty'] else ''}")
    print(f"wrote  : {path}")
    print()

    missing = []
    for name in PINNED_PACKAGES:
        version = versions[name]
        if version is None:
            # Mitiq is an optional extra; everything else is required.
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


if __name__ == "__main__":
    raise SystemExit(main())
