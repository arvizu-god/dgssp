"""
describe_account.py

Print everything the IBM Quantum account exposes, in one command, so that
``paper/hardware/account_facts.md`` can be filled from a printout instead of
from memory.

Reports plan / instances / usage / backends via
:func:`dgssp.runtime.print_account_summary`, and additionally flags which of the
visible devices are heavy-hex -- the only ones this paper targets.

This is the one script here that touches the network. It submits nothing and
costs no QPU time.

Run::

    python paper/scripts/describe_account.py
    python paper/scripts/describe_account.py --json paper/hardware/account_snapshot.json

Credentials are read from the saved default account. If none is saved, save one
once with ``dgssp.runtime.save_account(token, instance_crn)``; nothing in this
repository writes credentials implicitly, and the token is stripped from every
printed and saved payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import HARDWARE_DIR, record_provenance
from dgssp.runtime import get_service, print_account_summary


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns
    -------
    argparse.Namespace
        ``name`` (saved account to load) and ``json`` (optional output path).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Print plan, instances, usage and backends for the IBM Quantum "
            "account, to fill paper/hardware/account_facts.md."
        )
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name of a saved account to load (default: the default account).",
    )
    parser.add_argument(
        "--json",
        default=None,
        help=(
            "Write the raw payload here as JSON. Bare filenames land in "
            f"{HARDWARE_DIR}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """
    Print the account summary and optionally save it.

    Returns
    -------
    int
        ``0`` on success, ``1`` if the service could not be created (no saved
        credentials, or no network).
    """
    args = parse_args()

    try:
        service = get_service(name=args.name)
    except Exception as exc:
        print(f"Could not create a QiskitRuntimeService: {type(exc).__name__}: {exc}")
        print(
            "\nSave credentials once with:\n"
            "    python -c \"from dgssp.runtime import save_account; "
            'save_account(\'<TOKEN>\', \'<INSTANCE_CRN>\')"'
        )
        return 1

    info = print_account_summary(service)

    heavy_hex = [
        row
        for row in info.get("backends", [])
        if isinstance(row, dict)
        and not row.get("simulator")
        and isinstance(row.get("processor_type"), dict)
    ]
    if heavy_hex:
        print("\nheavy-hex candidates for this paper:")
        for row in sorted(heavy_hex, key=lambda r: r.get("num_qubits") or 0):
            family = row["processor_type"]
            print(
                f"  {row['name']:<24} {row['num_qubits']:>4}q  "
                f"{family.get('family')} r{family.get('revision')}"
            )

    if args.json:
        path = Path(args.json)
        if path.parent == Path("."):
            path = HARDWARE_DIR / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"account": info, "provenance": record_provenance()}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {path}")

    print(
        "\nNow fill paper/hardware/account_facts.md. Anything marked [verify] "
        "there is not answerable from this printout alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
