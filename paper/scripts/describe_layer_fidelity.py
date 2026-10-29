"""
describe_layer_fidelity.py

What layered-error figures does ``qiskit-ibm-runtime`` expose for a device,
and how does today's error per layered gate compare with alpha_hat times the
median CZ error?  Read-only: it reads ``backend.properties()`` and
``backend.configuration()`` through the public API, creates no job and costs
no QPU time.  It needs ``QISKIT_IBM_TOKEN`` (via
:func:`hardware_batch.connect`), so it is a command the author runs.

What 0.49 exposes (checked in its source): no layer-fidelity or EPLG
accessor.  The numbers arrive only as ``lf_<N>`` entries of
``BackendProperties.general`` (layer fidelity of an ``N``-qubit chain) with
the chain in ``general_qlists``; ``EPLG = 1 - LF**(1/(N-1))`` is derived
(:func:`dgssp.backends.layer_fidelity`).  The script also lists any other
``general`` name and any configuration key that mentions EPLG or layers, in
case the service adds one.  If nothing is exposed it says so and writes
nothing.  It never scrapes the dashboard.

The calibrations frozen for both jobs (``calibration_exec_<date>.json``)
already carry the run-day ``lf_<N>`` values; ``alpha_robustness.py`` uses
those.  This script's value is *today's*, not the run day's.

Output: ``paper/hardware/layer_fidelity_<today>.json`` (never overwritten
without ``--force``).

Run::

    python paper/scripts/describe_layer_fidelity.py
    python paper/scripts/describe_layer_fidelity.py --backend ibm_fez --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, HARDWARE_DIR, record_provenance, utc_timestamp  # noqa: E402
from dgssp.backends import layer_fidelity, median_gate_errors  # noqa: E402

SCHEMA = "dgssp/paper/layer-fidelity/1"

#: The paper's alpha fit (job of record, every error scaled).
ALPHA_FIT = DATA_DIR / "hw_alpha_exec_2026-10-25" / "fit.json"

#: ``general`` names that are per-coupler frequencies, not error figures.
_COUPLER_NAMES = re.compile(r"^(jq|zz)_\d+$")

#: Configuration keys worth recording if the service ever adds them.
_CONFIG_KEYS = re.compile(r"eplg|layer|lf_|error", re.IGNORECASE)


def _json_plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def exposed(properties: Mapping[str, Any] | None, configuration: Mapping[str, Any] | None
            ) -> dict[str, Any]:
    """
    Everything layered the two payloads carry.

    Returns
    -------
    dict
        ``layer_fidelity`` (:func:`dgssp.backends.layer_fidelity`, keys as
        strings), ``other_general`` (``general`` entries that are neither
        couplers nor ``lf_<N>``) and ``configuration`` (matching keys).
    """
    lf = layer_fidelity(properties)
    other = [row for row in (properties or {}).get("general") or []
             if not _COUPLER_NAMES.match(str(row.get("name", "")))
             and not str(row.get("name", "")).startswith("lf_")]
    config = {k: v for k, v in (configuration or {}).items() if _CONFIG_KEYS.search(str(k))}
    return {"layer_fidelity": {str(n): v for n, v in lf.items()},
            "other_general": _json_plain(other), "configuration": _json_plain(config)}


def comparison(lf: Mapping[str, Any], median_cz: float, alpha: Mapping[str, Any]) -> dict[str, Any]:
    """EPLG at the longest chain against ``alpha_hat * median CZ``."""
    n = max(int(k) for k in lf)
    top = lf[str(n)]
    eplg = top["eplg"]
    return {
        "chain_length": n,
        "lf": top["lf"],
        "eplg": eplg,
        "lf_date": top["date"],
        "median_cz_error": median_cz,
        "alpha_hat": alpha["alpha"],
        "alpha_hat_ci_scaled": alpha["ci_scaled"],
        "alpha_x_median_cz": alpha["alpha"] * median_cz,
        "ratio_eplg_to_median_cz": eplg / median_cz if eplg is not None else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layered error figures the service exposes "
                                                 "(read-only, no job).")
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--instance", default=None, help="Instance CRN; default from the env.")
    parser.add_argument("--dry-run", action="store_true", help="Print, write nothing.")
    parser.add_argument("--force", action="store_true", help="Replace today's file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from hardware_batch import connect

    today = datetime.now(timezone.utc).date().isoformat()
    out = HARDWARE_DIR / f"layer_fidelity_{today}.json"
    if out.exists() and not args.force and not args.dry_run:
        print(f"{out} exists; --force to replace it.")
        return 1

    _, backend, _ = connect(args.backend, offline=False, instance=args.instance)
    props = backend.properties(refresh=True)
    properties = props.to_dict() if props is not None else None
    configuration = backend.configuration().to_dict()
    found = exposed(properties, configuration)
    lf = found["layer_fidelity"]
    print(f"{args.backend}: properties last updated {getattr(props, 'last_update_date', None)}")
    print(f"  lf_<N> entries: {len(lf)}"
          + (f" (N = {min(map(int, lf))} .. {max(map(int, lf))})" if lf else ""))
    print(f"  other general entries: {sorted({r.get('name') for r in found['other_general']})}")
    print(f"  configuration keys: {sorted(found['configuration'])}")
    if not lf and not found["other_general"]:
        print("\nNothing layered is exposed through the API; nothing written. "
              "(Not scraping the dashboard.)")
        return 1

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "backend": args.backend,
        "fetched_utc": utc_timestamp(),
        "properties_last_update": str(getattr(props, "last_update_date", None)),
        "note": "today's value, fetched after the jobs: NOT the calibration in force on either "
                "run day (those lf_<N> values are in calibration_exec_<date>.json)",
        "exposed": found,
        "provenance": record_provenance(),
    }
    if lf:
        fit = json.loads(ALPHA_FIT.read_text(encoding="utf-8"))["fits"]["all"]
        cz = float(median_gate_errors(backend)["two_qubit"])
        payload |= {"comparison": comparison(lf, cz, fit)}
        payload |= {"eplg": payload["comparison"]["eplg"], "lf_date": payload["comparison"]["lf_date"]}
        c = payload["comparison"]
        print(f"\n  LF_{c['chain_length']} = {c['lf']:.4f} (measured {c['lf_date']}) -> "
              f"EPLG = {c['eplg']:.5f}")
        print(f"  median CZ error now {c['median_cz_error']:.5f}; alpha_hat x CZ = "
              f"{c['alpha_x_median_cz']:.5f}; EPLG / CZ = {c['ratio_eplg_to_median_cz']:.2f}")
        print("  (today's calibration, not the run day's)")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
