"""
iteration_sweep.py

Where the algorithm fails on current hardware: ``P(solution)`` against the
number of Grover iterations (C3).

Reads ``paper/instances/sweep.json`` -- the frozen sweep instances and their
``r`` ranges -- and, for each instance, runs every ``r``-variant through
:func:`dgssp.experiments.run_iteration_sweep`: each variant transpiled once at
optimization level 3 (the ``r_opt`` variant with the ``best_seed`` WP3
recorded in ``hardware_subset.json``, the others with SABRE seeds 0..31), the
whole sweep of one instance in ONE job, compared with the analytic
``sin^2((2r + 1) theta)``.

Executors:

* ``ideal``  -- noiseless Aer (the correctness check; no backend needed);
* ``noisy``  -- Aer with the noise model of a fake twin (``--backend``);
* ``zne``    -- the same, folded at ``--scales`` (partial folding), with the
  scalar exponential estimator, its bootstrap interval and resolvability flag,
  and optionally Mitiq's folds in the same job (``--mitiq``).

The recorded ``best_seed`` is only reused on the backend WP3 transpiled for
(``hardware_subset.json["backend"]``); on another backend every variant gets
the seed sweep, and the file says so.

Output: ``paper/data/sweep_<executor>.json`` (never overwritten).

Run::

    python paper/scripts/iteration_sweep.py
    python paper/scripts/iteration_sweep.py --executor noisy --backend fake_marrakesh
    python paper/scripts/iteration_sweep.py --executor zne --scales 1,1.5,2 --mitiq
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, INSTANCES_DIR, save_record  # noqa: E402
from dgssp.backends import list_fake_backends  # noqa: E402
from dgssp.experiments import run_iteration_sweep  # noqa: E402
from dgssp.instance import SubsetSumInstance  # noqa: E402
from dgssp.mitigation import mitiq_is_available  # noqa: E402
from dgssp.mitigation.zne import ZNESamplingConfig  # noqa: E402

SCHEMA = "dgssp/paper/iteration-sweep/1"
SEED_SIMULATOR = 20261020
SEED_MIN, SEED_MAX = 0, 32


def _fake_backend(name: str) -> Any:
    """A bundled fake device by name."""
    for backend in list_fake_backends():
        if backend.name == name:
            return backend
    raise SystemExit(f"No fake backend named {name!r} in this qiskit-ibm-runtime.")


def main(argv: list[str] | None = None) -> int:
    """Run the sweep for every instance in ``sweep.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--executor", choices=("ideal", "noisy", "zne"), default="ideal")
    parser.add_argument("--backend", default=None, help="fake twin (default: sweep.json's)")
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--scales", default="1,1.5,2", help="nominal ZNE scales")
    parser.add_argument("--mitiq", action="store_true", help="also run Mitiq's folds")
    args = parser.parse_args(argv)

    sweep = json.loads((INSTANCES_DIR / "sweep.json").read_text(encoding="utf-8"))
    subset = json.loads((INSTANCES_DIR / "hardware_subset.json").read_text(encoding="utf-8"))
    ranked = {r["name"]: r for r in subset.get("candidates", []) + subset["selected"]}

    backend_label = args.backend or sweep["backend"]
    device = None
    backend = None
    if args.executor != "ideal":
        from qiskit_aer import AerSimulator

        device = _fake_backend(backend_label)
        backend = AerSimulator.from_backend(device)
    reuse_seed = args.executor == "ideal" or backend_label == subset["backend"]

    zne_cfg = None
    if args.executor == "zne":
        zne_cfg = ZNESamplingConfig(
            scales=[float(s) for s in args.scales.split(",")],
            shots_per_scale=args.shots,
            mode="scalar",
            method="exponential",
            asymptote="p_uniform",
            seed_simulator=SEED_SIMULATOR,
        )

    started = time.time()
    results = []
    for spec in sweep["instances"]:
        instance = SubsetSumInstance(items=spec["items"], target=spec["target"], name=spec["name"])
        best_seeds = (
            {spec["r_opt"]: ranked[spec["name"]]["best_seed"]} if reuse_seed else None
        )
        out = run_iteration_sweep(
            instance,
            spec["iterations"],
            backend,
            executor=args.executor,
            zne_config=zne_cfg,
            shots=args.shots,
            best_seeds=best_seeds,
            seed_min=SEED_MIN,
            seed_max=SEED_MAX,
            include_mitiq=args.mitiq and mitiq_is_available(),
            seed_simulator=SEED_SIMULATOR,
        )
        out["r_opt"] = spec["r_opt"]
        out["best_seed_reused"] = best_seeds
        results.append(out)
        for row in out["rows"]:
            zne = row.get("zne") or {}
            print(
                f"{spec['name']:<12} r={row['iterations']}  P_sol={row['p_sol']:.4f} "
                f"[{row['p_sol_ci'][0]:.4f}, {row['p_sol_ci'][1]:.4f}]  "
                f"analytic={row['analytic']:.4f}  2q={row['two_q_count']}"
                + (
                    f"  zne={zne.get('value'):.4f} resolved={zne.get('resolved')}"
                    if zne
                    else ""
                )
            )

    payload = {
        "schema": SCHEMA,
        "executor": args.executor,
        "backend": backend_label if args.executor != "ideal" else "aer_simulator",
        "shots": args.shots,
        "zne_config": None if zne_cfg is None else {
            "scales": list(zne_cfg.scales),
            "method": zne_cfg.method,
            "asymptote": zne_cfg.asymptote,
            "fold_seed": zne_cfg.fold_seed,
        },
        "sweep_source": "sweep.json",
        "family_payload_sha256": sweep.get("family_payload_sha256"),
        "results": results,
        "elapsed_s": round(time.time() - started, 2),
    }
    from _common import record_provenance

    payload["provenance"] = record_provenance(device)
    path = save_record(DATA_DIR / f"sweep_{args.executor}.json", payload)
    print(f"\nwrote {path}  ({time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
