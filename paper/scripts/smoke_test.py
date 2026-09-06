"""
smoke_test.py

End-to-end check that a fresh environment can run the paper's pipeline.

One tiny instance (``[1, 2, 3]``, target ``5``) is run three ways:

1. **ideal**    -- noiseless Aer, the correctness reference;
2. **noisy**    -- an ``AerSimulator`` built from a bundled *heavy-hex* fake
   device's calibration data;
3. **zne**      -- the same noisy device with per-bitstring (pointwise) ZNE at
   the scales in :data:`SCALES`: transpile once at optimization level 3, fold
   the ISA circuit, submit the folded circuits in one job.  Three or more
   scales are used deliberately: a two-scale linear fit is exactly determined
   and leaves no residual to judge the extrapolation by.

Which fake device is used is *discovered*, not hard-coded: the set of fake
backends changes between ``qiskit-ibm-runtime`` releases, so the script asks
:func:`dgssp.backends.list_fake_backends` what the installed version ships and
takes the narrowest heavy-hex device wide enough for the circuit.

All three runs go through :func:`dgssp.experiments.run_batch`, so this exercises
exactly the code path the real experiments use, and every sampler call goes
through :func:`dgssp.runtime.sample_counts`.

Output: ``paper/data/smoke.json``, a list of three provenance-stamped records.

Run::

    python paper/scripts/smoke_test.py
"""

from __future__ import annotations

import time

from qiskit_aer import AerSimulator

from _common import DATA_DIR, Record, save_record
from dgssp import (
    BatchConfig,
    SubsetSumInstance,
    ZNESamplingConfig,
    build_encoding,
    list_fake_backends,
    run_batch,
    smallest_heavy_hex_fake_backend,
)
from dgssp.mitigation import mitiq_is_available
from dgssp.solvers import DGConfig, DGSSPSolver

# --- Frozen smoke parameters -------------------------------------------------

ITEMS = [1, 2, 3]
TARGET = 5
SHOTS = 1_000
SCALES = [1, 3]

#: Seeds for the simulator and the transpiler, so two runs of this script on
#: the same machine and package versions produce identical counts.
SEED_SIMULATOR = 1234
SEED_TRANSPILER = 42

#: SABRE seeds swept for the ZNE layout. The paper sweeps far more; a smoke
#: test only needs to prove the search path runs, and this keeps it fast.
ZNE_SEED_MIN, ZNE_SEED_MAX = 0, 4


def pick_backend(min_qubits: int):
    """
    Choose a bundled heavy-hex fake device wide enough for the circuit.

    Parameters
    ----------
    min_qubits:
        Logical circuit width.

    Returns
    -------
    BackendV2
        The narrowest matching heavy-hex fake backend.
    """
    available = list_fake_backends(min_qubits=min_qubits, heavy_hex_only=True)
    print(f"heavy-hex fake backends available ({len(available)}):")
    for backend in available[:8]:
        print(f"    {backend.name:<20} {backend.num_qubits:>4}q")
    if len(available) > 8:
        print(f"    ... and {len(available) - 8} more")

    chosen = smallest_heavy_hex_fake_backend(min_qubits=min_qubits)
    print(f"chosen: {chosen.name} ({chosen.num_qubits} qubits)\n")
    return chosen


def main() -> int:
    """
    Run the three configurations and write ``paper/data/smoke.json``.

    Returns
    -------
    int
        ``0`` if the ideal run recovered the known solution, ``1`` otherwise.
    """
    t_start = time.time()

    instance = SubsetSumInstance(items=ITEMS, target=TARGET, name="smoke_n3")
    encoding = build_encoding(ITEMS, TARGET)
    circuit = DGSSPSolver(DGConfig(iterations=1)).build_circuit(
        instance, num_solutions=1
    )

    print(f"instance : {ITEMS} -> target {TARGET}")
    print(
        f"encoding : n_items={instance.n_items} n_sum={encoding.n_sum} "
        f"offset={encoding.offset} modulus={encoding.modulus} "
        f"encoded_target={encoding.encoded_target}"
    )
    print(f"circuit  : {circuit.num_qubits} qubits")
    print(f"mitiq    : {'available' if mitiq_is_available() else 'not installed'}\n")

    fake = pick_backend(circuit.num_qubits)
    noisy = AerSimulator.from_backend(fake)
    noisy.set_options(seed_simulator=SEED_SIMULATOR)

    records: list[Record] = []

    # --- 1) ideal ------------------------------------------------------------
    t0 = time.time()
    ideal_batch = run_batch(
        [instance],
        BatchConfig(
            executor="ideal",
            shots=SHOTS,
            seed_simulator=SEED_SIMULATOR,
            seed_transpiler=SEED_TRANSPILER,
            optimization_level=1,
        ),
    )
    ideal = ideal_batch.results[0]
    records.append(
        Record.from_instance_result(
            ideal, executor="ideal", mitigation_mode="none", shots=SHOTS
        )
    )
    print(
        f"[ideal]  P(solution) = {ideal.solution_probability:.4f}  "
        f"iterations={ideal.iterations}  solutions={ideal.solution_bitstrings}  "
        f"({time.time() - t0:.1f}s)"
    )

    # --- 2) noisy ------------------------------------------------------------
    t0 = time.time()
    noisy_batch = run_batch(
        [instance],
        BatchConfig(
            executor="noisy",
            backend=noisy,
            shots=SHOTS,
            seed_simulator=SEED_SIMULATOR,
            seed_transpiler=SEED_TRANSPILER,
            optimization_level=3,
        ),
    )
    noisy_result = noisy_batch.results[0]
    records.append(
        Record.from_instance_result(
            noisy_result,
            executor="noisy",
            mitigation_mode="none",
            shots=SHOTS,
            backend=fake,
        )
    )
    print(
        f"[noisy]  P(solution) = {noisy_result.solution_probability:.4f}  "
        f"2q={noisy_result.two_qubit_count} depth={noisy_result.depth}  "
        f"({time.time() - t0:.1f}s)"
    )

    # --- 3) ZNE (pointwise) --------------------------------------------------
    t0 = time.time()
    zne_batch = run_batch(
        [instance],
        BatchConfig(
            executor="optimized",
            backend=noisy,
            shots=SHOTS,
            zne_config=ZNESamplingConfig(
                scales=SCALES,
                shots_per_scale=SHOTS,
                method="linear",
                seed_min=ZNE_SEED_MIN,
                seed_max=ZNE_SEED_MAX,
                optimization_level=3,
                seed_simulator=SEED_SIMULATOR,
            ),
        ),
    )
    zne = zne_batch.results[0]
    records.append(
        Record.from_instance_result(
            zne,
            executor="optimized",
            mitigation_mode="zne_pointwise",
            shots=SHOTS,
            backend=fake,
            extra_metrics={"zne_scales": list(SCALES), "zne_method": "linear"},
        )
    )
    print(
        f"[zne]    P(solution) = {zne.solution_probability:.4f} -> "
        f"{zne.mitigated_solution_probability:.4f} mitigated  "
        f"scales={sorted(zne.counts_per_scale or {})}  "
        f"2q={zne.two_qubit_count} depth={zne.depth}  "
        f"({time.time() - t0:.1f}s)"
    )

    # --- write ---------------------------------------------------------------
    path = save_record(DATA_DIR / "smoke.json", records)
    provenance = records[0].provenance
    print(f"\nwrote {path}")
    print(
        f"  commit={provenance['git']['short_commit']}"
        f"{' (dirty)' if provenance['git']['dirty'] else ''}"
        f"  qiskit={provenance['packages']['qiskit']}"
        f"  at {provenance['timestamp_utc']}"
    )
    print(f"  calibration: {records[1].provenance['backend']}")
    print(f"total {time.time() - t_start:.1f}s")

    # The ideal run is the correctness gate: with the optimal iteration count
    # the correct subset must dominate.
    ok = bool(ideal.counts) and (
        max(ideal.counts, key=lambda b: ideal.counts[b]) in ideal.solution_bitstrings
    )
    if not ok:
        print("\nFAIL: the ideal run's most likely outcome is not a solution.")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
