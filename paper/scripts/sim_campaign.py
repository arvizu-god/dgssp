"""
sim_campaign.py

The simulation campaign: every frozen instance, run every way, one stage at a
time, resumably.

Four stages, one command each:

``--stage ideal``
    Noiseless Aer, optimization level 1 at a fixed transpiler seed, 10 000
    shots.  This is the correctness gate for everything downstream: each
    ``M >= 1`` instance's ``P(solution)`` is compared with
    ``sin^2((2 r_opt + 1) theta)`` at 4 sigma and with the ``p_ideal`` already
    recorded in ``hardware_subset.json`` at 3 sigma, and the stage writes
    nothing unless every instance passes.

    Those per-instance thresholds are applied **with a multiplicity
    correction**, because they are applied 347 times.  A 3-sigma two-sided test
    has a per-test false-alarm rate of 2.7e-3; run over the 347 instances with
    ``M >= 1`` it fires with probability 0.61 on perfectly correct data, so as
    a literal gate it would stop a good run more often than not.  The stage
    therefore fails when a deviation exceeds the Bonferroni threshold (the
    sigma at which the *family-wise* false-alarm rate equals the nominal
    per-test one: 4.5 and 5.2 sigma here), or when the *number* of nominal
    exceedances is itself implausible -- which is what a small systematic bias
    would look like.  Every nominal exceedance is printed either way, with the
    number chance predicts.  ``--strict-acceptance`` restores the literal
    per-instance gate.

``--stage noisy``
    The same circuits at optimization level 3 on the noisy backend, unmitigated.

``--stage zne``
    Transpile **once** -- reusing the ``best_seed`` from
    ``hardware_subset.json`` when the backend is that file's backend, and
    searching the SABRE seed range otherwise -- then fold the ISA circuit at
    the stage's scale grid, add Mitiq's folds of the same circuits, and submit
    the lot in one job per chunk.  Every estimator runs offline from those
    counts: the scalar fit under all four extrapolation models, both pointwise
    renormalisations, the pathology diagnostic and Mitiq's own extrapolation.

``--stage sweep``
    ``P(solution)`` against the Grover iteration count for the two instances in
    ``sweep.json``, ideally, noisily and with scalar ZNE.

Scale grids (``--grid``)

``odd``
    ``[1, 3, 5]``: the negative control.  At ``n >= 4`` the folded signal is
    below the shot noise at ``lambda = 5``, only ``lambda = 1`` is informative,
    and the two-parameter exponential is not identified -- its chi-square is
    flat in the decay rate and the extrapolation tracks whatever bound
    ``exp_rate_max`` imposes.
``planned``
    :func:`dgssp.mitigation.zne.plan_scales` per instance, from the ``survival``
    and ``p_ideal`` already in ``hardware_subset.json``: the longest prefix of
    :data:`PLAN_CANDIDATES` whose *predicted* signal stays at least
    :data:`PLAN_K` shot-noise sigma above ``P_uniform``, thinned to
    :data:`PLAN_POINTS` points.
``odd7``
    ``[1, 3, 5, 7]`` on Family P, ``n <= 4`` only: four points, for comparing
    the four extrapolation models at a degree of freedom to spare.

An instance with no signal -- every ``M = 0`` instance, where
``p_ideal - p_uniform`` is exactly zero -- gets no planned grid at all
(``plan_scales`` returns ``[1.0]``), and a single scale can neither extrapolate
nor diagnose the pathology.  Those instances fall back to
:data:`PATHOLOGY_GRID`, and every record says which of the two happened in
``metrics["campaign"]["grid_source"]``.

Output

One record per (instance, stage, noise, grid), at
``paper/data/sim_<stage>_<noise>[_<grid>]/<instance>.json`` (the ideal stage
carries no noise tag: it does not depend on the device).  A re-run skips every
name that already exists, so the campaign is resumable and a completed stage
re-runs as a no-op.  Jobs are chunked by **circuits**, not instances: at most
``--chunk`` circuits per :func:`dgssp.runtime.sample_counts` call.

Staying inside the machine
--------------------------

Aer takes every core it can see and holds them for the whole run, which on a
laptop is a thermal event rather than a speed-up.  Four controls split a long
stage into sittings that finish:

``--max-threads N``
    Cap the simulator's thread count.  The counts do not depend on it -- with
    a fixed ``seed_simulator`` every experiment draws from its own seeded
    stream, so one thread and sixteen give bit-identical results -- which means
    a throttled run merges with an unthrottled one and a stage half-finished at
    full speed can be finished at half speed.
``--max-minutes M``
    Stop cleanly after the chunk that crosses ``M`` minutes and report what is
    left.  Re-running the same command continues from there.  It does not apply
    to the ideal stage, which must judge every instance together before it may
    write any of them (and takes 1.5 minutes anyway).
``--cooldown S``
    Idle ``S`` seconds between chunks.
``--n 2 3 4``
    Only these index-register widths.  Cost per instance rises steeply with
    ``n``: about 4 s at ``n = 2`` against 26 s at ``n = 6`` for a three-scale
    ZNE run, so ``n = 6`` alone is a third of the family's cost.

``OMP_NUM_THREADS`` caps NumPy's own BLAS threads and must be set *before*
Python starts; ``--max-threads`` cannot do it from inside the process.

Run::

    python paper/scripts/sim_campaign.py --stage ideal  --family all
    python paper/scripts/sim_campaign.py --stage noisy  --family all
    python paper/scripts/sim_campaign.py --stage zne    --family all --grid both
    python paper/scripts/sim_campaign.py --stage zne    --family P   --grid odd7
    python paper/scripts/sim_campaign.py --stage sweep

    # the same, in sittings, on a machine that has to stay usable
    python paper/scripts/sim_campaign.py --stage zne --family all --grid odd \\
        --n 2 3 4 --max-threads 8 --cooldown 10 --max-minutes 45
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scipy.stats import binom, norm

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import (  # noqa: E402
    DATA_DIR,
    HARDWARE_DIR,
    INSTANCES_DIR,
    Record,
    save_record,
)
from dgssp import SubsetSumInstance  # noqa: E402
from dgssp.backends import (  # noqa: E402
    calibration_timestamp,
    list_fake_backends,
    load_noise_model,
    save_noise_model,
)
from dgssp.execution import exact_index_distribution  # noqa: E402
from dgssp.experiments import (  # noqa: E402
    BatchConfig,
    PreparedInstance,
    build_instance_result,
    prepare_from_records,
    run_batch,
    run_iteration_sweep,
)
from dgssp.mitigation.zne import (  # noqa: E402
    ZNE_METHODS,
    ZNESamplingConfig,
    plan_scales,
    run_zne_batch,
)
from dgssp.runtime import backend_name  # noqa: E402
from dgssp.solvers import grover_success_probability, optimal_iterations  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen campaign parameters
# ---------------------------------------------------------------------------

STAGES: tuple[str, ...] = ("ideal", "noisy", "zne", "sweep")
GRIDS: tuple[str, ...] = ("odd", "planned", "both", "odd7")
FAMILIES: tuple[str, ...] = ("P", "S", "all")

#: Shots per circuit, per stage.  The ZNE stage spends its shots per *scale*.
IDEAL_SHOTS = 10_000
NOISY_SHOTS = 10_000
ZNE_SHOTS = 8_192

#: Seeds.  The master seed is the instance family's, so every number in the
#: paper traces back to one integer.
SEED_SIMULATOR = 20261020
SEED_TRANSPILER = 0
FOLD_SEED = 0
BOOTSTRAP_B = 500
BOOTSTRAP_SEED = 0

#: SABRE seed range used only when the backend is *not* the one
#: ``hardware_subset.json`` recorded its ``best_seed`` values against.
SEED_MIN, SEED_MAX = 0, 32

#: The negative control and the four-point model-comparison grid.
ODD_SCALES: tuple[float, ...] = (1.0, 3.0, 5.0)
ODD7_SCALES: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0)

#: ``odd7`` is run only where four scales are affordable and the signal at
#: lambda = 7 is not hopeless from the start.
ODD7_FAMILY = "P"
ODD7_MAX_N = 4

#: Candidates :func:`~dgssp.mitigation.zne.plan_scales` may choose from, the
#: required signal-to-noise ratio, and how many points survive the thinning.
PLAN_CANDIDATES: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)
PLAN_K = 3.0
PLAN_POINTS = 3

#: Grid for instances the planner gives a single scale -- every ``M = 0``
#: instance, whose signal is exactly zero.  Extrapolating them is meaningless,
#: but the *pathology* diagnostic is the point of running them: it needs
#: several scales to measure how far mitigation moves a distribution that
#: should not move at all.
PATHOLOGY_GRID: tuple[float, ...] = (1.0, 1.5, 2.0)

#: The reported estimator.  Every method in :data:`dgssp.mitigation.
#: ZNE_METHODS` is fitted beside it from the same counts.
ZNE_METHOD = "exponential"
ZNE_MODE = "scalar"
ZNE_RENORM = "clip"

#: Acceptance thresholds of the ideal stage, in sigma (see the module
#: docstring: they are nominal per-instance levels, corrected for multiplicity
#: before they stop a run).
ANALYTIC_SIGMA = 4.0
RECORDED_SIGMA = 3.0

#: Shots behind the ``p_ideal`` recorded in ``hardware_subset.json``.
RECORDED_IDEAL_SHOTS = 20_000

#: Tail probability at which the *count* of nominal exceedances is called
#: implausible.  A systematic bias shows up as too many mild exceedances long
#: before it shows up as one huge one.
COUNT_TAIL_ALPHA = 1e-3

#: Default circuits per job.
DEFAULT_CHUNK = 50


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_inputs(
    instances_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Read the three frozen instance files and check they describe one family.

    Parameters
    ----------
    instances_dir:
        ``paper/instances``.

    Returns
    -------
    tuple
        ``(family, hardware_subset, sweep)`` payloads.

    Raises
    ------
    ValueError
        If the subset or the sweep was built from a different family payload.
    """
    family = json.loads((instances_dir / "family.json").read_text(encoding="utf-8"))
    subset = json.loads(
        (instances_dir / "hardware_subset.json").read_text(encoding="utf-8")
    )
    sweep = json.loads((instances_dir / "sweep.json").read_text(encoding="utf-8"))

    digest = family["payload_sha256"]
    for name, payload in (("hardware_subset", subset), ("sweep", sweep)):
        if payload.get("family_payload_sha256") != digest:
            raise ValueError(
                f"{name}.json was built from a different family.json "
                f"({payload.get('family_payload_sha256')} != {digest}). "
                "Re-run select_hardware_subset.py / build_sweep.py first."
            )
    return family, subset, sweep


def select_rows(
    family: dict[str, Any],
    which: str,
    limit: int | None,
    only: Sequence[str] | None = None,
    n_values: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """
    The instance records a stage will run, in file order.

    Parameters
    ----------
    family:
        ``family.json``.
    which:
        ``"P"``, ``"S"`` or ``"all"``.
    limit:
        Keep only the first ``limit`` rows (a smoke run); ``None`` keeps all.
    only:
        Run exactly these instances, by name -- for re-running one that failed,
        or for timing a stratified sample.  Overrides ``--family``.
    n_values:
        Keep only these index-register widths.  Cost per instance is steeply
        increasing in ``n`` (26 s at ``n = 6`` against 4 s at ``n = 2`` for a
        three-scale ZNE run), so this is how a long stage is split into
        sittings that finish.

    Returns
    -------
    list[dict]
        The selected records.

    Raises
    ------
    ValueError
        If a requested name is not in the family file, or no instance has one
        of the requested widths.
    """
    if only:
        by_name = {row["name"]: row for row in family["instances"]}
        missing = [name for name in only if name not in by_name]
        if missing:
            raise ValueError(f"Not in family.json: {missing}")
        return [by_name[name] for name in only]
    if n_values is not None:
        available = {row["n_items"] for row in family["instances"]}
        unknown = sorted(set(n_values) - available)
        if unknown:
            raise ValueError(
                f"No instance has n_items in {unknown}; the family has "
                f"{sorted(available)}."
            )
    widths = set(n_values) if n_values is not None else None
    rows = [
        row
        for row in family["instances"]
        if (which == "all" or row["family"] == which)
        and (widths is None or row["n_items"] in widths)
    ]
    return rows[:limit] if limit else rows


def check_r_opt(rows: list[dict[str, Any]]) -> None:
    """
    Assert every recorded ``r_opt`` is the library's optimal iteration count.

    Parameters
    ----------
    rows:
        Instance records.

    Raises
    ------
    ValueError
        Naming every instance that disagrees.
    """
    bad = [
        row["name"]
        for row in rows
        if row["r_opt"] != optimal_iterations(row["n_items"], max(1, row["M"]))
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} instance(s) carry an r_opt the library disagrees with: "
            f"{bad[:5]}{'...' if len(bad) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class BackendPair:
    """
    The calibrated device and the simulator that carries its noise.

    Circuits are transpiled against ``device`` -- that is how the recorded
    ``best_seed`` values were produced, and an ``AerSimulator``'s target
    advertises extra instructions that can change basis translation -- and
    executed on ``executor``.

    Attributes
    ----------
    device:
        The calibrated backend (a fake twin, a real device, or a
        :class:`~dgssp.backends.CalibrationSnapshotBackend`).
    executor:
        The ``AerSimulator`` the circuits run on, or a noiseless simulator for
        the ideal stage.
    noise:
        ``"fake"``, ``"live"`` or ``"none"``.
    calibration_file:
        The snapshot a live run was frozen to, when there is one.
    max_threads:
        Cores the simulator was allowed, or ``None`` for Aer's default.
    """

    def __init__(
        self,
        device: Any,
        executor: Any,
        noise: str,
        calibration_file: Path | None = None,
        max_threads: int | None = None,
    ) -> None:
        self.device = device
        self.executor = executor
        self.noise = noise
        self.calibration_file = calibration_file
        self.max_threads = max_threads

    @property
    def name(self) -> str:
        """The calibrated device's name."""
        return backend_name(self.device)


def resolve_fake_backend(name: str) -> Any:
    """
    Look one bundled fake device up by name.

    Parameters
    ----------
    name:
        As ``qiskit-ibm-runtime`` spells it, e.g. ``"fake_marrakesh"``.

    Returns
    -------
    BackendV2
        The device.

    Raises
    ------
    RuntimeError
        If this release does not ship it.
    """
    available = {backend_name(b): b for b in list_fake_backends()}
    if name not in available:
        raise RuntimeError(
            f"{name!r} is not a fake device in this qiskit-ibm-runtime. "
            f"Available: {sorted(available)[:10]}..."
        )
    return available[name]


def throttle(simulator: Any, max_threads: int | None) -> Any:
    """
    Cap how many cores Aer is allowed to take, and seed it.

    Aer defaults to every core it can see and holds them for the whole run.
    On a laptop that is a thermal event, not a speed-up -- which is what
    ``--max-threads`` exists to prevent.  The counts do not depend on it: with
    a fixed ``seed_simulator`` every experiment draws from its own seeded
    stream, so one thread and sixteen give bit-identical results and a
    throttled run can be merged with an unthrottled one.

    Parameters
    ----------
    simulator:
        The ``AerSimulator``.
    max_threads:
        Cores Aer may use; ``None`` leaves Aer's own default (all of them).

    Returns
    -------
    AerSimulator
        The same object, configured.
    """
    options: dict[str, Any] = {"seed_simulator": SEED_SIMULATOR}
    if max_threads is not None:
        options["max_parallel_threads"] = int(max_threads)
    simulator.set_options(**options)
    return simulator


def build_backends(
    noise: str,
    name: str,
    *,
    calibration: Path | None,
    hardware_dir: Path,
    max_threads: int | None = None,
) -> BackendPair:
    """
    Build the (device, executor) pair a stage runs on.

    ``--noise fake`` resolves the bundled twin.  ``--noise live`` either loads
    a calibration snapshot (no service needed) or asks the service for the real
    device, derives ``NoiseModel.from_backend`` from it and freezes that
    calibration to ``paper/hardware/calibration_<date>.json``, so the same run
    is reproducible afterwards without the service.  Nothing is ever submitted
    to hardware here.

    Parameters
    ----------
    noise:
        ``"fake"`` or ``"live"``.
    name:
        Backend name.
    calibration:
        An existing snapshot to reload instead of calling the service.
    hardware_dir:
        ``paper/hardware``, where a new snapshot is written.
    max_threads:
        Cap on the simulator's thread count (see :func:`throttle`).

    Returns
    -------
    BackendPair
        Device plus noisy simulator.
    """
    from qiskit_aer import AerSimulator

    if noise == "fake":
        device = resolve_fake_backend(name)
        executor = throttle(AerSimulator.from_backend(device), max_threads)
        return BackendPair(device, executor, "fake", max_threads=max_threads)

    if calibration is not None:
        snapshot = load_noise_model(calibration)
        print(
            f"  calibration snapshot {calibration.name}: {snapshot.backend_name} "
            f"({snapshot.calibration_timestamp}), saved {snapshot.saved_at}"
        )
        return BackendPair(
            snapshot.backend,
            throttle(snapshot.simulator(), max_threads),
            "live",
            calibration,
            max_threads=max_threads,
        )

    from qiskit_aer.noise import NoiseModel

    from dgssp.runtime import get_service

    device = get_service().backend(name)
    stamp = (calibration_timestamp(device) or "unknown")[:10]
    path = hardware_dir / f"calibration_{stamp}.json"
    if path.exists():
        print(f"  snapshot {path.name} already exists; reloading it")
        snapshot = load_noise_model(path)
        return BackendPair(
            snapshot.backend,
            throttle(snapshot.simulator(), max_threads),
            "live",
            path,
            max_threads=max_threads,
        )

    model = NoiseModel.from_backend(device)
    save_noise_model(path, model, device)
    print(f"  froze {name} ({stamp}) to {path}")
    executor = throttle(AerSimulator.from_backend(device), max_threads)
    return BackendPair(device, executor, "live", path, max_threads=max_threads)


# ---------------------------------------------------------------------------
# Output layout and resume
# ---------------------------------------------------------------------------


def stage_dir(data_dir: Path, stage: str, noise: str, grid: str | None) -> Path:
    """
    Where one stage's records live.

    Parameters
    ----------
    data_dir:
        ``paper/data``.
    stage:
        Stage name.
    noise:
        ``"fake"`` or ``"live"``.
    grid:
        Scale grid, for the ZNE stage.

    Returns
    -------
    Path
        ``sim_<stage>[_<noise>][_<grid>]``.  The ideal stage carries no noise
        tag: it runs noiselessly and does not depend on the device.
    """
    parts = ["sim", stage]
    if stage != "ideal":
        parts.append(noise)
    if grid:
        parts.append(grid)
    return data_dir / "_".join(parts)


def pending(directory: Path, names: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Split names into those still to run, those already on disk, and duplicates.

    ``save_record`` never overwrites: it appends ``_001``, ``_002``, ...  A
    suffixed file in a stage directory therefore means the same instance was
    written twice, which happens when two campaigns run into the same
    directory at once -- both see the file missing, both run it, and the
    second lands beside the first.  Aggregation keys by instance name, so a
    duplicate does not double-count, but it is wasted compute and a confusing
    archive, so it is reported.  **Run one stage at a time.**

    Parameters
    ----------
    directory:
        The stage directory.
    names:
        Instance names.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(to_run, done, duplicated)``.
    """
    done = [n for n in names if (directory / f"{n}.json").exists()]
    finished = set(done)
    todo = [n for n in names if n not in finished]
    duplicated = [n for n in done if any(directory.glob(f"{n}_[0-9][0-9][0-9].json"))]
    return todo, done, duplicated


def campaign_metadata(
    *,
    stage: str,
    noise: str,
    grid: str | None,
    pair: BackendPair,
    shots: int,
    chunk_index: int,
    chunk_size: int,
    executor_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    The block every record carries under ``metrics["campaign"]``.

    Counts depend on how a job was chunked -- Aer gives each circuit in a job
    its own stream -- so the chunk a record came from is part of its
    provenance, not an implementation detail.

    Parameters
    ----------
    stage, noise, grid:
        What was run.
    pair:
        The backends used.
    shots:
        Shots per circuit (per scale, for ZNE).
    chunk_index, chunk_size:
        Which job this instance rode in, and how many circuits it held.
    executor_name:
        The backend circuits actually ran on, when it is not ``pair.executor``
        (the ideal stage runs on a noiseless simulator).
    extra:
        Stage-specific entries.

    Returns
    -------
    dict
        JSON-safe.
    """
    block: dict[str, Any] = {
        "stage": stage,
        "noise": noise,
        "grid": grid,
        "device": pair.name,
        "executor_backend": executor_name or backend_name(pair.executor),
        "calibration_file": (
            pair.calibration_file.name if pair.calibration_file else None
        ),
        "shots": shots,
        "seed_simulator": SEED_SIMULATOR,
        "max_parallel_threads": pair.max_threads,
        "chunk_index": chunk_index,
        "chunk_circuits": chunk_size,
    }
    if extra:
        block.update(extra)
    return block


class Progress:
    """
    Wall-clock progress, a measured ETA, and the budget that ends the run.

    A stage is resumable per instance, so it does not have to finish in one
    sitting: ``max_minutes`` stops it cleanly at a chunk boundary and the next
    invocation picks up exactly where it left off.  ``cooldown`` idles between
    chunks, which on a laptop is the difference between a long run and a
    thermal shutdown.

    Attributes
    ----------
    total:
        Total circuits to run.
    done:
        Circuits completed.
    max_minutes:
        Stop after a chunk once this much wall clock has passed.
    cooldown:
        Seconds to idle after each chunk.
    """

    def __init__(
        self,
        total: int,
        *,
        max_minutes: float | None = None,
        cooldown: float = 0.0,
    ) -> None:
        self.total = total
        self.done = 0
        self.max_minutes = max_minutes
        self.cooldown = cooldown
        self._started = time.time()

    @property
    def elapsed(self) -> float:
        """Seconds since the stage started."""
        return time.time() - self._started

    def update(self, circuits: int) -> str:
        """
        Record a finished chunk and describe where the stage stands.

        Parameters
        ----------
        circuits:
            Circuits in the chunk just finished.

        Returns
        -------
        str
            A line ending in the measured ETA.
        """
        self.done += circuits
        elapsed = self.elapsed
        rate = elapsed / max(1, self.done)
        remaining = max(0, self.total - self.done) * rate
        return (
            f"{self.done}/{self.total} circuits, {elapsed / 60:.1f} min elapsed, "
            f"{rate:.2f} s/circuit, ETA {remaining / 60:.1f} min"
        )

    def rest(self) -> None:
        """Idle for ``cooldown`` seconds, so the machine is not pinned flat."""
        if self.cooldown > 0:
            time.sleep(self.cooldown)

    def out_of_time(self) -> bool:
        """
        Whether the time budget is spent.

        Returns
        -------
        bool
            ``True`` once ``max_minutes`` has passed.  Checked only between
            chunks: a chunk is one job, and stopping inside one would leave an
            instance's scales half-sampled.
        """
        return self.max_minutes is not None and self.elapsed >= self.max_minutes * 60.0


# ---------------------------------------------------------------------------
# Stage: ideal
# ---------------------------------------------------------------------------


def _binomial_sigma(p: float, *shot_counts: int) -> float:
    """
    Standard error of a difference of proportions, with a one-count floor.

    The floor is what lets ``p = 1`` (``n = 2``, ``M = 1``, where the ideal
    circuit succeeds with certainty) be compared at all: the textbook
    ``sqrt(p(1-p)/N)`` is exactly zero there and would demand bit-equality of
    two finite samples.

    Parameters
    ----------
    p:
        The reference proportion.
    *shot_counts:
        Shots behind each sample being compared (one for a comparison against
        an exact value, two for a comparison of two measurements).

    Returns
    -------
    float
        The standard error.
    """
    variance = max(p * (1.0 - p), 1.0 / max(shot_counts))
    return math.sqrt(variance * sum(1.0 / n for n in shot_counts))


def corrected_sigma(nominal: float, n_tests: int) -> float:
    """
    The sigma at which ``n_tests`` tests are as safe as one at ``nominal``.

    Parameters
    ----------
    nominal:
        Per-test threshold in sigma.
    n_tests:
        How many times the test is applied.

    Returns
    -------
    float
        The Bonferroni-corrected threshold, in sigma.
    """
    if n_tests <= 1:
        return nominal
    return float(norm.isf(norm.sf(nominal) / n_tests))


def ideal_deviations(
    result: Any, row: dict[str, Any], candidate: dict[str, Any] | None, shots: int
) -> list[dict[str, Any]]:
    """
    How far one ideal result sits from theory and from the recorded ``p_ideal``.

    ``M = 0`` instances are skipped: their ``p_ideal`` is asserted, not
    measured, and the analytic formula is vacuous without a marked state.

    Parameters
    ----------
    result:
        The ``InstanceResult``.
    row:
        The instance's ``family.json`` record.
    candidate:
        Its ``hardware_subset.json`` row, if any.
    shots:
        Shots behind ``result``.

    Returns
    -------
    list[dict]
        One entry per comparison, with ``name``, ``check``, ``observed``,
        ``expected``, ``sigma`` (the deviation, in standard errors) and
        ``nominal`` (the per-test threshold).
    """
    if row["M"] < 1:
        return []

    observed = result.solution_probability
    analytic = grover_success_probability(row["n_items"], row["M"], row["r_opt"])
    out = [
        {
            "name": row["name"],
            "check": "analytic",
            "observed": observed,
            "expected": analytic,
            "sigma": abs(observed - analytic) / _binomial_sigma(analytic, shots),
            "nominal": ANALYTIC_SIGMA,
        }
    ]
    if candidate is not None and candidate.get("p_ideal") is not None:
        recorded = float(candidate["p_ideal"])
        out.append(
            {
                "name": row["name"],
                "check": "recorded",
                "observed": observed,
                "expected": recorded,
                "sigma": abs(observed - recorded)
                / _binomial_sigma(recorded, shots, RECORDED_IDEAL_SHOTS),
                "nominal": RECORDED_SIGMA,
            }
        )
    return out


def judge_ideal(
    deviations: list[dict[str, Any]], *, strict: bool
) -> tuple[bool, list[str]]:
    """
    Decide whether the ideal stage passed, and say why in either case.

    Parameters
    ----------
    deviations:
        Every comparison made (see :func:`ideal_deviations`).
    strict:
        Apply the nominal per-instance thresholds literally, with no
        multiplicity correction.

    Returns
    -------
    tuple[bool, list[str]]
        ``(passed, report)``.  The report is printed whether or not the stage
        passed: the exceedance counts are part of the evidence.
    """
    report: list[str] = []
    passed = True

    for check in ("analytic", "recorded"):
        rows = [d for d in deviations if d["check"] == check]
        if not rows:
            continue
        nominal = rows[0]["nominal"]
        n_tests = len(rows)
        alpha = 2.0 * float(norm.sf(nominal))
        limit = nominal if strict else corrected_sigma(nominal, n_tests)
        over = sorted(
            (d for d in rows if d["sigma"] > nominal),
            key=lambda d: -d["sigma"],
        )
        expected = alpha * n_tests
        # How many nominal exceedances would still be unremarkable?
        allowed = int(binom.isf(COUNT_TAIL_ALPHA, n_tests, alpha))

        report.append(
            f"  {check}: {n_tests} comparisons, worst "
            f"{max(d['sigma'] for d in rows):.2f} sigma; "
            f"{len(over)} over the nominal {nominal:.0f} sigma "
            f"(chance predicts {expected:.2f}, up to {allowed} is unremarkable); "
            f"stop threshold {limit:.2f} sigma"
        )
        for d in over[:5]:
            report.append(
                f"      {d['name']}: {d['observed']:.5f} vs {d['expected']:.5f} "
                f"= {d['sigma']:.2f} sigma"
            )
        if len(over) > 5:
            report.append(f"      ... and {len(over) - 5} more")

        hard = [d for d in rows if d["sigma"] > limit]
        if hard:
            passed = False
            report.append(
                f"    FAIL: {len(hard)} comparison(s) past {limit:.2f} sigma -- "
                "that is not chance."
            )
        if not strict and len(over) > allowed:
            passed = False
            report.append(
                f"    FAIL: {len(over)} nominal exceedances where at most "
                f"{allowed} is plausible -- that is a systematic bias, not noise."
            )
    return passed, report


def run_unmitigated_stage(
    stage: str,
    rows: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    pair: BackendPair,
    directory: Path,
    *,
    shots: int,
    chunk: int,
    strict_acceptance: bool = False,
    max_minutes: float | None = None,
    cooldown: float = 0.0,
) -> int:
    """
    Run the ideal or noisy stage: one circuit per instance, chunked into jobs.

    Parameters
    ----------
    stage:
        ``"ideal"`` or ``"noisy"``.
    rows:
        Instance records still to run.
    candidates:
        ``hardware_subset.json`` rows, keyed by name.
    pair:
        Device and executor.
    directory:
        Where records are written.
    shots:
        Shots per circuit.
    chunk:
        Circuits (here: instances) per job.
    strict_acceptance:
        Apply the nominal acceptance thresholds per instance, uncorrected for
        the number of instances they are applied to.
    max_minutes:
        Stop cleanly after the chunk that crosses this much wall clock, and
        report what is left.  Ignored for the ideal stage, which has to judge
        every instance together before it may write any of them.
    cooldown:
        Seconds to idle between chunks.

    Returns
    -------
    int
        ``0`` on success, ``1`` if the ideal stage failed its acceptance check
        -- in which case nothing from the stage is written at all.
    """
    is_ideal = stage == "ideal"
    if is_ideal and max_minutes is not None:
        print(
            "  ! --max-minutes does not apply to the ideal stage: it is the "
            "acceptance gate, and it writes nothing until every instance has "
            "been judged together. It takes about 1.5 minutes."
        )
        max_minutes = None
    config = BatchConfig(
        executor="ideal" if is_ideal else "noisy",
        shots=shots,
        backend=None if is_ideal else pair.executor,
        seed_simulator=SEED_SIMULATOR,
        seed_transpiler=SEED_TRANSPILER,
        optimization_level=1 if is_ideal else 3,
        bootstrap_samples=BOOTSTRAP_B,
        bootstrap_seed=BOOTSTRAP_SEED,
        compute_reference=True,
    )
    progress = Progress(len(rows), max_minutes=max_minutes, cooldown=cooldown)
    pending_writes: list[tuple[dict[str, Any], Any, int, int, int, str | None]] = []
    deviations: list[dict[str, Any]] = []
    mismatches: list[str] = []

    def write(
        row: dict[str, Any],
        result: Any,
        chunk_index: int,
        chunk_size: int,
        offset: int,
        executed_on: str | None,
    ) -> None:
        """Stamp one result and save it."""
        record = Record.from_instance_result(
            result,
            executor=stage,
            mitigation_mode="none",
            shots=shots,
            backend=None if is_ideal else pair.device,
            extra_metrics={
                "campaign": campaign_metadata(
                    stage=stage,
                    noise="none" if is_ideal else pair.noise,
                    grid=None,
                    pair=pair,
                    shots=shots,
                    chunk_index=chunk_index,
                    chunk_size=chunk_size,
                    executor_name=executed_on,
                    extra={
                        "family": row["family"],
                        "m_class": row["m_class"],
                        "position_in_chunk": offset,
                        "analytic_p_sol": grover_success_probability(
                            row["n_items"], row["M"], row["r_opt"]
                        ),
                    },
                )
            },
        )
        save_record(directory / f"{row['name']}.json", record)

    for index in range(0, len(rows), chunk):
        batch_rows = rows[index : index + chunk]
        instances = [
            SubsetSumInstance(
                items=list(r["items"]), target=r["target"], name=r["name"]
            )
            for r in batch_rows
        ]
        results = run_batch(instances, config).results
        for result in results:
            result.seed_source = "fixed_seed"
        executed_on = results[0].backend_name if results else None

        for offset, (row, result) in enumerate(zip(batch_rows, results, strict=True)):
            if sorted(result.solution_bitstrings) != sorted(row["solution_bitstrings"]):
                mismatches.append(
                    f"{row['name']}: ground truth disagrees with family.json "
                    f"({sorted(result.solution_bitstrings)} != "
                    f"{sorted(row['solution_bitstrings'])})"
                )
            if result.iterations != row["r_opt"]:
                mismatches.append(
                    f"{row['name']}: ran r={result.iterations}, file says "
                    f"r_opt={row['r_opt']}"
                )
            if is_ideal:
                deviations += ideal_deviations(
                    result, row, candidates.get(row["name"]), shots
                )
                # The ideal stage is the gate: nothing is written until every
                # instance has been seen and judged together.
                pending_writes.append(
                    (row, result, index // chunk, len(batch_rows), offset, executed_on)
                )
            else:
                write(row, result, index // chunk, len(batch_rows), offset, executed_on)

        print(f"  chunk {index // chunk}: {progress.update(len(batch_rows))}")
        remaining = len(rows) - (index + len(batch_rows))
        if progress.out_of_time() and remaining:
            print(
                f"\n  time budget spent ({max_minutes} min); {remaining} instance(s) "
                "left. Re-run the same command to continue where this stopped."
            )
            break
        if remaining:
            progress.rest()

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} instance(s) do not match family.json; "
              "nothing was written.\n")
        for message in mismatches[:20]:
            print(f"  {message}")
        return 1

    if is_ideal:
        passed, report = judge_ideal(deviations, strict=strict_acceptance)
        print("\nacceptance:")
        for line in report:
            print(line)
        if not passed:
            print("\nFAIL: the ideal stage did not pass; nothing was written.")
            return 1
        for args_tuple in pending_writes:
            write(*args_tuple)
        print(f"  wrote {len(pending_writes)} record(s)")
    return 0


# ---------------------------------------------------------------------------
# Stage: zne
# ---------------------------------------------------------------------------


def scales_for(
    row: dict[str, Any], candidate: dict[str, Any] | None, grid: str, shots: int
) -> tuple[list[float], str]:
    """
    This instance's nominal scale grid, and where it came from.

    Parameters
    ----------
    row:
        The ``family.json`` record.
    candidate:
        Its ``hardware_subset.json`` row, which carries ``survival``,
        ``p_ideal`` and ``p_uniform``.
    grid:
        ``"odd"``, ``"planned"`` or ``"odd7"``.
    shots:
        Planned shots per scale, which set the planner's noise floor.

    Returns
    -------
    tuple[list[float], str]
        The scales and a source tag (``"odd"``, ``"odd7"``, ``"planned"`` or
        ``"planned_floor"`` when the planner returned a single scale and
        :data:`PATHOLOGY_GRID` was used instead).

    Raises
    ------
    ValueError
        If ``"planned"`` is asked for without a candidate row: the plan needs
        the recorded survival, which only that file has.
    """
    if grid == "odd":
        return list(ODD_SCALES), "odd"
    if grid == "odd7":
        return list(ODD7_SCALES), "odd7"

    if candidate is None:
        raise ValueError(
            f"{row['name']}: the planned grid needs the survival recorded in "
            "hardware_subset.json, and this instance has no candidate row."
        )
    survival = candidate.get("survival")
    if not survival:
        raise ValueError(
            f"{row['name']}: hardware_subset.json records no survival, so no "
            "grid can be planned. Re-run select_hardware_subset.py."
        )
    p_uniform = float(candidate["p_uniform"])
    planned = plan_scales(
        float(survival),
        float(candidate["p_ideal"]) - p_uniform,
        shots,
        PLAN_CANDIDATES,
        PLAN_K,
        p_uniform=p_uniform,
        n_points=PLAN_POINTS,
    )
    if len(planned) < 2:
        # No signal to preserve (every M = 0 instance), so there is nothing to
        # extrapolate -- but the pathology diagnostic still needs scales.
        return list(PATHOLOGY_GRID), "planned_floor"
    return planned, "planned"


def chunk_by_circuits(
    prepared: list[PreparedInstance], limit: int, *, include_mitiq: bool
) -> list[list[PreparedInstance]]:
    """
    Pack instances into jobs of at most ``limit`` circuits.

    Parameters
    ----------
    prepared:
        The transpiled instances, each knowing how many circuits it submits.
    limit:
        Circuits per job.
    include_mitiq:
        Whether Mitiq's folds ride along (they do not add a circuit at
        ``lambda = 1``).

    Returns
    -------
    list[list[PreparedInstance]]
        Chunks in input order.  An instance that alone exceeds ``limit`` gets
        a chunk of its own rather than being split: a fold set must stay in one
        job, or the scales stop being comparable.
    """
    chunks: list[list[PreparedInstance]] = []
    current: list[PreparedInstance] = []
    size = 0
    for item in prepared:
        cost = item.n_circuits(include_mitiq=include_mitiq)
        if current and size + cost > limit:
            chunks.append(current)
            current, size = [], 0
        current.append(item)
        size += cost
    if current:
        chunks.append(current)
    return chunks


def run_zne_stage(
    rows: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    pair: BackendPair,
    directory: Path,
    *,
    grid: str,
    shots: int,
    chunk: int,
    include_mitiq: bool,
    use_recorded_seeds: bool,
    max_minutes: float | None = None,
    cooldown: float = 0.0,
) -> int:
    """
    Run the ZNE stage: transpile once, fold, one job per chunk, analyse offline.

    Parameters
    ----------
    rows:
        Instance records still to run.
    candidates:
        ``hardware_subset.json`` rows, keyed by name.
    pair:
        Device (transpilation target) and executor.
    directory:
        Where records are written.
    grid:
        Scale grid name.
    shots:
        Shots per scale.
    chunk:
        Circuits per job.
    include_mitiq:
        Submit Mitiq's folds of the same ISA circuits.
    use_recorded_seeds:
        Reuse ``best_seed``; false when the backend is not the one the subset
        file was built against.
    max_minutes:
        Stop cleanly after the chunk that crosses this much wall clock, and
        report what is left.  The transpilation pass runs first and is not
        counted -- it is a few seconds for the whole family.
    cooldown:
        Seconds to idle between chunks.

    Returns
    -------
    int
        ``0``.
    """
    cfg = ZNESamplingConfig(
        scales=list(ODD_SCALES),
        shots_per_scale=shots,
        method=ZNE_METHOD,
        mode=ZNE_MODE,
        renorm=ZNE_RENORM,
        asymptote="p_uniform",
        fold_seed=FOLD_SEED,
        n_bootstrap=BOOTSTRAP_B,
        bootstrap_seed=BOOTSTRAP_SEED,
        seed_min=SEED_MIN,
        seed_max=SEED_MAX,
        optimization_level=3,
        seed_simulator=SEED_SIMULATOR,
    )

    print(f"  transpiling {len(rows)} instance(s) "
          f"({'recorded seeds' if use_recorded_seeds else f'SABRE {SEED_MIN}..{SEED_MAX}'})")
    started = time.time()
    prepared: list[PreparedInstance] = []
    sources: dict[str, str] = {}
    for row in rows:
        candidate = candidates.get(row["name"])
        scales, source = scales_for(row, candidate, grid, shots)
        prepared.append(
            prepare_from_records(
                row,
                candidate,
                pair.device,
                cfg,
                scales=scales,
                use_recorded_seed=use_recorded_seeds,
            )
        )
        sources[row["name"]] = source
    print(f"  transpiled in {time.time() - started:.1f}s")

    by_name = {row["name"]: row for row in rows}
    chunks = chunk_by_circuits(prepared, chunk, include_mitiq=include_mitiq)
    total_circuits = sum(
        item.n_circuits(include_mitiq=include_mitiq) for item in prepared
    )
    progress = Progress(total_circuits, max_minutes=max_minutes, cooldown=cooldown)
    done_instances = 0

    for chunk_index, group in enumerate(chunks):
        batch = run_zne_batch(
            [item.zne_item() for item in group],
            pair.executor,
            cfg,
            include_mitiq=include_mitiq,
        )
        for item, run in zip(group, batch.items, strict=True):
            row = by_name[item.name]
            started = time.time()
            result = build_instance_result(
                item,
                run,
                pair.device,
                cfg,
                reference=exact_index_distribution(item.instance, item.iterations),
                bootstrap_samples=BOOTSTRAP_B,
                bootstrap_seed=BOOTSTRAP_SEED,
                methods=ZNE_METHODS,
                job_id=batch.job_id,
                elapsed_s=time.time() - started,
            )
            record = Record.from_instance_result(
                result,
                executor="optimized",
                shots=shots,
                backend=pair.device,
                extra_metrics={
                    "campaign": campaign_metadata(
                        stage="zne",
                        noise=pair.noise,
                        grid=grid,
                        pair=pair,
                        shots=shots,
                        chunk_index=chunk_index,
                        chunk_size=batch.n_circuits,
                        extra={
                            "family": row["family"],
                            "m_class": row["m_class"],
                            "grid_source": sources[item.name],
                            "nominal_scales": list(item.scales),
                            "seed_source": item.seed_source,
                            "methods": list(ZNE_METHODS),
                            "include_mitiq": include_mitiq,
                            "analytic_p_sol": grover_success_probability(
                                row["n_items"], row["M"], row["r_opt"]
                            ),
                        },
                    )
                },
            )
            save_record(directory / f"{item.name}.json", record)
        done_instances += len(group)
        print(f"  chunk {chunk_index}: {progress.update(batch.n_circuits)}")
        remaining = len(prepared) - done_instances
        if progress.out_of_time() and remaining:
            print(
                f"\n  time budget spent ({max_minutes} min); {remaining} instance(s) "
                "left. Re-run the same command to continue where this stopped."
            )
            break
        if remaining:
            progress.rest()
    return 0


# ---------------------------------------------------------------------------
# Stage: sweep
# ---------------------------------------------------------------------------


def run_sweep_stage(
    sweep: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    pair: BackendPair,
    directory: Path,
    *,
    include_mitiq: bool,
    use_recorded_seeds: bool,
) -> int:
    """
    Run the Grover-iteration sweep for both sweep instances, three ways.

    Every ``r``-variant of one instance (and, under ZNE, every fold of every
    variant) is one job, as :func:`dgssp.experiments.run_iteration_sweep`
    guarantees.  The ``r_opt`` variant reuses the recorded ``best_seed``; the
    others search the SABRE range, because no seed was ever recorded for them.

    Parameters
    ----------
    sweep:
        ``sweep.json``.
    candidates:
        ``hardware_subset.json`` rows, keyed by name.
    pair:
        Device and executor.
    directory:
        Where the sweeps are written.
    include_mitiq:
        Run Mitiq's folds in the ZNE executor.
    use_recorded_seeds:
        Reuse the ``r_opt`` ``best_seed``.

    Returns
    -------
    int
        ``0``.
    """
    cfg = ZNESamplingConfig(
        scales=list(ODD_SCALES),
        shots_per_scale=ZNE_SHOTS,
        method=ZNE_METHOD,
        mode=ZNE_MODE,
        renorm=ZNE_RENORM,
        asymptote="p_uniform",
        fold_seed=FOLD_SEED,
        n_bootstrap=BOOTSTRAP_B,
        bootstrap_seed=BOOTSTRAP_SEED,
        seed_simulator=SEED_SIMULATOR,
    )

    for row in sweep["instances"]:
        instance = SubsetSumInstance(
            items=list(row["items"]), target=row["target"], name=row["name"]
        )
        candidate = candidates.get(row["name"]) or {}
        seeds = (
            {row["r_opt"]: candidate["best_seed"]}
            if use_recorded_seeds and candidate.get("best_seed") is not None
            else {}
        )
        for executor in ("ideal", "noisy", "zne"):
            path = directory / f"{row['name']}_{executor}.json"
            if path.exists():
                print(f"  {path.name}: present, skipping")
                continue
            started = time.time()
            payload = run_iteration_sweep(
                instance,
                [int(r) for r in row["iterations"]],
                None if executor == "ideal" else pair.executor,
                executor=executor,
                zne_config=cfg if executor == "zne" else None,
                shots=IDEAL_SHOTS if executor == "ideal" else NOISY_SHOTS,
                best_seeds=seeds,
                seed_min=SEED_MIN,
                seed_max=SEED_MAX,
                optimization_level=3,
                include_mitiq=include_mitiq and executor == "zne",
                seed_simulator=SEED_SIMULATOR,
                bootstrap_samples=BOOTSTRAP_B,
                bootstrap_seed=BOOTSTRAP_SEED,
                methods=ZNE_METHODS if executor == "zne" else None,
            )
            payload["schema"] = "dgssp/paper/sim-sweep/1"
            payload["campaign"] = campaign_metadata(
                stage="sweep",
                noise="none" if executor == "ideal" else pair.noise,
                grid=None,
                pair=pair,
                shots=ZNE_SHOTS if executor == "zne" else IDEAL_SHOTS,
                chunk_index=0,
                chunk_size=len(row["iterations"]),
                extra={
                    "sweep_executor": executor,
                    "r_opt": row["r_opt"],
                    "best_seeds": {str(k): v for k, v in seeds.items()},
                    "family_payload_sha256": sweep["family_payload_sha256"],
                },
            )
            payload["device_calibration"] = calibration_timestamp(pair.device)
            save_record(path, payload)
            print(f"  {path.name}: {time.time() - started:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv:
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="all")
    parser.add_argument("--noise", choices=("fake", "live"), default="fake")
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend name; defaults to hardware_subset.json's backend.",
    )
    parser.add_argument("--grid", choices=GRIDS, default="planned")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Reload a saved calibration snapshot instead of calling the service.",
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run exactly these instances, by name (overrides --family).",
    )
    parser.add_argument(
        "--n",
        dest="n_values",
        nargs="+",
        type=int,
        default=None,
        help="Only instances of these index-register widths. Cost per instance "
        "rises steeply with n, so this is how a long stage is split.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Cap the simulator's thread count. Aer takes every core by "
        "default and holds it for the whole run; the counts do not depend on "
        "this, so a throttled run merges with an unthrottled one.",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Stop cleanly after the chunk that crosses this much wall clock. "
        "Re-running continues where it stopped.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="Seconds to idle between chunks, to let the machine cool.",
    )
    parser.add_argument("--no-mitiq", action="store_true")
    parser.add_argument(
        "--strict-acceptance",
        action="store_true",
        help="Apply the ideal stage's sigma thresholds per instance, without "
        "correcting for the number of instances tested.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES_DIR)
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Run one stage of the campaign.

    Parameters
    ----------
    argv:
        Command line; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when a check failed.
    """
    args = parse_args(argv)
    family, subset, sweep = load_inputs(args.instances_dir)
    candidates = {row["name"]: row for row in subset["candidates"]}

    backend = args.backend or subset["backend"]
    matches_subset = backend == subset["backend"]
    pair = build_backends(
        args.noise,
        backend,
        calibration=args.calibration,
        hardware_dir=args.hardware_dir,
        max_threads=args.max_threads,
    )

    from dgssp.mitigation import mitiq_is_available

    include_mitiq = (not args.no_mitiq) and mitiq_is_available()

    print(f"stage={args.stage} family={args.family} noise={args.noise} "
          f"backend={pair.name} calibration={calibration_timestamp(pair.device)}")
    cores = os.cpu_count() or 1
    print(
        f"  threads={args.max_threads if args.max_threads else f'all ({cores})'}"
        f"  cooldown={args.cooldown:g}s"
        f"  budget={f'{args.max_minutes:g} min' if args.max_minutes else 'none'}"
        f"  n={args.n_values if args.n_values else 'all'}"
    )
    if args.stage == "zne":
        print(f"  grid={args.grid} mitiq={'on' if include_mitiq else 'off'} "
              f"methods={list(ZNE_METHODS)}")
    if not matches_subset:
        print(
            f"  ! {backend} is not hardware_subset.json's backend "
            f"({subset['backend']}): the recorded best_seed values do not "
            "apply and every layout will be searched again."
        )

    if args.stage == "sweep":
        directory = stage_dir(args.data_dir, "sweep", pair.noise, None)
        directory.mkdir(parents=True, exist_ok=True)
        return run_sweep_stage(
            sweep,
            candidates,
            pair,
            directory,
            include_mitiq=include_mitiq,
            use_recorded_seeds=matches_subset,
        )

    rows = select_rows(family, args.family, args.limit, args.only, args.n_values)
    if args.stage == "zne" and args.grid == "odd7":
        rows = [
            r for r in rows if r["family"] == ODD7_FAMILY and r["n_items"] <= ODD7_MAX_N
        ]
        print(f"  odd7 covers Family {ODD7_FAMILY}, n <= {ODD7_MAX_N}: "
              f"{len(rows)} instance(s)")
    check_r_opt(rows)

    grids = (
        ["odd", "planned"]
        if (args.stage == "zne" and args.grid == "both")
        else [args.grid if args.stage == "zne" else None]
    )

    status = 0
    for grid in grids:
        directory = stage_dir(args.data_dir, args.stage, pair.noise, grid)
        directory.mkdir(parents=True, exist_ok=True)
        todo_names, done, duplicated = pending(directory, [r["name"] for r in rows])
        todo = [r for r in rows if r["name"] in set(todo_names)]
        label = f"{args.stage}{f' [{grid}]' if grid else ''}"
        print(f"\n{label} -> {directory.name}: "
              f"{len(todo)} to run, {len(done)} already present")
        if duplicated:
            print(
                f"  ! {len(duplicated)} instance(s) have a duplicate record "
                f"({duplicated[:3]}): two runs wrote into this directory at "
                "once. Aggregation keys by name so nothing is double-counted, "
                "but run one stage at a time."
            )
        if not todo:
            continue

        if args.stage in ("ideal", "noisy"):
            shots = args.shots or (
                IDEAL_SHOTS if args.stage == "ideal" else NOISY_SHOTS
            )
            status = run_unmitigated_stage(
                args.stage,
                todo,
                candidates,
                pair,
                directory,
                shots=shots,
                chunk=args.chunk,
                strict_acceptance=args.strict_acceptance,
                max_minutes=args.max_minutes,
                cooldown=args.cooldown,
            )
        else:
            status = run_zne_stage(
                todo,
                candidates,
                pair,
                directory,
                grid=grid or "planned",
                shots=args.shots or ZNE_SHOTS,
                chunk=args.chunk,
                include_mitiq=include_mitiq,
                use_recorded_seeds=matches_subset,
                max_minutes=args.max_minutes,
                cooldown=args.cooldown,
            )
        if status:
            return status

    print("\nOK")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
