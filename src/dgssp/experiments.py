"""
experiments.py

Multi-instance batch runner: the module you use to produce thesis figures.

Given a list of Subset Sum instances it will, in one call:

1. solve each instance classically (exhaustive DP) to obtain **ground truth**
   -- the set of correct bitstrings and their count ``M``;
2. use ``M`` to pick the optimal Grover iteration count per instance;
3. build and transpile one circuit per instance;
4. submit **all of them as a single job** via
   :func:`dgssp.runtime.sample_counts`, so a sweep costs one queue wait rather
   than one per instance;
5. decode each result against its ground truth and return structured,
   serialisable records.

Because ground truth is computed inside the runner, every result carries a
measured ``solution_probability`` -- success is quantified automatically across
the whole sweep, which is what makes "P(solution) vs n_items", "vs backend" or
"vs ZNE on/off" a one-liner.

The ``"optimized"`` executor cannot batch across instances (each instance gets
its own best backend and its own layout), so it loops -- but it still batches
the ZNE noise scales within each instance.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .backends import (
    BackendErrorMetrics,
    BackendLike,
    BackendSelectionConfig,
    build_all_backends,
    build_ideal_aer_backend,
    compute_accumulated_errors,
)
from .decoding import (
    counts_to_probs,
    distribution_solution_probability,
    exact_solution_bitstrings,
    solution_probability,
)
from .execution import select_best_backend_by_error
from .instance import SubsetSumInstance, random_instance
from .mitigation import (
    ZNESamplingConfig,
    mitiq_zne_solution_probability,
    transpile_once,
    zne_mitigated_distribution,
)
from .runtime import backend_name, execution_mode, sample_counts
from .solvers import DGConfig, DGSSPSolver, DPConfig, DPSSPSolver, optimal_iterations

ExecutorName = Literal["ideal", "noisy", "optimized"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """
    Settings for a multi-instance run.

    Attributes
    ----------
    dg_config:
        Solver configuration.  With the default ``iterations="auto"`` each
        instance gets its own optimal iteration count from the DP ground truth.
    executor:
        ``"ideal"`` (noiseless Aer), ``"noisy"`` (one shared noisy backend) or
        ``"optimized"`` (per-instance best backend + ZNE).
    shots:
        Shots per instance.
    backend:
        An explicit backend for the ``"noisy"`` executor.
    backend_config:
        Used to build a backend bundle when none is given explicitly.
    service:
        An existing ``QiskitRuntimeService``, required to reach real hardware.
    zne_config:
        ZNE settings for the ``"optimized"`` executor.  Defaults are supplied
        if omitted.
    run_mitiq_baseline:
        Also compute Mitiq's ZNE of ``P(solution)`` for comparison.  Requires
        the ``dgssp[mitiq]`` extra.
    seed_simulator:
        Simulator seed for reproducibility.
    seed_transpiler:
        Transpiler seed for the ``"ideal"``/``"noisy"`` paths.
    optimization_level:
        Preset optimization level for those same paths.
    mode:
        Execution mode passed to :func:`dgssp.runtime.execution_mode`.
    """

    dg_config: DGConfig = field(default_factory=DGConfig)
    executor: ExecutorName = "ideal"
    shots: int = 10_000
    backend: Any | None = None
    backend_config: BackendSelectionConfig | None = None
    service: Any | None = None
    zne_config: ZNESamplingConfig | None = None
    run_mitiq_baseline: bool = False
    seed_simulator: int | None = None
    seed_transpiler: int | None = None
    optimization_level: int = 1
    mode: str = "auto"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class InstanceResult:
    """
    Everything measured for one instance in a batch.

    Attributes
    ----------
    instance:
        The instance that was run.
    n_qubits:
        Width of the logical circuit (index + sum registers).
    iterations:
        Grover iterations actually applied.
    solution_bitstrings:
        Correct outcomes, from the classical DP solver.
    num_solutions:
        ``M``, the number of correct outcomes.
    counts:
        Raw measurement counts.
    distribution:
        Normalised counts.
    solution_probability:
        Measured probability of a correct outcome.
    mitigated_distribution:
        ZNE-mitigated distribution, when mitigation ran.
    mitigated_solution_probability:
        ``P(solution)`` under the mitigated distribution.
    mitiq_solution_probability:
        ``P(solution)`` from Mitiq's ZNE, when the baseline ran.
    error_metrics:
        Calibration error accumulation for the executed circuit.
    backend_name:
        Backend the circuit ran on.
    depth, two_qubit_depth:
        Depth of the executed circuit, overall and counting only two-qubit
        gates.
    elapsed_s:
        Wall-clock seconds spent on this instance's post-processing.
    """

    instance: SubsetSumInstance
    n_qubits: int
    iterations: int
    solution_bitstrings: list[str]
    num_solutions: int
    counts: dict[str, int]
    distribution: dict[str, float]
    solution_probability: float
    mitigated_distribution: dict[str, float] | None = None
    mitigated_solution_probability: float | None = None
    mitiq_solution_probability: float | None = None
    error_metrics: BackendErrorMetrics | None = None
    backend_name: str | None = None
    depth: int | None = None
    two_qubit_depth: int | None = None
    elapsed_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Flatten the record into a JSON-serialisable dictionary.

        Returns
        -------
        dict
            Nested dataclasses are expanded; the instance appears as its
            items, target and name.
        """
        return {
            "name": self.instance.name,
            "items": list(self.instance.items),
            "target": self.instance.target,
            "n_items": self.instance.n_items,
            "n_qubits": self.n_qubits,
            "iterations": self.iterations,
            "solution_bitstrings": list(self.solution_bitstrings),
            "num_solutions": self.num_solutions,
            "counts": dict(self.counts),
            "distribution": dict(self.distribution),
            "solution_probability": self.solution_probability,
            "mitigated_distribution": self.mitigated_distribution,
            "mitigated_solution_probability": self.mitigated_solution_probability,
            "mitiq_solution_probability": self.mitiq_solution_probability,
            "error_metrics": (
                self.error_metrics.to_dict() if self.error_metrics else None
            ),
            "backend_name": self.backend_name,
            "depth": self.depth,
            "two_qubit_depth": self.two_qubit_depth,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class BatchResult:
    """
    All instance results from one :func:`run_batch` call, plus the config.

    Attributes
    ----------
    results:
        One :class:`InstanceResult` per input instance, in input order.
    config:
        The :class:`BatchConfig` used.
    job_ids:
        Runtime job ids, for recovering or auditing a hardware run.
    """

    results: list[InstanceResult]
    config: BatchConfig
    job_ids: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of instance results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over the instance results."""
        return iter(self.results)

    def to_json(self, path: str | Path) -> Path:
        """
        Serialise the whole batch to a JSON file.

        Parameters
        ----------
        path:
            Destination path; parent directories are created.

        Returns
        -------
        Path
            The path written.
        """
        payload = {
            "config": _config_to_dict(self.config),
            "job_ids": list(self.job_ids),
            "results": [r.to_dict() for r in self.results],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return p

    def to_dataframe(self):
        """
        Return a tidy one-row-per-instance table.

        Returns
        -------
        pandas.DataFrame
            Scalar columns only (the counts and distribution dictionaries are
            omitted) so the result plots directly.

        Raises
        ------
        ImportError
            If pandas is not installed; install with ``pip install
            "dgssp[data]"``.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised without pandas
            raise ImportError(
                'pandas is required for to_dataframe(); install with: pip install "dgssp[data]"'
            ) from exc

        rows = []
        for r in self.results:
            row = r.to_dict()
            row.pop("counts", None)
            row.pop("distribution", None)
            row.pop("mitigated_distribution", None)
            metrics = row.pop("error_metrics", None) or {}
            row.update({f"err_{k}": v for k, v in metrics.items()})
            row["solution_bitstrings"] = ",".join(row["solution_bitstrings"])
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """
        Aggregate statistics across the batch.

        Returns
        -------
        dict
            Instance count, mean/min/max solution probability, the success
            rate (fraction of instances whose most likely outcome is correct)
            and the mean mitigated probability when mitigation ran.
        """
        if not self.results:
            return {"n_instances": 0}

        probs = [r.solution_probability for r in self.results]
        successes = sum(
            1
            for r in self.results
            if r.counts
            and max(r.counts, key=lambda b: r.counts[b]) in r.solution_bitstrings
        )
        mitigated = [
            r.mitigated_solution_probability
            for r in self.results
            if r.mitigated_solution_probability is not None
        ]

        out: dict[str, Any] = {
            "n_instances": len(self.results),
            "executor": self.config.executor,
            "shots": self.config.shots,
            "mean_solution_probability": sum(probs) / len(probs),
            "min_solution_probability": min(probs),
            "max_solution_probability": max(probs),
            "top_outcome_success_rate": successes / len(self.results),
        }
        if mitigated:
            out["mean_mitigated_solution_probability"] = sum(mitigated) / len(mitigated)
        return out


def _config_to_dict(config: BatchConfig) -> dict[str, Any]:
    """Serialise a BatchConfig, replacing live backend objects with names."""
    data = {
        "executor": config.executor,
        "shots": config.shots,
        "run_mitiq_baseline": config.run_mitiq_baseline,
        "seed_simulator": config.seed_simulator,
        "seed_transpiler": config.seed_transpiler,
        "optimization_level": config.optimization_level,
        "mode": config.mode,
        "dg_config": asdict(config.dg_config),
        "backend": backend_name(config.backend) if config.backend else None,
    }
    if config.zne_config is not None:
        zne = asdict(config.zne_config)
        zne["scales"] = list(zne["scales"])
        data["zne_config"] = zne
    if config.backend_config is not None:
        data["backend_config"] = asdict(config.backend_config)
    return data


# ---------------------------------------------------------------------------
# Circuit preparation
# ---------------------------------------------------------------------------


@dataclass
class _InstanceMeta:
    """Per-instance bookkeeping carried between the build and assemble phases."""

    instance: SubsetSumInstance
    solution_bitstrings: list[str]
    iterations: int
    circuit: QuantumCircuit


def prepare_instances(
    instances: Sequence[SubsetSumInstance], config: BatchConfig
) -> list[_InstanceMeta]:
    """
    Solve each instance classically and build its D-G circuit.

    Parameters
    ----------
    instances:
        The instances to prepare.
    config:
        Supplies the solver configuration; when ``iterations == "auto"`` the
        DP solution count drives :func:`dgssp.solvers.optimal_iterations`.

    Returns
    -------
    list[_InstanceMeta]
        One record per instance with its ground-truth bitstrings, iteration
        count and logical circuit.
    """
    dp = DPSSPSolver(DPConfig(enumerate_all=True))
    metas: list[_InstanceMeta] = []

    for inst in instances:
        sol_bits = exact_solution_bitstrings(inst, dp.solve(inst))
        num_solutions = max(1, len(sol_bits))

        if config.dg_config.iterations == "auto":
            iterations = optimal_iterations(inst.n_items, num_solutions)
        else:
            iterations = int(config.dg_config.iterations)

        cfg = replace(config.dg_config, iterations=iterations)
        qc = DGSSPSolver(cfg).build_circuit(inst, num_solutions=num_solutions)

        metas.append(
            _InstanceMeta(
                instance=inst,
                solution_bitstrings=sol_bits,
                iterations=iterations,
                circuit=qc,
            )
        )

    return metas


def _resolve_backend(config: BatchConfig) -> BackendLike:
    """Choose the backend for the ideal/noisy paths."""
    if config.executor == "ideal":
        return build_ideal_aer_backend(seed_simulator=config.seed_simulator)

    if config.backend is not None:
        return config.backend

    if config.backend_config is None:
        raise ValueError(
            "BatchConfig.backend or BatchConfig.backend_config must be set for "
            "the 'noisy' executor."
        )

    bundle = build_all_backends(config.backend_config, service=config.service)
    pool = bundle["real_backends"] or bundle["fake_backends"]
    if not pool:
        raise RuntimeError(
            "No noisy backends available. Pass a QiskitRuntimeService via "
            "BatchConfig.service, or an explicit BatchConfig.backend."
        )
    return pool[0]


def _two_qubit_depth(qc: QuantumCircuit) -> int:
    """Depth counting only two-qubit operations."""
    return qc.depth(lambda instr: instr.operation.num_qubits == 2)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_batch(
    instances: Sequence[SubsetSumInstance], config: BatchConfig | None = None
) -> BatchResult:
    """
    Build, execute and decode a whole set of instances.

    For the ``"ideal"`` and ``"noisy"`` executors every circuit is submitted in
    a **single** job.  For ``"optimized"`` the runner loops per instance
    (each needs its own backend and layout) but still batches the ZNE scales.

    Parameters
    ----------
    instances:
        The instances to run.
    config:
        Batch settings; defaults to ``BatchConfig()`` (ideal executor).

    Returns
    -------
    BatchResult
        One :class:`InstanceResult` per instance, plus the config and job ids.

    Raises
    ------
    ValueError
        If ``instances`` is empty or the executor name is unknown.
    """
    cfg = config or BatchConfig()
    if not instances:
        raise ValueError("run_batch requires at least one instance.")

    metas = prepare_instances(instances, cfg)

    if cfg.executor in ("ideal", "noisy"):
        return _run_batched(metas, cfg)
    if cfg.executor == "optimized":
        return _run_optimized(metas, cfg)

    raise ValueError(
        f"Unknown executor '{cfg.executor}'. Use 'ideal', 'noisy' or 'optimized'."
    )


def _run_batched(metas: list[_InstanceMeta], cfg: BatchConfig) -> BatchResult:
    """Execute every circuit in one job (ideal / noisy executors)."""
    backend = _resolve_backend(cfg)
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=cfg.optimization_level,
        seed_transpiler=cfg.seed_transpiler,
    )
    tqcs = [pm.run(m.circuit) for m in metas]

    started = time.time()
    with execution_mode(backend, mode=cfg.mode) as exec_mode:  # type: ignore[arg-type]
        counts_list = sample_counts(
            exec_mode,
            tqcs,
            shots=cfg.shots,
            seed_simulator=cfg.seed_simulator,
            meta={"stage": f"batch_{cfg.executor}", "n_instances": len(metas)},
        )
    elapsed = (time.time() - started) / max(1, len(metas))

    results: list[InstanceResult] = []
    for meta, tqc, counts in zip(metas, tqcs, counts_list, strict=True):
        try:
            metrics: BackendErrorMetrics | None = (
                None
                if cfg.executor == "ideal"
                else compute_accumulated_errors(backend, tqc)
            )
        except AttributeError:
            metrics = None

        results.append(
            InstanceResult(
                instance=meta.instance,
                n_qubits=meta.circuit.num_qubits,
                iterations=meta.iterations,
                solution_bitstrings=meta.solution_bitstrings,
                num_solutions=len(meta.solution_bitstrings),
                counts=counts,
                distribution=counts_to_probs(counts),
                solution_probability=solution_probability(
                    counts, meta.solution_bitstrings
                ),
                error_metrics=metrics,
                backend_name=backend_name(backend),
                depth=tqc.depth(),
                two_qubit_depth=_two_qubit_depth(tqc),
                elapsed_s=elapsed,
            )
        )

    return BatchResult(results=results, config=cfg)


def _run_optimized(metas: list[_InstanceMeta], cfg: BatchConfig) -> BatchResult:
    """Per-instance best backend, one layout search, unmitigated + ZNE."""
    zne_cfg = cfg.zne_config or ZNESamplingConfig(
        scales=[1, 3, 5],
        shots_per_scale=cfg.shots,
        method="linear",
        seed_simulator=cfg.seed_simulator,
    )

    if cfg.backend is not None:
        candidates: list[BackendLike] = [cfg.backend]
    else:
        if cfg.backend_config is None:
            raise ValueError(
                "BatchConfig.backend or BatchConfig.backend_config must be set "
                "for the 'optimized' executor."
            )
        bundle = build_all_backends(cfg.backend_config, service=cfg.service)
        candidates = list(bundle["real_backends"]) or list(bundle["fake_backends"])
        if not candidates:
            raise RuntimeError("No noisy backends available for the 'optimized' executor.")

    results: list[InstanceResult] = []
    for meta in metas:
        started = time.time()

        backend, _ = select_best_backend_by_error(meta.circuit, candidates)
        if backend is None:
            backend = candidates[0]

        tqc = transpile_once(meta.circuit, backend, zne_cfg)

        with execution_mode(backend, mode=cfg.mode) as exec_mode:  # type: ignore[arg-type]
            counts = sample_counts(
                exec_mode,
                tqc,
                shots=cfg.shots,
                seed_simulator=zne_cfg.seed_simulator,
                meta={"stage": "batch_optimized", "instance": meta.instance.name},
            )[0]

        mitigated = zne_mitigated_distribution(
            meta.circuit, backend, zne_cfg, transpiled=tqc
        )

        mitiq_prob: float | None = None
        if cfg.run_mitiq_baseline:
            mitiq_prob = mitiq_zne_solution_probability(
                tqc,
                backend,
                meta.solution_bitstrings,
                shots=zne_cfg.shots_per_scale,
                scale_factors=tuple(float(s) for s in zne_cfg.scales),
                seed_simulator=zne_cfg.seed_simulator,
            )

        try:
            metrics: BackendErrorMetrics | None = compute_accumulated_errors(
                backend, tqc
            )
        except AttributeError:
            metrics = None

        results.append(
            InstanceResult(
                instance=meta.instance,
                n_qubits=meta.circuit.num_qubits,
                iterations=meta.iterations,
                solution_bitstrings=meta.solution_bitstrings,
                num_solutions=len(meta.solution_bitstrings),
                counts=counts,
                distribution=counts_to_probs(counts),
                solution_probability=solution_probability(
                    counts, meta.solution_bitstrings
                ),
                mitigated_distribution=mitigated,
                mitigated_solution_probability=distribution_solution_probability(
                    mitigated, meta.solution_bitstrings
                ),
                mitiq_solution_probability=mitiq_prob,
                error_metrics=metrics,
                backend_name=backend_name(backend),
                depth=tqc.depth(),
                two_qubit_depth=_two_qubit_depth(tqc),
                elapsed_s=time.time() - started,
            )
        )

    return BatchResult(results=results, config=cfg)


def run_random_batch(
    n_instances: int,
    n_items: int,
    *,
    max_value: int = 20,
    seed: int = 0,
    config: BatchConfig | None = None,
) -> BatchResult:
    """
    Generate random feasible instances and run them as one batch.

    Parameters
    ----------
    n_instances:
        How many instances to generate.
    n_items:
        Items per instance (equals the index-register width).
    max_value:
        Maximum item value.
    seed:
        Base seed; instance ``k`` uses ``seed + k``, so batches are
        reproducible and non-overlapping.
    config:
        Batch settings; defaults to ``BatchConfig()``.

    Returns
    -------
    BatchResult
        As from :func:`run_batch`.
    """
    instances = [
        random_instance(
            n_items,
            max_value=max_value,
            seed=seed + k,
            name=f"rand_n{n_items}_{k}",
        )
        for k in range(n_instances)
    ]
    return run_batch(instances, config or BatchConfig())


__all__ = [
    "BatchConfig",
    "InstanceResult",
    "BatchResult",
    "prepare_instances",
    "run_batch",
    "run_random_batch",
]
