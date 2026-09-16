"""
experiments.py

Multi-instance batch runner: the module you use to produce the paper's data.

Given a list of Subset Sum instances it will, in one call:

1. solve each instance classically (exhaustive DP) to obtain **ground truth**
   -- the set of correct bitstrings and their count ``M``;
2. use ``M`` to pick the optimal Grover iteration count per instance;
3. build and transpile one circuit per instance;
4. submit **all of them as a single job** via
   :func:`dgssp.runtime.sample_counts`, so a sweep costs one queue wait rather
   than one per instance;
5. decode each result against its ground truth and return structured,
   serialisable records, with bootstrap confidence intervals.

The ``"optimized"`` executor transpiles every instance once, folds all of them
at all their scales -- plus Mitiq's folds when the baseline is requested --
and submits everything through :func:`dgssp.mitigation.zne.run_zne_batch`: one
job per backend, so one job whenever a backend is given explicitly.

:func:`run_iteration_sweep` does the same for the Grover-iteration sweep: every
``r``-variant of one instance (and, for ZNE, every fold of every variant) goes
into one job.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
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
    calibration_timestamp,
    compute_accumulated_errors,
    median_gate_errors,
    physical_gate_counts,
)
from .decoding import exact_solution_bitstrings
from .encoding import build_encoding
from .execution import exact_index_distribution, select_best_backend_by_error
from .instance import SubsetSumInstance, random_instance
from .metrics import (
    bootstrap_metric,
    classical_fidelity,
    counts_to_probs,
    hellinger,
    solution_probability,
    survival_estimate,
    total_variation,
)
from .mitigation import (
    ZNEItem,
    ZNESamplingConfig,
    analyse_zne_item,
    run_zne_batch,
    transpile_once_seeded,
)
from .records import SCHEMA_VERSION, ResultRecord, json_safe, scale_key
from .runtime import backend_name, execution_mode, sample_counts
from .solvers import (
    DGConfig,
    DGSSPSolver,
    DPConfig,
    DPSSPSolver,
    grover_success_probability,
    optimal_iterations,
)
from .transpilation import transpiled_metrics

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
        ``"optimized"`` (transpile once + ZNE, one job).
    shots:
        Shots per instance.
    backend:
        An explicit backend for the ``"noisy"`` and ``"optimized"`` executors.
    backend_config:
        Used to build a backend bundle when none is given explicitly.
    service:
        An existing ``QiskitRuntimeService``, required to reach real hardware.
    zne_config:
        ZNE settings for the ``"optimized"`` executor.  Defaults are supplied
        if omitted.  Scale 1 is always measured (it is the unmitigated
        result) and is added when missing.
    run_mitiq_baseline:
        Also fold with Mitiq (inside the same job) and extrapolate offline
        with its factories.  Requires the ``dgssp[mitiq]`` extra.
    seed_simulator:
        Simulator seed for reproducibility.
    seed_transpiler:
        Transpiler seed for the ``"ideal"``/``"noisy"`` paths.
    optimization_level:
        Preset optimization level for those same paths.
    mode:
        Execution mode passed to :func:`dgssp.runtime.execution_mode`.
    bootstrap_samples, bootstrap_seed:
        Resamples and seed for the confidence intervals on unmitigated
        metrics (the ZNE intervals use ``zne_config.n_bootstrap``).
    compute_reference:
        Compute the exact noiseless distribution of every instance (a
        statevector simulation) so each record carries distances to it.
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
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0
    compute_reference: bool = True


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
        Raw measurement counts (the ``lambda = 1`` counts under ZNE).
    distribution:
        Normalised counts.
    solution_probability:
        Measured probability of a correct outcome.
    mitigated_distribution:
        Pointwise-ZNE distribution (``zne_config.renorm``), when mitigation ran.
    mitigated_solution_probability:
        The mitigated ``P(solution)`` of ``zne_config.mode``: the scalar
        extrapolation, or the pointwise distribution's solution mass.
    counts_per_scale:
        Raw counts keyed by **realised** ZNE noise scale.  Archived because
        they cannot be reconstructed from anything else, and resampling them is
        what the bootstrap intervals need.
    mitiq_solution_probability:
        ``P(solution)`` from Mitiq's extrapolation, when the baseline ran.
    error_metrics:
        Calibration error accumulation for the executed circuit.
    backend_name:
        Backend the circuit ran on.
    depth, two_qubit_depth:
        Depth of the executed circuit, overall and counting only two-qubit
        gates.
    two_qubit_count:
        Two-qubit gate applications in the executed circuit.
    physical_qubits:
        Physical qubits the executed circuit landed on, in virtual-qubit
        order, or ``None`` when the backend imposed no layout.
    elapsed_s:
        Wall-clock seconds spent on this instance's post-processing.
    best_seed:
        Transpiler seed of the executed circuit.
    one_q_physical, two_q_physical, virtual_gate_count:
        Gate counts from :func:`dgssp.backends.physical_gate_counts`.
    survival:
        :func:`dgssp.metrics.survival_estimate` from those counts.
    calibration_timestamp:
        Calibration snapshot of the backend, when it exposes one.
    job_id:
        Runtime job id of the job that produced the counts.
    p_uniform:
        ``M / 2**n``, the random-guessing baseline.
    reference_distribution:
        Exact noiseless distribution of the index register.
    metrics_ci:
        Bootstrap intervals and distances: ``solution_probability``
        ``[lo, hi]``, ``tv_to_reference`` ``{point, lo, hi}``,
        ``classical_fidelity``, ``hellinger``, ``mitigated_solution_probability``
        and ``mitiq_solution_probability`` ``[lo, hi]``.
    zne_fit, pointwise, pathology, mitiq:
        The blocks of :func:`dgssp.mitigation.zne.analyse_zne_item`.
    mitigation_mode:
        ``"none"``, ``"zne_scalar"`` or ``"zne_pointwise"``.
    renorm, fold_seed:
        The pointwise renormalisation and the folding seed used.
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
    counts_per_scale: dict[float, dict[str, int]] | None = None
    mitiq_solution_probability: float | None = None
    error_metrics: BackendErrorMetrics | None = None
    backend_name: str | None = None
    depth: int | None = None
    two_qubit_depth: int | None = None
    two_qubit_count: int | None = None
    physical_qubits: list[int] | None = None
    elapsed_s: float | None = None
    best_seed: int | None = None
    one_q_physical: int | None = None
    two_q_physical: int | None = None
    virtual_gate_count: int | None = None
    survival: float | None = None
    calibration_timestamp: str | None = None
    job_id: str | None = None
    p_uniform: float | None = None
    reference_distribution: dict[str, float] | None = None
    metrics_ci: dict[str, Any] = field(default_factory=dict)
    zne_fit: dict[str, Any] | None = None
    pointwise: dict[str, Any] | None = None
    pathology: dict[str, Any] | None = None
    mitiq: dict[str, Any] | None = None
    mitigation_mode: str = "none"
    renorm: str | None = None
    fold_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Flatten the record into a JSON-serialisable dictionary.

        Returns
        -------
        dict
            Nested dataclasses are expanded; the instance appears as its
            items, target and name; scale keys use :func:`dgssp.records.
            scale_key`; NaN / inf become ``None``.
        """
        return json_safe(
            {
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
                "counts_per_scale": (
                    {scale_key(s): dict(c) for s, c in self.counts_per_scale.items()}
                    if self.counts_per_scale is not None
                    else None
                ),
                "mitiq_solution_probability": self.mitiq_solution_probability,
                "error_metrics": (
                    self.error_metrics.to_dict() if self.error_metrics else None
                ),
                "backend_name": self.backend_name,
                "depth": self.depth,
                "two_qubit_depth": self.two_qubit_depth,
                "two_qubit_count": self.two_qubit_count,
                "physical_qubits": (
                    list(self.physical_qubits) if self.physical_qubits is not None else None
                ),
                "elapsed_s": self.elapsed_s,
                "best_seed": self.best_seed,
                "one_q_physical": self.one_q_physical,
                "two_q_physical": self.two_q_physical,
                "virtual_gate_count": self.virtual_gate_count,
                "survival": self.survival,
                "calibration_timestamp": self.calibration_timestamp,
                "job_id": self.job_id,
                "p_uniform": self.p_uniform,
                "metrics_ci": self.metrics_ci,
                "mitigation_mode": self.mitigation_mode,
            }
        )

    def transpiled_summary(self) -> dict[str, Any]:
        """
        The executed circuit's size and noise metrics, keyed as the paper records them.

        Returns
        -------
        dict
            ``{"two_q_count", "depth", "two_q_depth", "physical_qubits"}``
            (matching :func:`dgssp.transpilation.transpiled_metrics`) plus
            ``{"one_q_physical", "two_q_physical", "virtual_gate_count",
            "survival", "best_seed"}``.
        """
        return {
            "two_q_count": self.two_qubit_count,
            "depth": self.depth,
            "two_q_depth": self.two_qubit_depth,
            "physical_qubits": (
                list(self.physical_qubits) if self.physical_qubits is not None else None
            ),
            "one_q_physical": self.one_q_physical,
            "two_q_physical": self.two_q_physical,
            "virtual_gate_count": self.virtual_gate_count,
            "survival": self.survival,
            "best_seed": self.best_seed,
        }

    def to_record(
        self,
        *,
        executor: str,
        shots: int,
        mitigation_mode: str | None = None,
        job_id: str | None = None,
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> ResultRecord:
        """
        Convert this result into the frozen archive schema.

        Parameters
        ----------
        executor:
            Label for the execution path (``"ideal"``, ``"noisy"``,
            ``"optimized"``, ``"hardware"``).
        shots:
            Shots per circuit, per noise scale.
        mitigation_mode:
            Overrides :attr:`mitigation_mode`.
        job_id:
            Overrides :attr:`job_id`.
        extra_metrics:
            Additional entries merged into ``metrics``.

        Returns
        -------
        ResultRecord
            With ``schema_version = SCHEMA_VERSION`` and no provenance (the
            paper scripts stamp it).
        """
        instance = self.instance
        encoding = build_encoding(list(instance.items), instance.target)

        if self.counts_per_scale:
            counts = {scale_key(s): dict(c) for s, c in self.counts_per_scale.items()}
        else:
            counts = {"1": dict(self.counts)}

        metrics: dict[str, Any] = {
            "solution_bitstrings": list(self.solution_bitstrings),
            "num_solutions": self.num_solutions,
            "p_uniform": self.p_uniform,
            "solution_probability": self.solution_probability,
            "mitigated_solution_probability": self.mitigated_solution_probability,
            "mitiq_solution_probability": self.mitiq_solution_probability,
            "n_qubits": self.n_qubits,
            "elapsed_s": self.elapsed_s,
        }
        # Intervals ([lo, hi]) are stored under "<metric>_ci"; point values and
        # {point, lo, hi} blocks keep the metric's own name.
        for key, value in self.metrics_ci.items():
            metrics[f"{key}_ci" if isinstance(value, list) else key] = value
        if self.reference_distribution is not None:
            metrics["reference_distribution"] = dict(self.reference_distribution)
        if self.error_metrics is not None:
            metrics["error_metrics"] = self.error_metrics.to_dict()
        if self.mitigated_distribution is not None:
            metrics["mitigated_distribution"] = dict(self.mitigated_distribution)
        if extra_metrics:
            metrics.update(extra_metrics)

        return ResultRecord(
            instance={
                "items": list(instance.items),
                "target": instance.target,
                "name": instance.name,
            },
            n_items=instance.n_items,
            n_sum=encoding.n_sum,
            iterations=self.iterations,
            executor=executor,
            mitigation_mode=mitigation_mode or self.mitigation_mode,
            backend=self.backend_name,
            shots=shots,
            counts=counts,
            metrics=metrics,
            transpiled=self.transpiled_summary(),
            job_id=job_id if job_id is not None else self.job_id,
            schema_version=SCHEMA_VERSION,
            calibration_timestamp=self.calibration_timestamp,
            zne_fit=self.zne_fit,
            pathology=self.pathology,
            mitiq=self.mitiq,
            pointwise=self.pointwise,
            renorm=self.renorm,
            fold_seed=self.fold_seed,
        )


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
            for key in ("counts", "distribution", "mitigated_distribution", "counts_per_scale"):
                row.pop(key, None)
            metrics = row.pop("error_metrics", None) or {}
            row.update({f"err_{k}": v for k, v in metrics.items()})
            ci = row.pop("metrics_ci", None) or {}
            for key, value in ci.items():
                if isinstance(value, list) and len(value) == 2:
                    row[f"{key}_lo"], row[f"{key}_hi"] = value
            row["solution_bitstrings"] = ",".join(row["solution_bitstrings"])
            qubits = row.get("physical_qubits")
            row["physical_qubits"] = ",".join(str(q) for q in qubits) if qubits else None
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
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
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


# ---------------------------------------------------------------------------
# Shared measurement helpers
# ---------------------------------------------------------------------------


def _p_uniform(instance: SubsetSumInstance, solutions: Sequence[str]) -> float:
    """``M / 2**n``: the probability a uniform guess is correct."""
    return len(solutions) / 2**instance.n_items


def _noise_counts(backend: BackendLike, tqc: QuantumCircuit) -> dict[str, Any]:
    """Physical gate counts and survival, or ``None`` entries when uncalibrated."""
    out: dict[str, Any] = {
        "one_q_physical": None,
        "two_q_physical": None,
        "virtual_gate_count": None,
        "survival": None,
    }
    try:
        errors = median_gate_errors(backend)
        if errors.get("two_qubit") is None:
            return out
        physical = physical_gate_counts(tqc, errors)
    except (AttributeError, ValueError):
        return out
    out.update(
        {
            "one_q_physical": physical["one_q_physical"],
            "two_q_physical": physical["two_q_physical"],
            "virtual_gate_count": physical["virtual"],
            "survival": survival_estimate(
                physical["two_q_physical"], physical["one_q_physical"], errors
            )["survival"],
        }
    )
    return out


def _distribution_metrics(
    counts: Mapping[str, int],
    solutions: Sequence[str],
    reference: Mapping[str, float] | None,
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap interval on P(solution), and distances to the reference."""
    sols = list(solutions)
    _, lo, hi = bootstrap_metric(counts, lambda c: solution_probability(c, sols), B, seed)
    out: dict[str, Any] = {"solution_probability": [lo, hi]}
    if reference is not None:
        tv, tv_lo, tv_hi = bootstrap_metric(
            counts, lambda c: total_variation(c, reference), B, seed
        )
        out["tv_to_reference"] = {"point": tv, "lo": tv_lo, "hi": tv_hi}
        out["classical_fidelity"] = classical_fidelity(counts, reference)
        out["hellinger"] = hellinger(counts, reference)
    return out


def _error_metrics(backend: BackendLike, tqc: QuantumCircuit) -> BackendErrorMetrics | None:
    """Calibration error accumulation, or ``None`` for an uncalibrated backend."""
    try:
        return compute_accumulated_errors(backend, tqc)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_batch(
    instances: Sequence[SubsetSumInstance], config: BatchConfig | None = None
) -> BatchResult:
    """
    Build, execute and decode a whole set of instances.

    For the ``"ideal"`` and ``"noisy"`` executors every circuit is submitted in
    a **single** job.  For ``"optimized"`` every fold of every instance is
    submitted in a single job per backend (see :func:`_run_optimized`).

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
    job_id = getattr(counts_list, "job_id", None)
    cal = calibration_timestamp(backend)

    results: list[InstanceResult] = []
    for meta, tqc, counts in zip(metas, tqcs, counts_list, strict=True):
        noisy = cfg.executor != "ideal"
        reference = (
            exact_index_distribution(meta.instance, meta.iterations)
            if cfg.compute_reference
            else None
        )
        sizes = transpiled_metrics(tqc)
        results.append(
            InstanceResult(
                instance=meta.instance,
                n_qubits=meta.circuit.num_qubits,
                iterations=meta.iterations,
                solution_bitstrings=meta.solution_bitstrings,
                num_solutions=len(meta.solution_bitstrings),
                counts=counts,
                distribution=counts_to_probs(counts),
                solution_probability=solution_probability(counts, meta.solution_bitstrings),
                error_metrics=_error_metrics(backend, tqc) if noisy else None,
                backend_name=backend_name(backend),
                depth=sizes["depth"],  # type: ignore[arg-type]
                two_qubit_depth=sizes["two_q_depth"],  # type: ignore[arg-type]
                two_qubit_count=sizes["two_q_count"],  # type: ignore[arg-type]
                physical_qubits=sizes["physical_qubits"],  # type: ignore[arg-type]
                elapsed_s=elapsed,
                best_seed=cfg.seed_transpiler,
                calibration_timestamp=cal,
                job_id=job_id,
                p_uniform=_p_uniform(meta.instance, meta.solution_bitstrings),
                reference_distribution=reference,
                metrics_ci=_distribution_metrics(
                    counts,
                    meta.solution_bitstrings,
                    reference,
                    B=cfg.bootstrap_samples,
                    seed=cfg.bootstrap_seed,
                ),
                **(_noise_counts(backend, tqc) if noisy else {}),
            )
        )

    return BatchResult(results=results, config=cfg, job_ids=[job_id] if job_id else [])


def _with_unit_scale(scales: Sequence[float]) -> list[float]:
    """The scales, with 1 prepended when absent (it is the unmitigated run)."""
    out = [float(s) for s in scales]
    return out if any(s == 1.0 for s in out) else [1.0, *out]


def _run_optimized(metas: list[_InstanceMeta], cfg: BatchConfig) -> BatchResult:
    """
    Transpile each instance once, then fold and sample everything in one job.

    With an explicit ``cfg.backend`` (the paper's path) all instances share it
    and the whole batch -- every scale of every instance, plus Mitiq's folds
    -- is one :func:`dgssp.runtime.sample_counts` call.  With a candidate pool
    each instance picks its lowest-error backend and there is one job per
    distinct backend chosen.
    """
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

    # Transpile once per instance, grouped by the backend each will run on.
    groups: dict[int, tuple[BackendLike, list[int]]] = {}
    prepared: list[tuple[QuantumCircuit, int]] = []
    for index, meta in enumerate(metas):
        backend = candidates[0]
        if len(candidates) > 1:
            chosen, _ = select_best_backend_by_error(meta.circuit, candidates)
            backend = chosen if chosen is not None else candidates[0]
        prepared.append(transpile_once_seeded(meta.circuit, backend, zne_cfg))
        groups.setdefault(id(backend), (backend, []))[1].append(index)

    results: list[InstanceResult | None] = [None] * len(metas)
    job_ids: list[str] = []
    scales = _with_unit_scale(zne_cfg.scales)
    mode_label = f"zne_{zne_cfg.mode}"

    for backend, indices in groups.values():
        items = [
            ZNEItem(
                name=metas[i].instance.name or f"instance_{i}",
                circuit=prepared[i][0],
                scales=scales,
                solution_bitstrings=metas[i].solution_bitstrings,
            )
            for i in indices
        ]
        batch = run_zne_batch(
            items, backend, zne_cfg, include_mitiq=cfg.run_mitiq_baseline, mode=cfg.mode
        )
        if batch.job_id:
            job_ids.append(batch.job_id)
        cal = calibration_timestamp(backend)

        for i, run in zip(indices, batch.items, strict=True):
            started = time.time()
            meta = metas[i]
            tqc, seed = prepared[i]
            sols = meta.solution_bitstrings
            p_u = _p_uniform(meta.instance, sols)
            reference = exact_index_distribution(meta.instance, meta.iterations)
            analysis = analyse_zne_item(
                run, zne_cfg, solution_bitstrings=sols, p_uniform=p_u, reference=reference
            )
            counts = run.counts_per_scale[1.0]
            scalar = analysis["scalar"]
            pointwise_block = analysis["pointwise"]
            mitiq_block = analysis["mitiq"]

            if zne_cfg.mode == "scalar":
                mitigated_p = scalar["value"]
                mitigated_ci = scalar["ci"]
            else:
                mitigated_p = pointwise_block[zne_cfg.renorm]["solution_probability"]
                boot = pointwise_block["bootstrap"]["solution_probability"]
                mitigated_ci = [boot["lo"], boot["hi"]]

            metrics_ci = _distribution_metrics(
                counts, sols, reference, B=cfg.bootstrap_samples, seed=cfg.bootstrap_seed
            )
            metrics_ci["mitigated_solution_probability"] = mitigated_ci
            if mitiq_block is not None:
                metrics_ci["mitiq_solution_probability"] = [mitiq_block["lo"], mitiq_block["hi"]]

            sizes = transpiled_metrics(tqc)
            results[i] = InstanceResult(
                instance=meta.instance,
                n_qubits=meta.circuit.num_qubits,
                iterations=meta.iterations,
                solution_bitstrings=sols,
                num_solutions=len(sols),
                counts=counts,
                distribution=counts_to_probs(counts),
                solution_probability=solution_probability(counts, sols),
                mitigated_distribution=pointwise_block[zne_cfg.renorm]["distribution"],
                mitigated_solution_probability=mitigated_p,
                counts_per_scale=run.counts_per_scale,
                mitiq_solution_probability=(
                    mitiq_block["value"] if mitiq_block is not None else None
                ),
                error_metrics=_error_metrics(backend, tqc),
                backend_name=backend_name(backend),
                depth=sizes["depth"],  # type: ignore[arg-type]
                two_qubit_depth=sizes["two_q_depth"],  # type: ignore[arg-type]
                two_qubit_count=sizes["two_q_count"],  # type: ignore[arg-type]
                physical_qubits=sizes["physical_qubits"],  # type: ignore[arg-type]
                elapsed_s=time.time() - started,
                best_seed=seed,
                calibration_timestamp=cal,
                job_id=batch.job_id,
                p_uniform=p_u,
                reference_distribution=reference,
                metrics_ci=metrics_ci,
                zne_fit=scalar,
                pointwise=pointwise_block,
                pathology=analysis["pathology"],
                mitiq=mitiq_block,
                mitigation_mode=mode_label,
                renorm=zne_cfg.renorm,
                fold_seed=zne_cfg.fold_seed,
                **_noise_counts(backend, tqc),
            )

    return BatchResult(
        results=[r for r in results if r is not None], config=cfg, job_ids=job_ids
    )


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


# ---------------------------------------------------------------------------
# Grover-iteration sweep
# ---------------------------------------------------------------------------

SweepExecutor = Literal["ideal", "noisy", "zne"]


def run_iteration_sweep(
    instance: SubsetSumInstance,
    r_values: Sequence[int],
    backend: BackendLike | None = None,
    *,
    executor: SweepExecutor = "ideal",
    zne_config: ZNESamplingConfig | None = None,
    shots: int = 10_000,
    best_seeds: Mapping[int, int] | None = None,
    seed_min: int = 0,
    seed_max: int = 32,
    optimization_level: int = 3,
    scales_per_r: Mapping[int, Sequence[float]] | None = None,
    include_mitiq: bool = False,
    seed_simulator: int | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    mode: str = "auto",
) -> dict[str, Any]:
    """
    ``P(solution)`` against the number of Grover iterations, in ONE job.

    Every ``r``-variant of the instance is built and transpiled once at
    ``optimization_level`` -- with its recorded seed when ``best_seeds`` has
    one (the WP3 ``best_seed`` of the ``r_opt`` variant), otherwise by the
    ``[seed_min, seed_max)`` SABRE sweep.  For ``executor="zne"`` each variant
    is folded at its scales.  The whole set is a single
    :func:`dgssp.runtime.sample_counts` call.

    Parameters
    ----------
    instance:
        The instance (from ``paper/instances/sweep.json``).
    r_values:
        Iteration counts to run.
    backend:
        The backend; ``None`` with ``executor="ideal"`` builds a noiseless Aer
        simulator.  Required for ``"noisy"`` and ``"zne"``.
    executor:
        ``"ideal"``, ``"noisy"`` or ``"zne"``.
    zne_config:
        ZNE settings (``"zne"`` only); its ``shots_per_scale`` is the shot
        count per circuit.
    shots:
        Shots per circuit for ``"ideal"`` / ``"noisy"``.
    best_seeds:
        ``{r: seed}`` for variants whose transpilation must be reproduced.
    seed_min, seed_max:
        SABRE sweep range for the other variants.  On a backend without a
        coupling map (the ideal simulator) only ``seed_min`` is used: there is
        no layout to search.
    optimization_level:
        Preset optimization level of the one transpilation.
    scales_per_r:
        ``{r: scales}`` overriding ``zne_config.scales`` per variant (e.g.
        from :func:`dgssp.mitigation.zne.plan_scales`).  Scale 1 is always
        included.
    include_mitiq:
        Also run Mitiq's folds (``"zne"`` only).
    seed_simulator:
        Simulator seed.
    bootstrap_samples, bootstrap_seed:
        Bootstrap settings for the unmitigated ``P(solution)`` intervals.
    mode:
        Execution mode.

    Returns
    -------
    dict
        ``{"instance", "M", "p_uniform", "solution_bitstrings", "executor",
        "backend", "job_id", "rows"}``.  Each row holds ``iterations``,
        ``best_seed``, the transpiled sizes, ``survival`` (calibrated
        backends), ``p_sol`` with ``p_sol_ci``, the analytic
        ``sin^2((2r+1) theta)``, the raw counts, and for ZNE the scalar fit
        (``zne``: value, CI, resolved, reported value) and the Mitiq block.
    """
    if not r_values:
        raise ValueError("run_iteration_sweep needs at least one iteration count.")
    if executor not in ("ideal", "noisy", "zne"):
        raise ValueError(f"Unknown executor '{executor}'. Use 'ideal', 'noisy' or 'zne'.")
    if backend is None:
        if executor != "ideal":
            raise ValueError(f"executor='{executor}' needs an explicit backend.")
        backend = build_ideal_aer_backend(seed_simulator=seed_simulator)

    sols = exact_solution_bitstrings(instance, DPSSPSolver(DPConfig(enumerate_all=True)).solve(instance))
    n = instance.n_items
    p_u = len(sols) / 2**n
    seeds = dict(best_seeds or {})
    has_layout = getattr(backend, "coupling_map", None) is not None

    tqcs: list[QuantumCircuit] = []
    seeds_used: list[int] = []
    for r in r_values:
        qc = DGSSPSolver(DGConfig(iterations=int(r))).build_circuit(instance)
        if r in seeds:
            seed_lo, seed_hi = seeds[r], seeds[r] + 1
        elif has_layout:
            seed_lo, seed_hi = seed_min, seed_max
        else:
            seed_lo, seed_hi = seed_min, seed_min + 1
        tqc, seed = transpile_once_seeded(
            qc,
            backend,
            ZNESamplingConfig(
                seed_min=seed_lo, seed_max=seed_hi, optimization_level=optimization_level
            ),
        )
        tqcs.append(tqc)
        seeds_used.append(seed)

    rows: list[dict[str, Any]] = []
    job_id: str | None

    if executor == "zne":
        cfg = zne_config or ZNESamplingConfig(seed_simulator=seed_simulator)
        overrides = dict(scales_per_r or {})
        items = [
            ZNEItem(
                name=f"{instance.name}_r{r}",
                circuit=tqc,
                scales=_with_unit_scale(overrides.get(r, cfg.scales)),
                solution_bitstrings=sols,
            )
            for r, tqc in zip(r_values, tqcs, strict=True)
        ]
        batch = run_zne_batch(items, backend, cfg, include_mitiq=include_mitiq, mode=mode)
        job_id = batch.job_id
        per_r_counts = [run.counts_per_scale[1.0] for run in batch.items]
        runs: list[Any] = list(batch.items)
        circuit_shots = cfg.shots_per_scale
    else:
        with execution_mode(backend, mode=mode) as exec_mode:  # type: ignore[arg-type]
            counts_list = sample_counts(
                exec_mode,
                tqcs,
                shots=shots,
                seed_simulator=seed_simulator,
                meta={"stage": f"sweep_{executor}", "instance": instance.name},
            )
        job_id = getattr(counts_list, "job_id", None)
        per_r_counts = list(counts_list)
        runs = [None] * len(tqcs)
        circuit_shots = shots

    for r, tqc, seed, counts, run in zip(r_values, tqcs, seeds_used, per_r_counts, runs, strict=True):
        p_sol, lo, hi = bootstrap_metric(
            counts, lambda c: solution_probability(c, sols), bootstrap_samples, bootstrap_seed
        )
        row: dict[str, Any] = {
            "iterations": int(r),
            "best_seed": seed,
            **transpiled_metrics(tqc),
            **(_noise_counts(backend, tqc) if executor != "ideal" else {}),
            "shots": circuit_shots,
            "p_sol": p_sol,
            "p_sol_ci": [lo, hi],
            "analytic": grover_success_probability(n, len(sols), int(r)),
            "counts": {"1": dict(counts)},
            "zne": None,
            "mitiq": None,
        }
        if run is not None:
            analysis = analyse_zne_item(
                run,
                cfg,
                solution_bitstrings=sols,
                p_uniform=p_u,
                reference=exact_index_distribution(instance, int(r)),
            )
            row["counts"] = {scale_key(s): dict(c) for s, c in run.counts_per_scale.items()}
            row["zne"] = analysis["scalar"]
            row["pathology"] = analysis["pathology"]
            row["mitiq"] = analysis["mitiq"]
        rows.append(row)

    return json_safe(
        {
            "instance": {"items": list(instance.items), "target": instance.target, "name": instance.name},
            "M": len(sols),
            "p_uniform": p_u,
            "solution_bitstrings": list(sols),
            "executor": executor,
            "backend": backend_name(backend),
            "job_id": job_id,
            "rows": rows,
        }
    )


__all__ = [
    "BatchConfig",
    "InstanceResult",
    "BatchResult",
    "prepare_instances",
    "run_batch",
    "run_random_batch",
    "run_iteration_sweep",
]
