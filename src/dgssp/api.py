"""
api.py

The single-instance convenience layer.

:func:`run_dgssp` takes an instance and an executor name and does everything
in between: build the circuit, resolve a backend, execute, and (for the
``"optimized"`` path) mitigate.  It is the "just run it" entry point; reach for
:func:`dgssp.experiments.run_batch` instead as soon as you have more than one
instance, since that batches them into a single job.

Unlike the previous version, the configuration object passed in is **never
mutated**: defaults are filled into a local copy, so re-using one
:class:`DGRunConfig` across calls cannot silently change its meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .backends import BackendLike, BackendSelectionConfig, build_all_backends
from .execution import (
    ExecutionResult,
    execute_ideal,
    execute_noisy,
    execute_optimized_mitigated,
)
from .instance import SubsetSumInstance
from .mitigation import ZNESamplingConfig
from .solvers import DGConfig, DGSSPSolver, DPConfig, DPSSPSolver

ExecutorName = Literal["ideal", "noisy", "optimized"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DGRunConfig:
    """
    Settings for a single-instance run.

    Attributes
    ----------
    dg_config:
        Solver configuration.  ``None`` means ``DGConfig()``.
    backend_config:
        Backend filtering/seeding, used when no explicit backend is given.
    backend_kind:
        Which pool to draw from in ``"noisy"`` mode: ``"fake"`` or ``"real"``.
    backend:
        An explicit backend, which overrides ``backend_kind`` and
        ``backend_config``.
    service:
        An existing ``QiskitRuntimeService``, required to reach hardware.
    zne_config:
        ZNE settings for ``"optimized"`` mode; a sensible default is built if
        omitted.
    shots:
        Shots for the ``"ideal"`` and ``"noisy"`` modes.
    shots_unmitigated:
        Shots for the unmitigated run inside ``"optimized"`` mode.
    auto_num_solutions:
        When ``dg_config.iterations == "auto"``, run the classical DP solver
        first to obtain the true solution count ``M`` (exact but exponential in
        the worst case).  Set ``False`` to assume ``M = 1``.
    seed_simulator, seed_transpiler:
        Reproducibility seeds.
    """

    dg_config: DGConfig | None = None
    backend_config: BackendSelectionConfig | None = None
    backend_kind: Literal["real", "fake"] = "fake"
    backend: BackendLike | None = None
    service: Any | None = None
    zne_config: ZNESamplingConfig | None = None
    shots: int = 10_000
    shots_unmitigated: int = 10_000
    auto_num_solutions: bool = True
    seed_simulator: int | None = None
    seed_transpiler: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_dg_circuit(
    instance: SubsetSumInstance,
    dg_config: DGConfig | None = None,
    *,
    auto_num_solutions: bool = True,
):
    """
    Build the D-G circuit for an instance, resolving ``iterations="auto"``.

    Parameters
    ----------
    instance:
        The Subset Sum instance.
    dg_config:
        Solver configuration; ``None`` means ``DGConfig()``.
    auto_num_solutions:
        Whether to run the DP solver to count the true number of solutions
        when the iteration count is automatic.

    Returns
    -------
    tuple[QuantumCircuit, DGConfig]
        The circuit and the configuration it was built with.
    """
    cfg = dg_config or DGConfig()

    num_solutions: int | None = None
    if cfg.iterations == "auto" and auto_num_solutions:
        dp = DPSSPSolver(DPConfig(enumerate_all=True)).solve(instance)
        num_solutions = max(1, sum(1 for s in dp.all_solutions if s.is_exact))

    qc = DGSSPSolver(cfg).build_circuit(instance, num_solutions=num_solutions)
    return qc, cfg


def _resolve_noisy_backend(config: DGRunConfig) -> BackendLike:
    """Pick the backend for the ``"noisy"`` executor."""
    if config.backend is not None:
        return config.backend

    if config.backend_config is None:
        raise ValueError(
            "DGRunConfig.backend or DGRunConfig.backend_config must be set for "
            "the 'noisy' executor."
        )

    bundle = build_all_backends(config.backend_config, service=config.service)
    key = "fake_backends" if config.backend_kind == "fake" else "real_backends"
    pool: list[BackendLike] = bundle.get(key, [])
    if not pool:
        raise RuntimeError(
            f"No {config.backend_kind!r} backends available. Pass a "
            "QiskitRuntimeService via DGRunConfig.service, or an explicit backend."
        )
    return pool[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_dgssp(
    instance: SubsetSumInstance,
    *,
    executor: ExecutorName = "ideal",
    config: DGRunConfig | None = None,
) -> ExecutionResult | dict[str, ExecutionResult | None]:
    """
    Run the D-G algorithm on one instance.

    Parameters
    ----------
    instance:
        The Subset Sum instance.
    executor:
        ``"ideal"`` (noiseless simulator), ``"noisy"`` (one noisy backend) or
        ``"optimized"`` (best backend + best layout + ZNE).
    config:
        Run settings; defaults to ``DGRunConfig()``.  Never mutated.

    Returns
    -------
    ExecutionResult or dict[str, ExecutionResult | None]
        A single result for ``"ideal"`` and ``"noisy"``; for ``"optimized"``,
        ``{"best_fake": ..., "best_real": ...}``.

    Raises
    ------
    ValueError
        If the executor name is unknown or required configuration is missing.
    """
    cfg = config or DGRunConfig()
    qc, _ = build_dg_circuit(
        instance, cfg.dg_config, auto_num_solutions=cfg.auto_num_solutions
    )

    if executor == "ideal":
        return execute_ideal(
            qc,
            shots=cfg.shots,
            seed_simulator=cfg.seed_simulator,
            optimization_level=1,
        )

    if executor == "noisy":
        backend = _resolve_noisy_backend(cfg)
        seed_sim = cfg.seed_simulator
        if seed_sim is None and cfg.backend_config is not None:
            seed_sim = cfg.backend_config.seed_simulator
        return execute_noisy(
            qc,
            backend,
            shots=cfg.shots,
            seed_transpiler=cfg.seed_transpiler,
            seed_simulator=seed_sim,
            optimization_level=1,
        )

    if executor == "optimized":
        if cfg.backend_config is None and cfg.backend is None:
            raise ValueError(
                "DGRunConfig.backend_config or DGRunConfig.backend is required "
                "for the 'optimized' executor."
            )

        # Build a local copy rather than mutating the caller's config.
        zne_cfg = cfg.zne_config or ZNESamplingConfig(
            scales=[1, 3, 5],
            shots_per_scale=5_000,
            method="linear",
            seed_min=0,
            seed_max=64,
            optimization_level=3,
            layout_method="sabre",
            seed_simulator=(
                cfg.seed_simulator
                if cfg.seed_simulator is not None
                else getattr(cfg.backend_config, "seed_simulator", None)
            ),
        )

        if cfg.backend is not None:
            bundle: dict[str, Any] = {
                "service": cfg.service,
                "ideal": None,
                "real_backends": [],
                "fake_backends": [cfg.backend],
            }
        else:
            assert cfg.backend_config is not None
            bundle = build_all_backends(cfg.backend_config, service=cfg.service)

        return execute_optimized_mitigated(
            qc, bundle, zne_cfg, shots_unmitigated=cfg.shots_unmitigated
        )

    raise ValueError(
        f"Unknown executor '{executor}'. Use 'ideal', 'noisy' or 'optimized'."
    )


def run_dgssp_ideal(
    instance: SubsetSumInstance, *, config: DGRunConfig | None = None
) -> ExecutionResult:
    """
    Shortcut for ``run_dgssp(..., executor="ideal")``.

    Parameters
    ----------
    instance:
        The Subset Sum instance.
    config:
        Run settings.

    Returns
    -------
    ExecutionResult
        The noiseless run.
    """
    return run_dgssp(instance, executor="ideal", config=config)  # type: ignore[return-value]


def run_dgssp_noisy(
    instance: SubsetSumInstance, *, config: DGRunConfig | None = None
) -> ExecutionResult:
    """
    Shortcut for ``run_dgssp(..., executor="noisy")``.

    Parameters
    ----------
    instance:
        The Subset Sum instance.
    config:
        Run settings; must supply a backend or a backend config.

    Returns
    -------
    ExecutionResult
        The noisy run.
    """
    return run_dgssp(instance, executor="noisy", config=config)  # type: ignore[return-value]


def run_dgssp_optimized(
    instance: SubsetSumInstance, *, config: DGRunConfig | None = None
) -> dict[str, ExecutionResult | None]:
    """
    Shortcut for ``run_dgssp(..., executor="optimized")``.

    Parameters
    ----------
    instance:
        The Subset Sum instance.
    config:
        Run settings; must supply a backend or a backend config.

    Returns
    -------
    dict[str, ExecutionResult | None]
        ``{"best_fake": ..., "best_real": ...}``.
    """
    return run_dgssp(instance, executor="optimized", config=config)  # type: ignore[return-value]


__all__ = [
    "DGRunConfig",
    "build_dg_circuit",
    "run_dgssp",
    "run_dgssp_ideal",
    "run_dgssp_noisy",
    "run_dgssp_optimized",
]
