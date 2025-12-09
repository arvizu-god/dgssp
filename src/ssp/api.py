# src/ssp/api.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Union, List

from .instance import SubsetSumInstance
from .solvers import DGSSPSolver, DGConfig
from .backends import (
    BackendSelectionConfig,
    build_all_backends,
    BackendLike,
)
from .execution import (
    ExecutionResult,
    execute_ideal,
    execute_noisy,
    execute_optimized_mitigated,
)
from .mitigation import ZNESamplingConfig


# ---------------------------------------------------------------------------
# High-level configuration for DG-SSP API
# ---------------------------------------------------------------------------

@dataclass
class DGRunConfig:
    """
    Configuration for running the DG-SSP algorithm through the high-level API.

    Attributes
    ----------
    dg_config:
        Configuration for the Draper-Grover solver (iterations, assembly_type, ...).
        If None, the DGConfig default is used.
    backend_config:
        BackendSelectionConfig used when an IBM service / bundle of backends
        needs to be constructed (for 'noisy' and 'optimized' modes).
    backend_kind:
        For 'noisy' mode, which backend list to draw from when backend is not
        provided explicitly: 'fake' or 'real'. Defaults to 'fake'.
    backend:
        Optional explicit backend to use in 'noisy' mode. If provided, the
        API will use this backend directly and ignore backend_kind/backend_config.
    zne_config:
        ZNE configuration used in 'optimized' mode. If None, a reasonable
        default is constructed.
    shots:
        Number of shots used in 'ideal' and 'noisy' modes.
    shots_unmitigated:
        Number of shots for the unmitigated execution inside the
        'optimized' pipeline.
    """

    dg_config: Optional[DGConfig] = None

    backend_config: Optional[BackendSelectionConfig] = None
    backend_kind: Literal["real", "fake"] = "fake"
    backend: Optional[BackendLike] = None

    zne_config: Optional[ZNESamplingConfig] = None

    shots: int = 10_000
    shots_unmitigated: int = 10_000


# ---------------------------------------------------------------------------
# Internal helper: build DG-SSP circuit from instance + config
# ---------------------------------------------------------------------------

def _build_dg_circuit(
    instance: SubsetSumInstance,
    dg_config: Optional[DGConfig] = None,
):
    if dg_config is None:
        dg_config = DGConfig()  # assumes DGConfig has sensible defaults
    solver = DGSSPSolver(dg_config)
    qc = solver.build_circuit(instance)
    return qc, dg_config


# ---------------------------------------------------------------------------
# Core API: run_dgssp
# ---------------------------------------------------------------------------

def run_dgssp(
    instance: SubsetSumInstance,
    *,
    executor: Literal["ideal", "noisy", "optimized"] = "ideal",
    config: Optional[DGRunConfig] = None,
) -> Union[
    ExecutionResult,
    Dict[str, Optional[ExecutionResult]],
]:
    """
    High-level API entrypoint for running the DG-SSP algorithm on a given
    subset-sum instance.

    Parameters
    ----------
    instance:
        SubsetSumInstance describing the SSP problem.
    executor:
        Which execution pipeline to use:
          - 'ideal'     → run on ideal AerSimulator (no noise, no mitigation),
          - 'noisy'     → run on a fake or real backend with basic transpilation,
          - 'optimized' → pick best fake and real backends by error, optimize
                          layout with SABRE + best seed, and apply ZNE to obtain
                          a mitigated distribution.
    config:
        DGRunConfig with detailed settings (DGConfig, backend selection, ZNE, etc.).
        If None, a DGRunConfig with all defaults is used.

    Returns
    -------
    - For executor='ideal'  : ExecutionResult
    - For executor='noisy'  : ExecutionResult
    - For executor='optimized' : Dict[str, Optional[ExecutionResult]]
        {
          "best_fake": ExecutionResult or None,
          "best_real": ExecutionResult or None,
        }
    """
    if config is None:
        config = DGRunConfig()

    # 1) Build DG circuit
    qc, dg_config = _build_dg_circuit(instance, config.dg_config)

    # ----------------------------------------------------------------------
    # A) IDEAL EXECUTOR
    # ----------------------------------------------------------------------
    if executor == "ideal":
        result = execute_ideal(
            qc,
            shots=config.shots,
            seed_simulator=None,  # or expose via config if you prefer
            optimization_level=1,
        )
        return result

    # ----------------------------------------------------------------------
    # B) NOISY EXECUTOR
    # ----------------------------------------------------------------------
    if executor == "noisy":
        # If user provided a backend explicitly, use it
        if config.backend is not None:
            backend = config.backend
        else:
            if config.backend_config is None:
                raise ValueError(
                    "backend_config must be provided in DGRunConfig when "
                    "no explicit backend is passed for 'noisy' executor."
                )
            bundle = build_all_backends(config.backend_config)
            if config.backend_kind == "fake":
                backends_list: List[BackendLike] = bundle.get("fake_backends", [])
            else:
                backends_list = bundle.get("real_backends", [])

            if not backends_list:
                raise RuntimeError(
                    f"No {config.backend_kind!r} backends available "
                    "for the given BackendSelectionConfig."
                )

            backend = backends_list[0]

        result = execute_noisy(
            qc,
            backend,
            shots=config.shots,
            seed_transpiler=None,
            seed_simulator=getattr(config.backend_config, "seed_simulator", None)
            if config.backend_config is not None
            else None,
            optimization_level=1,
        )
        return result

    # ----------------------------------------------------------------------
    # C) OPTIMIZED / MITIGATED EXECUTOR
    # ----------------------------------------------------------------------
    if executor == "optimized":
        if config.backend_config is None:
            raise ValueError(
                "backend_config must be provided in DGRunConfig for the "
                "'optimized' executor."
            )

        # ZNE config: if not given, build a simple default
        if config.zne_config is None:
            config.zne_config = ZNESamplingConfig(
                scales=[1, 3, 5],
                shots_per_scale=5_000,
                method="linear",
                clip=True,
                renormalize=True,
                seed_min=0,
                seed_max=64,
                optimization_level=3,
                layout_method="sabre",
                seed_simulator=config.backend_config.seed_simulator
                if config.backend_config is not None
                else None,
            )

        # Build bundle of real/fake backends
        bundle = build_all_backends(config.backend_config)

        results = execute_optimized_mitigated(
            qc,
            bundle,
            config.zne_config,
            shots_unmitigated=config.shots_unmitigated,
        )
        return results

    # ----------------------------------------------------------------------
    # Unknown executor
    # ----------------------------------------------------------------------
    raise ValueError(
        f"Unknown executor '{executor}'. Use 'ideal', 'noisy', or 'optimized'."
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def run_dgssp_ideal(
    instance: SubsetSumInstance,
    *,
    config: Optional[DGRunConfig] = None,
) -> ExecutionResult:
    """Shortcut for run_dgssp(..., executor='ideal')."""
    return run_dgssp(instance, executor="ideal", config=config)  # type: ignore[return-value]


def run_dgssp_noisy(
    instance: SubsetSumInstance,
    *,
    config: Optional[DGRunConfig] = None,
) -> ExecutionResult:
    """Shortcut for run_dgssp(..., executor='noisy')."""
    return run_dgssp(instance, executor="noisy", config=config)  # type: ignore[return-value]


def run_dgssp_optimized(
    instance: SubsetSumInstance,
    *,
    config: Optional[DGRunConfig] = None,
) -> Dict[str, Optional[ExecutionResult]]:
    """Shortcut for run_dgssp(..., executor='optimized')."""
    return run_dgssp(instance, executor="optimized", config=config)  # type: ignore[return-value]


__all__ = [
    "DGRunConfig",
    "run_dgssp",
    "run_dgssp_ideal",
    "run_dgssp_noisy",
    "run_dgssp_optimized",
]
