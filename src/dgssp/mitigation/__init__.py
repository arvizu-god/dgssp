"""
dgssp.mitigation

Quantum error mitigation for the D-G sampling algorithm.

Two complementary approaches live here:

* :mod:`~dgssp.mitigation.zne` -- the library's own ZNE: a scalar estimator of
  ``P(solution)`` (weighted fits, bootstrap intervals, resolvability flag), a
  pointwise estimator of the whole distribution (clip or simplex
  renormalisation), partial ISA folding, and the single-job batch path.
* :mod:`~dgssp.mitigation.mitiq_baseline` -- Mitiq's folding and
  extrapolation applied to ``P(solution)``, as an independent baseline.

The Mitiq symbols are re-exported here but import Mitiq lazily, so this package
is importable without the optional dependency installed.
"""

from __future__ import annotations

from .mitiq_baseline import (
    make_sampling_executor,
    mitiq_bootstrap,
    mitiq_extrapolate,
    mitiq_fold_isa,
    mitiq_is_available,
    mitiq_zne_solution_probability,
)
from .zne import (
    ZNE_METHODS,
    ZNE_RENORMS,
    FoldResult,
    ZNEBatchRun,
    ZNEItem,
    ZNEItemRun,
    ZNESamplingConfig,
    analyse_zne_item,
    clip_renormalise,
    dgssp_zne_mitigated_distribution,
    extrapolate_distribution,
    extrapolate_scalar,
    fit_scalar_series,
    fold_isa,
    fold_transpiled,
    is_resolved,
    pathology_diagnostic,
    plan_scales,
    project_to_simplex,
    realised_scale,
    rebase_to_backend,
    run_zne_batch,
    run_zne_scales,
    simplex_renormalise,
    transpile_once,
    transpile_once_seeded,
    zne_bootstrap,
    zne_fit_single_value,
    zne_mitigated_distribution,
)

__all__ = [
    "ZNESamplingConfig",
    "ZNE_METHODS",
    "ZNE_RENORMS",
    "FoldResult",
    "fold_isa",
    "fold_transpiled",
    "realised_scale",
    "rebase_to_backend",
    "zne_fit_single_value",
    "fit_scalar_series",
    "extrapolate_scalar",
    "is_resolved",
    "project_to_simplex",
    "clip_renormalise",
    "simplex_renormalise",
    "extrapolate_distribution",
    "zne_bootstrap",
    "plan_scales",
    "pathology_diagnostic",
    "transpile_once",
    "transpile_once_seeded",
    "run_zne_scales",
    "ZNEItem",
    "ZNEItemRun",
    "ZNEBatchRun",
    "run_zne_batch",
    "analyse_zne_item",
    "zne_mitigated_distribution",
    "dgssp_zne_mitigated_distribution",
    "make_sampling_executor",
    "mitiq_fold_isa",
    "mitiq_extrapolate",
    "mitiq_bootstrap",
    "mitiq_zne_solution_probability",
    "mitiq_is_available",
]
