"""
dgssp.mitigation

Quantum error mitigation for the D-G sampling algorithm.

Two complementary approaches live here:

* :mod:`~dgssp.mitigation.zne` -- the library's own per-bitstring ZNE, which
  returns a full mitigated distribution.
* :mod:`~dgssp.mitigation.mitiq_baseline` -- Mitiq's ZNE applied to the scalar
  ``P(solution)``, as an independent comparison baseline.

The Mitiq symbols are re-exported here but import Mitiq lazily, so this package
is importable without the optional dependency installed.
"""

from __future__ import annotations

from .mitiq_baseline import (
    make_sampling_executor,
    mitiq_is_available,
    mitiq_zne_solution_probability,
)
from .zne import (
    ZNESamplingConfig,
    dgssp_zne_mitigated_distribution,
    extrapolate_distribution,
    fold_transpiled,
    rebase_to_backend,
    run_zne_scales,
    transpile_once,
    zne_fit_single_value,
    zne_mitigated_distribution,
)

__all__ = [
    "ZNESamplingConfig",
    "fold_transpiled",
    "rebase_to_backend",
    "zne_fit_single_value",
    "transpile_once",
    "run_zne_scales",
    "extrapolate_distribution",
    "zne_mitigated_distribution",
    "dgssp_zne_mitigated_distribution",
    "make_sampling_executor",
    "mitiq_zne_solution_probability",
    "mitiq_is_available",
]
