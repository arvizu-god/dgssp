"""
dgssp

A Draper-adder + Grover implementation of the Subset Sum Problem, with a
classical dynamic-programming baseline, IBM Quantum execution helpers and
zero-noise-extrapolation error mitigation.

Layout
------
``instance``      Problem data structures and random-instance generation.
``encoding``      Offset + guard-bit arithmetic for the sum register.
``decoding``      Bitstring <-> subset translation.
``metrics``       P(solution), distances, survival, bootstrap intervals.
``records``       The frozen results schema.
``solvers``       The D-G circuit builder and the classical DP baseline.
``runtime``       IBM service, execution modes, batched sampling, job logging.
``backends``      Backend discovery, noise models, calibration-based scoring.
``transpilation`` SABRE layout search minimising two-qubit calibration error.
``mitigation``    Scalar and pointwise ZNE, one-job batches, Mitiq baseline.
``execution``     Single-circuit executors: ideal, noisy, optimized+mitigated.
``experiments``   Multi-instance batch runner with built-in ground truth.
``api``           Single-instance convenience entry point.
``cli``           ``dgssp-run`` command-line front end.

Quickstart
----------
>>> from dgssp import SubsetSumInstance, BatchConfig, run_batch
>>> instance = SubsetSumInstance(items=[3, 5, -2, 7], target=8)
>>> result = run_batch([instance], BatchConfig(shots=4096, seed_simulator=1234))
>>> result.results[0].solution_probability > 0.5
True

Importing this package has no side effects: no network calls, no credential
writes.  See :func:`dgssp.save_account` for opt-in credential storage.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import (
    DGRunConfig,
    build_dg_circuit,
    run_dgssp,
    run_dgssp_ideal,
    run_dgssp_noisy,
    run_dgssp_optimized,
)
from .backends import (
    BackendErrorMetrics,
    BackendPerformance,
    BackendSelectionConfig,
    build_all_backends,
    build_ideal_aer_backend,
    calibration_timestamp,
    compute_accumulated_errors,
    evaluate_backends_for_circuit,
    is_heavy_hex,
    list_fake_backends,
    median_gate_errors,
    physical_gate_counts,
    processor_family,
    select_best_backends,
    smallest_heavy_hex_fake_backend,
)
from .decoding import (
    bit_is_solution,
    bitstring_to_indices,
    bitstring_to_solution,
    bitstring_to_subset,
    exact_solution_bitstrings,
    indices_to_bitstring,
)
from .encoding import SumRegisterEncoding, build_encoding
from .execution import (
    ExecutionResult,
    exact_index_distribution,
    execute_ideal,
    execute_noisy,
    execute_optimized_mitigated,
)
from .experiments import (
    BatchConfig,
    BatchResult,
    InstanceResult,
    run_batch,
    run_iteration_sweep,
    run_random_batch,
)
from .instance import DPResult, SSPSolution, SubsetSumInstance, random_instance
from .metrics import (
    bootstrap_metric,
    classical_fidelity,
    counts_to_probs,
    distribution_solution_probability,
    hellinger,
    solution_probability,
    survival_estimate,
    total_variation,
    uniform_distribution,
)
from .mitigation import (
    ZNEItem,
    ZNESamplingConfig,
    analyse_zne_item,
    dgssp_zne_mitigated_distribution,
    extrapolate_scalar,
    fold_isa,
    fold_transpiled,
    mitiq_is_available,
    mitiq_zne_solution_probability,
    pathology_diagnostic,
    plan_scales,
    project_to_simplex,
    rebase_to_backend,
    run_zne_batch,
    zne_bootstrap,
    zne_mitigated_distribution,
)
from .records import SCHEMA_VERSION, ResultRecord, scale_key
from .runtime import (
    describe_account,
    execution_mode,
    fetch_result,
    get_service,
    print_account_summary,
    read_job_log,
    sample_counts,
    save_account,
)
from .solvers import (
    BaseClassicalSSPSolver,
    BaseQuantumSSPSolver,
    BaseSSPSolver,
    DGConfig,
    DGResourceModel,
    DGSSPSolver,
    DPConfig,
    DPSSPSolver,
    encoding_without_guard_bit,
    flatten_containers,
    optimal_iterations,
    resource_model,
)
from .transpilation import (
    BestSeedResult,
    find_best_seed,
    logical_counts,
    mcz_noaux_1q_cost,
    mcz_noaux_cx_cost,
    to_logical_basis,
    transpiled_metrics,
)

__all__ = [
    "__version__",
    # instance
    "SubsetSumInstance",
    "SSPSolution",
    "DPResult",
    "random_instance",
    # encoding / decoding
    "SumRegisterEncoding",
    "build_encoding",
    "indices_to_bitstring",
    "bitstring_to_indices",
    "bitstring_to_subset",
    "bitstring_to_solution",
    "bit_is_solution",
    "exact_solution_bitstrings",
    # metrics
    "counts_to_probs",
    "solution_probability",
    "distribution_solution_probability",
    "total_variation",
    "classical_fidelity",
    "hellinger",
    "uniform_distribution",
    "survival_estimate",
    "bootstrap_metric",
    # records
    "ResultRecord",
    "SCHEMA_VERSION",
    "scale_key",
    # solvers
    "BaseSSPSolver",
    "BaseQuantumSSPSolver",
    "BaseClassicalSSPSolver",
    "DGSSPSolver",
    "DGConfig",
    "DGResourceModel",
    "resource_model",
    "flatten_containers",
    "encoding_without_guard_bit",
    "optimal_iterations",
    "DPSSPSolver",
    "DPConfig",
    # runtime
    "get_service",
    "save_account",
    "describe_account",
    "print_account_summary",
    "execution_mode",
    "sample_counts",
    "fetch_result",
    "read_job_log",
    # backends / transpilation
    "BackendSelectionConfig",
    "BackendErrorMetrics",
    "BackendPerformance",
    "build_all_backends",
    "build_ideal_aer_backend",
    "compute_accumulated_errors",
    "median_gate_errors",
    "physical_gate_counts",
    "evaluate_backends_for_circuit",
    "select_best_backends",
    "processor_family",
    "is_heavy_hex",
    "calibration_timestamp",
    "list_fake_backends",
    "smallest_heavy_hex_fake_backend",
    "find_best_seed",
    "BestSeedResult",
    "transpiled_metrics",
    "to_logical_basis",
    "logical_counts",
    "mcz_noaux_cx_cost",
    "mcz_noaux_1q_cost",
    # mitigation
    "ZNESamplingConfig",
    "fold_isa",
    "fold_transpiled",
    "rebase_to_backend",
    "extrapolate_scalar",
    "zne_bootstrap",
    "plan_scales",
    "pathology_diagnostic",
    "project_to_simplex",
    "ZNEItem",
    "run_zne_batch",
    "analyse_zne_item",
    "zne_mitigated_distribution",
    "dgssp_zne_mitigated_distribution",
    "mitiq_zne_solution_probability",
    "mitiq_is_available",
    # execution
    "ExecutionResult",
    "execute_ideal",
    "exact_index_distribution",
    "execute_noisy",
    "execute_optimized_mitigated",
    # experiments
    "BatchConfig",
    "BatchResult",
    "InstanceResult",
    "run_batch",
    "run_random_batch",
    "run_iteration_sweep",
    # api
    "DGRunConfig",
    "build_dg_circuit",
    "run_dgssp",
    "run_dgssp_ideal",
    "run_dgssp_noisy",
    "run_dgssp_optimized",
]
