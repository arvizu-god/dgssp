"""
dgssp.solvers

Solver package: the quantum Draper--Grover solver and the classical
dynamic-programming baseline, plus the abstract bases they share.

The QPE-based variant that used to live here has been removed.
"""

from __future__ import annotations

from .base import BaseClassicalSSPSolver, BaseQuantumSSPSolver, BaseSSPSolver
from .classical import DPConfig, DPSSPSolver
from .draper_grover import (
    CONTAINER_GATE_NAMES,
    DGConfig,
    DGResourceModel,
    DGSSPSolver,
    encoding_without_guard_bit,
    flatten_containers,
    optimal_iterations,
    resource_model,
    two_adic_valuation,
)

__all__ = [
    "BaseSSPSolver",
    "BaseQuantumSSPSolver",
    "BaseClassicalSSPSolver",
    "DGSSPSolver",
    "DGConfig",
    "DGResourceModel",
    "CONTAINER_GATE_NAMES",
    "flatten_containers",
    "resource_model",
    "two_adic_valuation",
    "encoding_without_guard_bit",
    "optimal_iterations",
    "DPSSPSolver",
    "DPConfig",
]
