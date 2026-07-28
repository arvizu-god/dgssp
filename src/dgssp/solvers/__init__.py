"""
dgssp.solvers

Solver package: the quantum Draper--Grover solver and the classical
dynamic-programming baseline, plus the abstract bases they share.

The QPE-based variant that used to live here has been removed.
"""

from __future__ import annotations

from .base import BaseClassicalSSPSolver, BaseQuantumSSPSolver, BaseSSPSolver
from .classical import DPConfig, DPSSPSolver
from .draper_grover import DGConfig, DGSSPSolver, optimal_iterations

__all__ = [
    "BaseSSPSolver",
    "BaseQuantumSSPSolver",
    "BaseClassicalSSPSolver",
    "DGSSPSolver",
    "DGConfig",
    "optimal_iterations",
    "DPSSPSolver",
    "DPConfig",
]
