"""
Solver classes and utilities for the Subset Sum Problem.

Exposes:
- Quantum solvers:
    * DGSSPSolver (Draper+Grover-based)
    * QPESSPSolver (QPE-based)
- Classical solvers:
    * DPSSPSolver (dynamic-programming baseline)
- Abstract base classes for implementing new solvers.
"""

from __future__ import annotations

from .base import BaseSSPSolver, BaseQuantumSSPSolver, BaseClassicalSSPSolver
from .draper_grover import DGSSPSolver, DGConfig
from .qpe import QPESSPSolver, QPEConfig
from .classical_dp import DPSSPSolver, DPConfig

__all__ = [
    # Base classes
    "BaseSSPSolver",
    "BaseQuantumSSPSolver",
    "BaseClassicalSSPSolver",
    # Quantum solvers
    "DGSSPSolver",
    "DGConfig",
    "QPESSPSolver",
    "QPEConfig",
    # Classical solver
    "DPSSPSolver",
    "DPConfig",
]
