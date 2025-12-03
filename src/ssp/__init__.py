"""
dgssp: Draper+Grover-based Subset Sum Problem solvers.

This package provides:
- SubsetSumInstance & related data structures (dgssp.problems)
- Quantum and classical solvers for SSP (dgssp.solvers)
- Backend-agnostic execution, transpilation, and QEM utilities
- A high-level `solve_ssp(...)` function for typical use cases
"""

from __future__ import annotations

from .problems import SubsetSumInstance, SSPSolution
from .api import solve_ssp

# Package semantic version (keep in sync with pyproject.toml)
__version__ = "0.1.0"

__all__ = [
    "SubsetSumInstance",
    "SSPSolution",
    "solve_ssp",
    "__version__",
]
