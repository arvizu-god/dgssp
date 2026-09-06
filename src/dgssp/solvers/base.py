"""
base.py

Abstract base classes for Subset Sum Problem (SSP) solvers.

We distinguish between:
- BaseSSPSolver: common interface for all SSP solvers.
- BaseQuantumSSPSolver: solvers that build quantum circuits.
- BaseClassicalSSPSolver: solvers that run purely classical algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qiskit import QuantumCircuit

from ..instance import DPResult, SubsetSumInstance


class BaseSSPSolver(ABC):
    """
    Common interface for all SSP solvers (quantum and classical).

    At minimum, a solver must implement `solve(instance, ...)` which
    returns some form of result object (classical or quantum).
    """

    @abstractmethod
    def solve(self, instance: SubsetSumInstance, *args: Any, **kwargs: Any) -> Any:
        """
        Solve the given SubsetSumInstance.

        Subclasses decide what they return:
        - Classical solvers: typically a DPResult.
        - Quantum solvers: some QuantumSSPResult (to be defined later).

        Parameters
        ----------
        instance:
            The SSP instance to solve.
        """
        raise NotImplementedError


class BaseQuantumSSPSolver(BaseSSPSolver, ABC):
    """
    Base class for quantum SSP solvers.

    Quantum solvers at least need to know how to build the quantum
    circuit corresponding to a given instance. How that circuit is
    executed (backend, shots, mitigation, etc.) is delegated to other
    parts of the library.
    """

    @abstractmethod
    def build_circuit(
        self, instance: SubsetSumInstance, *, num_solutions: int | None = None
    ) -> QuantumCircuit:
        """
        Build the quantum circuit encoding the search for a given SSP instance.

        Parameters
        ----------
        instance:
            The Subset Sum instance to encode.
        num_solutions:
            ``M``, the number of correct outcomes, when the implementation
            needs it to choose an iteration count.  ``None`` means "assume
            one" or "the configuration already fixes the iterations".

        Returns
        -------
        QuantumCircuit
            Circuit ready to be transpiled / executed by the execution layer.
        """
        raise NotImplementedError

    # Default `solve` implementation: for now we *only* build the circuit.
    # Later we can override this to accept an Executor and Mitigator if desired.
    def solve(self, instance: SubsetSumInstance, *args: Any, **kwargs: Any) -> QuantumCircuit:
        """
        By default, quantum solvers just build and return the circuit.

        The idea is that users (or higher-level API functions) will pass
        this circuit to an Executor which takes care of running it on a
        backend and decoding the results.
        """
        return self.build_circuit(instance, **kwargs)


class BaseClassicalSSPSolver(BaseSSPSolver, ABC):
    """
    Base class for classical SSP solvers.

    Classical solvers usually return a DPResult which contains the
    best solution, all solutions, and some metadata.
    """

    @abstractmethod
    def solve(self, instance: SubsetSumInstance) -> DPResult:
        """
        Solve the SSP instance using a classical algorithm.

        Returns
        -------
        DPResult
            Result of the classical solver.
        """
        raise NotImplementedError


__all__ = [
    "BaseSSPSolver",
    "BaseQuantumSSPSolver",
    "BaseClassicalSSPSolver",
]
