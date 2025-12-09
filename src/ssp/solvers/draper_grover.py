"""
draper_grover.py

Draper–Grover Subset Sum Problem (SSP) solver.

This module defines the DGSSPSolver class, which constructs a Draper-style
QFT adder/subtractor combined with a Grover search over the index register.

The solver is purely circuit-building: it takes a SubsetSumInstance and
returns a QuantumCircuit implementing the Grover search for that instance.
Execution (choice of backend, shots, QEM, etc.) is handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import QFT

from ..instance import SubsetSumInstance
from .base import BaseQuantumSSPSolver  # will be defined in solvers/base.py


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DGConfig:
    """
    Configuration options for the Draper–Grover SSP solver.

    Attributes
    ----------
    assembly_type:
        How to assemble the sum register evolution:
        - "FullQFT": use QFT and inverse-QFT for both add and subtract.
        - "HalfQFT": use Hadamards in place of the initial QFT, following
                     the "Half QFT" variant.
    iterations:
        Number of Grover iterations to apply. For now this must be set
        explicitly by the user. In future we may support automatic
        selection based on the number of expected solutions.
    """

    assembly_type: Literal["FullQFT", "HalfQFT"] = "FullQFT"
    iterations: int = 1


@dataclass
class _DGLayout:
    """
    Internal helper describing the layout and numeric parameters for
    a given SubsetSumInstance.
    """

    items: list[int]
    target: int
    sum_neg: int
    sum_pos: int
    range_len: int
    n_sum: int
    n_ind: int
    qft_gate: Gate
    iqft_gate: Gate


# ---------------------------------------------------------------------------
# Solver implementation
# ---------------------------------------------------------------------------


class DGSSPSolver(BaseQuantumSSPSolver):
    """
    Draper–Grover Subset Sum Problem solver.

    This class encapsulates:
    - Construction of controlled-phase sum and subtract gates.
    - Construction of the phase oracle marking the target sum.
    - Construction of the Grover diffuser on the index register.
    - Assembly of a single DGSSP step (add → oracle → subtract).
    - Assembly of the full Grover loop (with measurements).

    Usage
    -----
    >>> from dgssp.instance import SubsetSumInstance
    >>> from dgssp.solvers import DGSSPSolver, DGConfig
    >>>
    >>> instance = SubsetSumInstance(items=[3, 5, 7, 10], target=15)
    >>> solver = DGSSPSolver(DGConfig(assembly_type="FullQFT", iterations=2))
    >>> qc = solver.build_circuit(instance)
    >>> qc.draw("mpl")
    """

    def __init__(self, config: DGConfig | None = None) -> None:
        self.config = config or DGConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_circuit(self, instance: SubsetSumInstance) -> QuantumCircuit:
        """
        Build the full Draper–Grover circuit for a given SSP instance.

        The resulting circuit:
        - Uses an index register of n_ind qubits.
        - Uses a sum register of n_sum qubits.
        - Applies Hadamards to the index register to create a uniform
          superposition.
        - Applies `config.iterations` Grover iterations, each consisting of:
            * DG step (add → oracle → subtract),
            * Grover diffuser on the index register.
        - Measures the index register into classical bits.

        Parameters
        ----------
        instance:
            The SubsetSumInstance to encode and solve.

        Returns
        -------
        QuantumCircuit
            A circuit named "DGSSPSolve" ready to be executed on a backend.
        """
        layout = self._build_layout(instance)

        ind = QuantumRegister(layout.n_ind, name="i")
        summ = QuantumRegister(layout.n_sum, name="s")
        creg = ClassicalRegister(layout.n_ind, name="c")
        qc = QuantumCircuit(ind, summ, creg, name="DGSSPSolve")

        # Initialize the index register in uniform superposition
        qc.h(ind)
        qc.barrier()

        # Prebuild one DG step gate and the diffuser
        step_gate = self._build_step_gate(layout).to_gate(label="DGSSP_Step")
        diffuser = self._grover_diffuser(layout).to_gate(label="GroverDiffuser")

        # Grover iterations
        if self.config.iterations < 1:
            raise ValueError("DGConfig.iterations must be >= 1.")
        for _ in range(self.config.iterations):
            qc.append(step_gate, ind[:] + summ[:])
            qc.barrier()
            qc.append(diffuser, ind[:])
            qc.barrier()

        # Measure index register
        qc.measure(ind, creg)

        # Attach minimal metadata so downstream utilities can recover
        # the original instance parameters from the circuit.
        metadata = dict(qc.metadata) if qc.metadata is not None else {}
        metadata.setdefault("dgssp_instance", {})
        metadata["dgssp_instance"].update(
            {
                "items": list(layout.items),
                "target": layout.target,
                "assembly_type": self.config.assembly_type,
                "n_ind": layout.n_ind,
                "n_sum": layout.n_sum,
            }
        )
        qc.metadata = metadata

        return qc

    # NOTE: The BaseQuantumSSPSolver interface may define a `solve(...)`
    # method taking an executor and mitigation object. That can be
    # implemented later on top of `build_circuit`. For now, this class
    # focuses solely on circuit generation.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_layout(self, instance: SubsetSumInstance) -> _DGLayout:
        """
        Compute the numeric and register layout parameters for a given instance.
        """
        items = list(instance.items)
        target = int(instance.target)

        sum_neg = sum(a for a in items if a < 0)
        sum_pos = sum(a for a in items if a > 0)
        range_len = sum_pos - sum_neg + 1

        if range_len <= 0:
            raise ValueError(
                "Invalid range length for sum register: "
                f"sum_pos={sum_pos}, sum_neg={sum_neg}."
            )

        n_sum = int(np.ceil(np.log2(range_len)))
        if n_sum < 1:
            n_sum = 1

        n_ind = len(items)

        # QFT gates on the sum register
        qft = QFT(n_sum, do_swaps=False).to_gate(label="QFT")
        iqft = qft.inverse()
        iqft.label = "iQFT"

        return _DGLayout(
            items=items,
            target=target,
            sum_neg=sum_neg,
            sum_pos=sum_pos,
            range_len=range_len,
            n_sum=n_sum,
            n_ind=n_ind,
            qft_gate=qft,
            iqft_gate=iqft,
        )

    # ----- Gate-building helpers --------------------------------------

    def _sum_gate(self, layout: _DGLayout) -> Gate:
        """
        Controlled-phase adder for items into the sum register.

        This gate assumes:
        - index register has layout.n_ind qubits
        - sum register has layout.n_sum qubits
        """
        ind = QuantumRegister(layout.n_ind, name="i")
        summ = QuantumRegister(layout.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="SumGate")

        modulus = 2 ** layout.n_sum

        for k, a in enumerate(layout.items):
            a_mod = a % modulus
            for j in range(layout.n_sum):
                phi = (2 * np.pi * a_mod) / (2 ** (j + 1))
                qc.cp(phi, ind[k], summ[j])

        return qc.to_gate(label="SumGate")

    def _sub_gate(self, layout: _DGLayout) -> Gate:
        """
        Controlled-phase subtractor for items from the sum register.
        """
        ind = QuantumRegister(layout.n_ind, name="i")
        summ = QuantumRegister(layout.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="SubGate")

        modulus = 2 ** layout.n_sum

        for k, a in enumerate(layout.items):
            neg = (-a) % modulus
            for j in range(layout.n_sum):
                phi = (2 * np.pi * neg) / (2 ** (j + 1))
                qc.cp(phi, ind[k], summ[j])

        return qc.to_gate(label="SubGate")

    def _oracle_gate(self, layout: _DGLayout) -> Gate:
        """
        Phase oracle that flips the phase of |s⟩ = |target⟩ in the sum register.
        """
        summ = QuantumRegister(layout.n_sum, name="s")
        qc = QuantumCircuit(summ, name="OracleGate")

        # Flip the bits where target has a 0
        for j in range(layout.n_sum):
            if ((layout.target >> j) & 1) == 0:
                qc.x(summ[j])

        # Multi-controlled Z via H–MCX–H on the last qubit
        qc.h(summ[-1])
        qc.mcx(summ[:-1], summ[-1], mode="noancilla")
        qc.h(summ[-1])

        # Unflip
        for j in range(layout.n_sum):
            if ((layout.target >> j) & 1) == 0:
                qc.x(summ[j])

        return qc.to_gate(label="OracleGate")

    def _grover_diffuser(self, layout: _DGLayout) -> Gate:
        """
        Standard Grover diffuser on the n_ind-qubit index register.
        """
        ind = QuantumRegister(layout.n_ind, name="i")
        qc = QuantumCircuit(ind, name="GroverDiffuser")

        # H–X on all
        for qb in ind:
            qc.h(qb)
            qc.x(qb)

        # Multi-controlled Z
        qc.h(ind[-1])
        qc.mcx(ind[:-1], ind[-1], mode="noancilla")
        qc.h(ind[-1])

        # X–H on all
        for qb in ind:
            qc.x(qb)
            qc.h(qb)

        return qc

    def _build_step_gate(self, layout: _DGLayout) -> QuantumCircuit:
        """
        Build a single DGSSP step circuit (add → oracle → subtract),
        using either the FullQFT or HalfQFT assembly strategy.
        """
        ind = QuantumRegister(layout.n_ind, name="i")
        summ = QuantumRegister(layout.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="DGSSP_Step")

        sum_gate = self._sum_gate(layout)
        sub_gate = self._sub_gate(layout)
        oracle_gate = self._oracle_gate(layout)

        if self.config.assembly_type == "FullQFT":
            # Add
            qc.append(layout.qft_gate, summ[:])
            qc.append(sum_gate, ind[:] + summ[:])
            qc.append(layout.iqft_gate, summ[:])

            # Oracle
            qc.append(oracle_gate, summ[:])

            # Subtract
            qc.append(layout.qft_gate, summ[:])
            qc.append(sub_gate, ind[:] + summ[:])
            qc.append(layout.iqft_gate, summ[:])

        elif self.config.assembly_type == "HalfQFT":
            # Add
            qc.h(summ[:])
            qc.append(sum_gate, ind[:] + summ[:])
            qc.append(layout.iqft_gate, summ[:])

            # Oracle
            qc.append(oracle_gate, summ[:])

            # Subtract
            qc.append(layout.iqft_gate.inverse(), summ[:])
            qc.append(sub_gate, ind[:] + summ[:])
            qc.h(summ[:])

        else:
            raise ValueError(
                f"Unknown assembly_type '{self.config.assembly_type}'. "
                "Supported: 'FullQFT', 'HalfQFT'."
            )

        return qc


__all__ = [
    "DGConfig",
    "DGSSPSolver",
]
