"""
draper_grover.py

The Draper--Grover (D-G) Subset Sum solver: circuit construction only.

The algorithm searches the space of *subsets* with Grover's algorithm, using a
Draper (QFT) phase adder to compute the sum of the selected items into an
auxiliary register and a phase oracle to mark the register value corresponding
to the target.

One Grover iteration consists of

1. **Add**  ``QFT -> controlled phase additions (+ constant offset) -> QFT^-1``
   leaves the sum register holding ``encoded_sum(i) = sum(i) + offset``.
2. **Oracle**  a multi-controlled Z conjugated by X gates flips the phase of
   the basis state ``|encoded_target>``.
3. **Subtract**  the exact inverse of step 1, returning the sum register to
   ``|0...0>`` so that it stays unentangled from the index register.
4. **Diffuser**  the standard Grover inversion-about-the-mean on the index
   register.

Signed instances are handled entirely through :mod:`dgssp.encoding`: an offset
of ``-sum_neg`` is added as an *uncontrolled* constant so every encoded sum is
non-negative, and a guard bit guarantees the encoding never wraps.

Only the ``FullQFT`` assembly is implemented; the historical ``HalfQFT``
variant has been removed.  Execution (backends, shots, mitigation) is the
responsibility of other modules -- this one is side-effect free and returns a
plain :class:`~qiskit.circuit.QuantumCircuit`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

from ..encoding import SumRegisterEncoding, build_encoding
from ..instance import SubsetSumInstance
from .base import BaseQuantumSSPSolver


def qft_no_swaps(n: int) -> Gate:
    """
    Build the swap-free Quantum Fourier Transform on ``n`` qubits.

    The Draper adder below assumes the *no-swap* convention, in which the
    Fourier-basis phase carried by qubit ``j`` is ``2*pi*x / 2**(j+1)``.
    Building the transform explicitly -- rather than relying on a library class
    (``QFT``) and a keyword (``do_swaps``) whose home has moved between Qiskit
    releases -- keeps the adder convention pinned and version-proof.

    Parameters
    ----------
    n:
        Number of qubits.

    Returns
    -------
    Gate
        An ``n``-qubit gate labelled ``"QFT"``.
    """
    qc = QuantumCircuit(n, name="QFT")
    for j in range(n - 1, -1, -1):
        qc.h(j)
        for k in range(j):
            qc.cp(np.pi / (2 ** (j - k)), k, j)
    return qc.to_gate(label="QFT")


# ---------------------------------------------------------------------------
# Iteration-count helper
# ---------------------------------------------------------------------------


def optimal_iterations(n_ind: int, num_solutions: int) -> int:
    """
    Optimal number of Grover iterations for a known number of solutions.

    Implements ``r = floor(pi / (4 * theta))`` with
    ``theta = arcsin(sqrt(M / N))``, ``N = 2**n_ind`` and ``M = num_solutions``.

    Parameters
    ----------
    n_ind:
        Number of index qubits, i.e. ``2**n_ind`` candidate subsets.
    num_solutions:
        Number ``M`` of marked states.  Values below 1 are clamped to 1.

    Returns
    -------
    int
        Iteration count, always ``>= 1``.  Returns 1 when ``M >= N`` (the
        search is already saturated and amplification would over-rotate).
    """
    n_total = 2**int(n_ind)
    n_marked = max(1, int(num_solutions))
    if n_marked >= n_total:
        return 1
    theta = math.asin(math.sqrt(n_marked / n_total))
    return max(1, math.floor(math.pi / (4 * theta)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DGConfig:
    """
    Configuration for :class:`DGSSPSolver`.

    Attributes
    ----------
    iterations:
        Number of Grover iterations.  Either an explicit positive integer or
        the string ``"auto"``, in which case the count is derived from
        :func:`optimal_iterations` using the ``num_solutions`` argument passed
        to :meth:`DGSSPSolver.build_circuit`.
    measure:
        Whether to append measurement of the index register.  Set to ``False``
        when the circuit is to be inspected with a statevector simulator.
    """

    iterations: int | Literal["auto"] = "auto"
    measure: bool = True


# ---------------------------------------------------------------------------
# Solver implementation
# ---------------------------------------------------------------------------


class DGSSPSolver(BaseQuantumSSPSolver):
    """
    Draper--Grover Subset Sum Problem solver (circuit builder).

    Parameters
    ----------
    config:
        A :class:`DGConfig`.  Defaults to ``DGConfig()`` (``iterations="auto"``).

    Examples
    --------
    >>> from dgssp import SubsetSumInstance
    >>> from dgssp.solvers import DGSSPSolver, DGConfig
    >>> instance = SubsetSumInstance(items=[3, 5, 7, 10], target=15)
    >>> qc = DGSSPSolver(DGConfig(iterations=2)).build_circuit(instance)
    >>> qc.num_qubits > 0
    True
    """

    def __init__(self, config: DGConfig | None = None) -> None:
        self.config = config or DGConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_circuit(
        self,
        instance: SubsetSumInstance,
        *,
        num_solutions: int | None = None,
    ) -> QuantumCircuit:
        """
        Build the full Draper--Grover circuit for an instance.

        Parameters
        ----------
        instance:
            The Subset Sum instance to encode.
        num_solutions:
            Number of exact solutions ``M``, used only when
            ``config.iterations == "auto"``.  Typically supplied by the
            classical DP solver upstream (see
            :func:`dgssp.experiments.run_batch`).  If omitted while
            ``iterations == "auto"``, ``M = 1`` is assumed and a warning is
            emitted.

        Returns
        -------
        QuantumCircuit
            Circuit named ``"DGSSPSolve"`` with an index register ``i``, a sum
            register ``s`` and (unless ``config.measure`` is ``False``) a
            classical register ``c`` holding the measured index bits.  The
            circuit's ``metadata`` carries the instance parameters, the
            register widths, the iteration count and the encoding.

        Raises
        ------
        ValueError
            If ``config.iterations`` is an integer below 1, or is neither an
            integer nor ``"auto"``.
        """
        enc = build_encoding(instance.items, instance.target)
        iterations = self._resolve_iterations(enc.n_ind, num_solutions)

        if not enc.target_is_representable:
            warnings.warn(
                f"Target {instance.target} is outside the reachable sum range "
                f"[{enc.sum_neg}, {enc.sum_pos}]; the oracle will mark no "
                "state and the search cannot succeed.",
                UserWarning,
                stacklevel=2,
            )

        ind = QuantumRegister(enc.n_ind, name="i")
        summ = QuantumRegister(enc.n_sum, name="s")
        if self.config.measure:
            creg = ClassicalRegister(enc.n_ind, name="c")
            qc = QuantumCircuit(ind, summ, creg, name="DGSSPSolve")
        else:
            qc = QuantumCircuit(ind, summ, name="DGSSPSolve")

        # Uniform superposition over all subsets
        qc.h(ind)
        qc.barrier()

        step_gate = self._step_gate(enc)
        diffuser = self._grover_diffuser(enc)

        for _ in range(iterations):
            qc.append(step_gate, list(ind) + list(summ))
            qc.barrier()
            qc.append(diffuser, list(ind))
            qc.barrier()

        if self.config.measure:
            qc.measure(ind, qc.cregs[0])

        metadata = dict(qc.metadata) if qc.metadata else {}
        metadata["dgssp_instance"] = {
            "items": list(enc.items),
            "target": enc.target,
            "n_ind": enc.n_ind,
            "n_sum": enc.n_sum,
            "offset": enc.offset,
            "modulus": enc.modulus,
            "encoded_target": enc.encoded_target,
            "iterations": iterations,
            "assembly": "FullQFT",
        }
        qc.metadata = metadata
        return qc

    def build_step_circuit(self, instance: SubsetSumInstance) -> QuantumCircuit:
        """
        Build a single add -> oracle -> subtract step as a standalone circuit.

        Useful for the uncomputation test and for circuit diagrams.

        Parameters
        ----------
        instance:
            The Subset Sum instance to encode.

        Returns
        -------
        QuantumCircuit
            An ``(n_ind + n_sum)``-qubit circuit with no measurements.
        """
        enc = build_encoding(instance.items, instance.target)
        ind = QuantumRegister(enc.n_ind, name="i")
        summ = QuantumRegister(enc.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="DGSSP_Step")
        qc.append(self._step_gate(enc), list(ind) + list(summ))
        return qc

    def encoding_for(self, instance: SubsetSumInstance) -> SumRegisterEncoding:
        """
        Expose the register encoding used for an instance.

        Parameters
        ----------
        instance:
            The Subset Sum instance.

        Returns
        -------
        SumRegisterEncoding
            The offset/guard-bit encoding, for callers that need to interpret
            the sum register directly.
        """
        return build_encoding(instance.items, instance.target)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_iterations(self, n_ind: int, num_solutions: int | None) -> int:
        """Turn ``config.iterations`` into a concrete positive integer."""
        iters = self.config.iterations
        if iters == "auto":
            if num_solutions is None:
                warnings.warn(
                    "DGConfig.iterations='auto' but num_solutions was not "
                    "provided; assuming a single solution (M=1). Pass "
                    "num_solutions=... for the optimal iteration count.",
                    UserWarning,
                    stacklevel=3,
                )
                num_solutions = 1
            return optimal_iterations(n_ind, num_solutions)
        if isinstance(iters, bool) or not isinstance(iters, int):
            raise ValueError(
                f"DGConfig.iterations must be a positive int or 'auto', got {iters!r}."
            )
        if iters < 1:
            raise ValueError("DGConfig.iterations must be >= 1.")
        return iters

    # ----- Gate-building helpers --------------------------------------

    def _sum_gate(self, enc: SumRegisterEncoding) -> Gate:
        """
        Fourier-basis adder: adds every selected item plus the constant offset.

        Parameters
        ----------
        enc:
            The register encoding.

        Returns
        -------
        Gate
            An ``(n_ind + n_sum)``-qubit gate labelled ``"SumGate"``, to be
            applied between a QFT and an inverse QFT on the sum register.
        """
        ind = QuantumRegister(enc.n_ind, name="i")
        summ = QuantumRegister(enc.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="SumGate")
        mod = enc.modulus

        for k, a in enumerate(enc.items):
            a_mod = a % mod
            if a_mod == 0:
                continue
            for j in range(enc.n_sum):
                qc.cp((2 * np.pi * a_mod) / (2 ** (j + 1)), ind[k], summ[j])

        # Uncontrolled constant offset so that encoded sums are non-negative.
        off = enc.offset % mod
        if off:
            for j in range(enc.n_sum):
                qc.p((2 * np.pi * off) / (2 ** (j + 1)), summ[j])

        return qc.to_gate(label="SumGate")

    def _sub_gate(self, enc: SumRegisterEncoding) -> Gate:
        """
        Exact inverse of :meth:`_sum_gate`, restoring the sum register to zero.

        Parameters
        ----------
        enc:
            The register encoding.

        Returns
        -------
        Gate
            An ``(n_ind + n_sum)``-qubit gate labelled ``"SubGate"``.
        """
        ind = QuantumRegister(enc.n_ind, name="i")
        summ = QuantumRegister(enc.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="SubGate")
        mod = enc.modulus

        for k, a in enumerate(enc.items):
            neg = (-a) % mod
            if neg == 0:
                continue
            for j in range(enc.n_sum):
                qc.cp((2 * np.pi * neg) / (2 ** (j + 1)), ind[k], summ[j])

        neg_off = (-enc.offset) % mod
        if neg_off:
            for j in range(enc.n_sum):
                qc.p((2 * np.pi * neg_off) / (2 ** (j + 1)), summ[j])

        return qc.to_gate(label="SubGate")

    def _oracle_gate(self, enc: SumRegisterEncoding) -> Gate:
        """
        Phase oracle marking ``|encoded_target>`` in the sum register.

        Parameters
        ----------
        enc:
            The register encoding (supplies ``encoded_target``, which is
            guaranteed non-negative).

        Returns
        -------
        Gate
            An ``n_sum``-qubit gate labelled ``"OracleGate"`` implementing a
            multi-controlled Z conjugated by X gates.
        """
        summ = QuantumRegister(enc.n_sum, name="s")
        qc = QuantumCircuit(summ, name="OracleGate")
        t = enc.encoded_target

        zeros = [j for j in range(enc.n_sum) if ((t >> j) & 1) == 0]
        for j in zeros:
            qc.x(summ[j])

        self._multi_controlled_z(qc, list(summ))

        for j in zeros:
            qc.x(summ[j])

        return qc.to_gate(label="OracleGate")

    def _grover_diffuser(self, enc: SumRegisterEncoding) -> Gate:
        """
        Grover diffuser (inversion about the mean) on the index register.

        Parameters
        ----------
        enc:
            The register encoding (supplies ``n_ind``).

        Returns
        -------
        Gate
            An ``n_ind``-qubit gate labelled ``"GroverDiffuser"``.
        """
        ind = QuantumRegister(enc.n_ind, name="i")
        qc = QuantumCircuit(ind, name="GroverDiffuser")

        qc.h(ind)
        qc.x(ind)
        self._multi_controlled_z(qc, list(ind))
        qc.x(ind)
        qc.h(ind)

        return qc.to_gate(label="GroverDiffuser")

    @staticmethod
    def _multi_controlled_z(qc: QuantumCircuit, qubits: list) -> None:
        """
        Apply a multi-controlled Z across ``qubits`` in place.

        Uses ``H - MCX - H`` on the last qubit.  Handles the degenerate widths
        (1 and 2 qubits) that would otherwise make ``mcx`` fail with an empty
        control list.

        Parameters
        ----------
        qc:
            Circuit to modify.
        qubits:
            The qubits spanned by the multi-controlled Z.

        Returns
        -------
        None
        """
        if len(qubits) == 1:
            qc.z(qubits[0])
        elif len(qubits) == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])

    def _step_gate(self, enc: SumRegisterEncoding) -> Gate:
        """
        Assemble one D-G step: add -> oracle -> subtract (FullQFT assembly).

        Parameters
        ----------
        enc:
            The register encoding.

        Returns
        -------
        Gate
            An ``(n_ind + n_sum)``-qubit gate labelled ``"DGSSP_Step"``.
        """
        ind = QuantumRegister(enc.n_ind, name="i")
        summ = QuantumRegister(enc.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="DGSSP_Step")

        qft = qft_no_swaps(enc.n_sum)
        iqft = qft.inverse()
        iqft.label = "iQFT"

        # Add
        qc.append(qft, list(summ))
        qc.append(self._sum_gate(enc), list(ind) + list(summ))
        qc.append(iqft, list(summ))

        # Mark
        qc.append(self._oracle_gate(enc), list(summ))

        # Subtract (exact inverse of the add block)
        qc.append(qft, list(summ))
        qc.append(self._sub_gate(enc), list(ind) + list(summ))
        qc.append(iqft, list(summ))

        return qc.to_gate(label="DGSSP_Step")


__all__ = [
    "DGConfig",
    "DGSSPSolver",
    "optimal_iterations",
    "qft_no_swaps",
]
