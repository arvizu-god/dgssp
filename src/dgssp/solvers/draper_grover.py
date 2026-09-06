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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# Resource model (contribution C1)
# ---------------------------------------------------------------------------

#: Names of the composite gates :class:`DGSSPSolver` assembles the circuit
#: from.  ``QFT_dg`` is the name ``Gate.inverse()`` gives the inverse
#: transform; it is listed explicitly rather than left to be matched by the
#: ``"QFT"`` pattern.
CONTAINER_GATE_NAMES: tuple[str, ...] = (
    "DGSSP_Step",
    "GroverDiffuser",
    "QFT",
    "QFT_dg",
    "SumGate",
    "SubGate",
    "OracleGate",
)


def flatten_containers(circuit: QuantumCircuit, *, max_rounds: int = 8) -> QuantumCircuit:
    """
    Expand every composite gate of the D-G circuit, and nothing else.

    ``QuantumCircuit.decompose()`` with no argument peels one layer off *every*
    instruction, so it turns ``h`` into ``u`` and opens ``mcx`` on the same
    pass that opens the adder.  This helper names the composites explicitly and
    repeats until none remain, leaving the circuit at the *gate level* --
    ``{cp, p, h, x, z, cz, ccx, mcx, measure, barrier}`` -- which is the level
    the closed-form inventory in :func:`resource_model` counts.

    Parameters
    ----------
    circuit:
        A circuit built by :class:`DGSSPSolver`.
    max_rounds:
        Safety bound on the number of decomposition rounds.

    Returns
    -------
    QuantumCircuit
        The flattened circuit.

    Raises
    ------
    RuntimeError
        If composites remain after ``max_rounds`` rounds, which would mean
        :data:`CONTAINER_GATE_NAMES` has gone stale.
    """
    current = circuit
    names = set(CONTAINER_GATE_NAMES)
    for _ in range(max_rounds):
        present = {instruction.operation.name for instruction in current.data}
        if not (present & names):
            return current
        current = current.decompose(list(CONTAINER_GATE_NAMES))

    remaining = sorted({i.operation.name for i in current.data} & names)
    raise RuntimeError(
        f"Composite gates {remaining} survived {max_rounds} decomposition "
        "rounds; CONTAINER_GATE_NAMES is out of date."
    )


def two_adic_valuation(value: int) -> int:
    """
    Largest ``v`` such that ``2**v`` divides ``value``.

    Parameters
    ----------
    value:
        A non-zero integer.

    Returns
    -------
    int
        The 2-adic valuation.

    Raises
    ------
    ValueError
        If ``value`` is zero, whose valuation is undefined.
    """
    if value == 0:
        raise ValueError("two_adic_valuation is undefined for 0.")
    return int((value & -value).bit_length() - 1)


@dataclass(frozen=True)
class DGResourceModel:
    """
    Closed-form gate inventory of a D-G circuit, per iteration and in total.

    Counts refer to the circuit *at the gate level*: after
    :func:`flatten_containers`, before any basis translation.  They are exact,
    not bounds, and they reproduce ``flatten_containers(circuit).count_ops()``
    term by term.

    Attributes
    ----------
    n_ind, n_sum:
        Index- and sum-register widths (``n`` and ``b`` in the note).
    iterations:
        Grover iterations ``r`` the circuit was built with.
    nonzero_items:
        Number of items whose residue ``a mod 2**b`` is non-zero.  An item with
        a zero residue is skipped outright by :meth:`DGSSPSolver._sum_gate` and
        costs nothing.
    offset_is_nonzero:
        Whether the uncontrolled offset addition is emitted at all.  It is
        emitted exactly when the instance has a negative item.
    target_zero_bits:
        Number of zero bits of ``encoded_target`` within the ``b``-bit window;
        each contributes two ``X`` gates to the oracle.
    qft_cp_per_iteration:
        Controlled phases from the four QFT-type blocks.
    adder_cp_per_iteration:
        Controlled phases from the two controlled-addition blocks.
    adder_cp_per_iteration_skippable:
        What ``adder_cp_per_iteration`` would be if rotations that are exact
        multiples of ``2*pi`` were dropped.  The implementation does *not* drop
        them, so this is a lower bound the current code does not reach; it is
        recorded to size the missed saving.
    offset_p_per_iteration, offset_p_per_iteration_skippable:
        The same pair for the uncontrolled offset addition's single-qubit
        phases.
    hadamard_per_iteration, pauli_x_per_iteration:
        Single-qubit gate counts inside one iteration.  The initial Hadamard
        layer on the index register is outside the loop and is added by
        :attr:`hadamard`.
    mcz_widths:
        Widths of the two multi-controlled-Z blocks per iteration, oracle
        first: ``(n_sum, n_ind)``.
    """

    n_ind: int
    n_sum: int
    iterations: int
    nonzero_items: int
    offset_is_nonzero: bool
    target_zero_bits: int
    qft_cp_per_iteration: int
    adder_cp_per_iteration: int
    adder_cp_per_iteration_skippable: int
    offset_p_per_iteration: int
    offset_p_per_iteration_skippable: int
    hadamard_per_iteration: int
    pauli_x_per_iteration: int
    mcz_widths: tuple[int, int]

    # -- derived totals --------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Total qubits, ``n_ind + n_sum``.  The construction uses no ancilla."""
        return self.n_ind + self.n_sum

    @property
    def num_ancillas(self) -> int:
        """Ancilla qubits, of which this construction uses none."""
        return 0

    @property
    def controlled_phase(self) -> int:
        """Total ``cp`` gates: ``r * (QFT block + adder block)``."""
        return self.iterations * (self.qft_cp_per_iteration + self.adder_cp_per_iteration)

    @property
    def controlled_phase_skippable(self) -> int:
        """``controlled_phase`` with the ``2*pi`` multiples dropped."""
        return self.iterations * (
            self.qft_cp_per_iteration + self.adder_cp_per_iteration_skippable
        )

    @property
    def phase(self) -> int:
        """Total uncontrolled ``p`` gates: the offset addition and its inverse."""
        return self.iterations * self.offset_p_per_iteration

    @property
    def phase_skippable(self) -> int:
        """``phase`` with the ``2*pi`` multiples dropped."""
        return self.iterations * self.offset_p_per_iteration_skippable

    @property
    def hadamard(self) -> int:
        """Total ``h`` gates, including the initial superposition layer."""
        return self.n_ind + self.iterations * self.hadamard_per_iteration

    @property
    def pauli_x(self) -> int:
        """Total ``x`` gates."""
        return self.iterations * self.pauli_x_per_iteration

    @property
    def multi_controlled_z(self) -> int:
        """Number of multi-controlled-Z blocks: two per iteration."""
        return 2 * self.iterations

    def gate_level_counts(self) -> dict[str, int]:
        """
        Predicted ``count_ops()`` of the flattened circuit for the plain gates.

        Returns
        -------
        dict[str, int]
            Keys ``"cp"``, ``"p"``, ``"h"``, ``"x"``.  The multi-controlled
            blocks are excluded because their ``count_ops`` name depends on
            their width; :meth:`multi_controlled_op_counts` resolves those.
        """
        return {
            "cp": self.controlled_phase,
            "p": self.phase,
            "h": self.hadamard,
            "x": self.pauli_x,
        }

    def multi_controlled_op_counts(self) -> dict[str, int]:
        """
        Predicted ``count_ops()`` contribution of the two MCZ blocks.

        Returns
        -------
        dict[str, int]
            ``{"z": ...}`` at width 1, ``{"cz": ...}`` at width 2,
            ``{"ccx": ...}`` at width 3 (Qiskit names a two-control ``mcx``
            ``ccx``) and ``{"mcx": ...}`` above that, summed over both blocks
            and all iterations.
        """
        counts: dict[str, int] = {}
        for width in self.mcz_widths:
            if width == 1:
                name = "z"
            elif width == 2:
                name = "cz"
            elif width == 3:
                name = "ccx"
            else:
                name = "mcx"
            counts[name] = counts.get(name, 0) + self.iterations
        return counts

    def two_qubit_reference(self, mcz_cx_cost: Callable[[int], int]) -> int:
        """
        Two-qubit gate count after decomposition to ``{CX, 1Q}``.

        Every ``cp`` costs two ``CX``; each MCZ costs whatever the supplied
        synthesis charges for its width.

        Parameters
        ----------
        mcz_cx_cost:
            Callable ``width -> CX count`` for a multi-controlled Z.  Pass
            :func:`dgssp.transpilation.mcz_noaux_cx_cost` for the paper's
            ancilla-free reference synthesis.

        Returns
        -------
        int
            Reference two-qubit count for the whole circuit.
        """
        per_iteration = (
            2 * (self.qft_cp_per_iteration + self.adder_cp_per_iteration)
            + mcz_cx_cost(self.mcz_widths[0])
            + mcz_cx_cost(self.mcz_widths[1])
        )
        return self.iterations * per_iteration

    def one_qubit_reference(self, mcx_1q_cost: Callable[[int], int]) -> int:
        """
        Single-qubit gate count after decomposition to ``{CX, 1Q}``.

        Each ``cp`` contributes three single-qubit phases, each ``p`` one, and
        the ``h`` and ``x`` gates one each.  The MCZ blocks contribute only
        their *interior*: the two Hadamards that conjugate the ``mcx`` are
        already inside :attr:`hadamard_per_iteration`, so adding a whole-MCZ
        cost here would double-count them.

        Parameters
        ----------
        mcx_1q_cost:
            Callable ``num_controls -> single-qubit gate count`` for an
            ancilla-free multi-controlled X.  Pass
            :func:`dgssp.transpilation.mcx_noaux_1q_cost`.

        Returns
        -------
        int
            Reference single-qubit count for the whole circuit.
        """

        def interior(width: int) -> int:
            if width == 1:
                return 1  # a bare Z is one single-qubit rotation
            if width == 2:
                return 2  # CZ translates to one CX plus two rotations
            return mcx_1q_cost(width - 1)

        per_iteration = (
            3 * (self.qft_cp_per_iteration + self.adder_cp_per_iteration)
            + self.offset_p_per_iteration
            + self.hadamard_per_iteration
            + self.pauli_x_per_iteration
            + interior(self.mcz_widths[0])
            + interior(self.mcz_widths[1])
        )
        return self.n_ind + self.iterations * per_iteration

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-serialisable view: every stored field plus every derived total.

        Returns
        -------
        dict
            Suitable for embedding in a results record.
        """
        return {
            "n_ind": self.n_ind,
            "n_sum": self.n_sum,
            "iterations": self.iterations,
            "nonzero_items": self.nonzero_items,
            "offset_is_nonzero": self.offset_is_nonzero,
            "target_zero_bits": self.target_zero_bits,
            "qft_cp_per_iteration": self.qft_cp_per_iteration,
            "adder_cp_per_iteration": self.adder_cp_per_iteration,
            "adder_cp_per_iteration_skippable": self.adder_cp_per_iteration_skippable,
            "offset_p_per_iteration": self.offset_p_per_iteration,
            "offset_p_per_iteration_skippable": self.offset_p_per_iteration_skippable,
            "hadamard_per_iteration": self.hadamard_per_iteration,
            "pauli_x_per_iteration": self.pauli_x_per_iteration,
            "mcz_widths": list(self.mcz_widths),
            "num_qubits": self.num_qubits,
            "num_ancillas": self.num_ancillas,
            "controlled_phase": self.controlled_phase,
            "controlled_phase_skippable": self.controlled_phase_skippable,
            "phase": self.phase,
            "phase_skippable": self.phase_skippable,
            "hadamard": self.hadamard,
            "pauli_x": self.pauli_x,
            "multi_controlled_z": self.multi_controlled_z,
        }


def resource_model(encoding: SumRegisterEncoding, iterations: int = 1) -> DGResourceModel:
    """
    Closed-form gate inventory of the circuit :class:`DGSSPSolver` would build.

    The counts are derived from the construction in this module, not measured:
    four QFT-type blocks of ``b(b-1)/2`` controlled phases each, two
    controlled-addition blocks of ``b`` controlled phases per non-zero item, an
    uncontrolled offset addition and its inverse, one multi-controlled Z on the
    sum register (the oracle) and one on the index register (the diffuser).

    See ``paper/notes/resource_model.md`` for the derivation and
    ``paper/scripts/resource_scaling.py`` for the check against the circuit the
    solver actually builds.

    Parameters
    ----------
    encoding:
        The register encoding from :func:`dgssp.encoding.build_encoding`.
    iterations:
        Grover iterations ``r``.

    Returns
    -------
    DGResourceModel
        The inventory.

    Raises
    ------
    ValueError
        If ``iterations`` is below 1.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")

    n = encoding.n_ind
    b = encoding.n_sum
    modulus = encoding.modulus

    residues = [a % modulus for a in encoding.items]
    nonzero = [r for r in residues if r]
    offset = encoding.offset % modulus
    zero_bits = sum(1 for j in range(b) if not ((encoding.encoded_target >> j) & 1))

    # Four QFT-type blocks per iteration: two forward, two inverse.
    qft_cp = 4 * (b * (b - 1) // 2)

    # Add and uncompute: b controlled phases per non-zero item, in each block.
    adder_cp = 2 * len(nonzero) * b
    adder_cp_min = 2 * sum(b - two_adic_valuation(r) for r in nonzero)

    offset_p = 2 * b if offset else 0
    offset_p_min = 2 * (b - two_adic_valuation(offset)) if offset else 0

    # Hadamards: four QFT blocks, the oracle's MCZ, and the diffuser.
    hadamard = 4 * b + (2 if b >= 3 else 0) + 2 * n + (2 if n >= 3 else 0)
    # X gates: the oracle's zero-bit conjugation and the diffuser's.
    pauli_x = 2 * zero_bits + 2 * n

    return DGResourceModel(
        n_ind=n,
        n_sum=b,
        iterations=int(iterations),
        nonzero_items=len(nonzero),
        offset_is_nonzero=bool(offset),
        target_zero_bits=zero_bits,
        qft_cp_per_iteration=qft_cp,
        adder_cp_per_iteration=adder_cp,
        adder_cp_per_iteration_skippable=adder_cp_min,
        offset_p_per_iteration=offset_p,
        offset_p_per_iteration_skippable=offset_p_min,
        hadamard_per_iteration=hadamard,
        pauli_x_per_iteration=pauli_x,
        mcz_widths=(b, n),
    )


def encoding_without_guard_bit(encoding: SumRegisterEncoding) -> SumRegisterEncoding:
    """
    The same encoding with the guard bit removed, so its cost can be priced.

    The result is *not* a usable encoding -- without the guard bit an encoded
    sum can alias a different reachable sum, which is exactly why
    :func:`dgssp.encoding.build_encoding` adds the bit.  It is, however, the
    right counterfactual for "what does the guard bit cost?", which is answered
    by differencing two :func:`resource_model` calls.

    Parameters
    ----------
    encoding:
        A real encoding, with ``n_sum >= 2``.

    Returns
    -------
    SumRegisterEncoding
        A copy one bit narrower.

    Raises
    ------
    ValueError
        If ``encoding.n_sum`` is already 1.
    """
    if encoding.n_sum < 2:
        raise ValueError("Cannot remove the guard bit from a 1-qubit sum register.")
    return replace(
        encoding,
        n_sum=encoding.n_sum - 1,
        modulus=2 ** (encoding.n_sum - 1),
    )


__all__ = [
    "CONTAINER_GATE_NAMES",
    "DGConfig",
    "DGResourceModel",
    "DGSSPSolver",
    "encoding_without_guard_bit",
    "flatten_containers",
    "optimal_iterations",
    "qft_no_swaps",
    "resource_model",
    "two_adic_valuation",
]
