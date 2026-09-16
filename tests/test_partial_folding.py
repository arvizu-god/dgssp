"""
Tests for real-valued (partial) ISA folding and the transpile-once reproduction.

The contract of :func:`dgssp.mitigation.zne.fold_isa`:

* the realised two-qubit scale is exact, rational and within 1/N_2q of the
  nominal one;
* odd integers reproduce the integer-only implementation that preceded it,
  gate for gate;
* the output is a deterministic function of (circuit, scale, seed);
* after the basis rebase the folded circuit is ISA-valid for the backend,
  with no re-transpilation.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from dgssp import DGConfig, DGSSPSolver, SubsetSumInstance
from dgssp.mitigation import mitiq_is_available
from dgssp.mitigation.zne import (
    ZNESamplingConfig,
    fold_isa,
    fold_plan,
    fold_transpiled,
    realised_scale,
    rebase_to_backend,
    transpile_once_seeded,
)
from dgssp.transpilation import transpiled_metrics, two_qubit_count

SUBSET_FILE = Path(__file__).resolve().parents[1] / "paper" / "instances" / "hardware_subset.json"


def _legacy_fold(tqc: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """The integer-only fold_transpiled of commit 26ee9b6, verbatim in behaviour."""
    if scale_factor == 1:
        return tqc.copy()
    n_repeat = (scale_factor - 1) // 2
    folded = QuantumCircuit(*tqc.qregs, *tqc.cregs, name=f"{tqc.name}_x{scale_factor}")
    qubit_map = dict(zip(tqc.qubits, folded.qubits, strict=True))
    clbit_map = dict(zip(tqc.clbits, folded.clbits, strict=True))
    for instruction in tqc.data:
        op = instruction.operation
        qargs = [qubit_map[q] for q in instruction.qubits]
        cargs = [clbit_map[c] for c in instruction.clbits]
        folded.append(op, qargs, cargs)
        if op.name in ("measure", "barrier", "delay", "reset"):
            continue
        for _ in range(n_repeat):
            folded.append(op.inverse(), qargs, cargs)
            folded.append(op, qargs, cargs)
    if getattr(tqc, "layout", None) is not None:
        folded._layout = tqc.layout
    return folded


@pytest.fixture(scope="module")
def device():
    """fake_marrakesh, the stand-in for the primary device."""
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    except Exception:  # pragma: no cover
        pytest.skip("FakeMarrakesh unavailable")
    return FakeMarrakesh()


@pytest.fixture(scope="module")
def subset():
    """The 10 frozen hardware-subset instances."""
    return json.loads(SUBSET_FILE.read_text(encoding="utf-8"))["selected"]


@pytest.fixture(scope="module")
def isa(device, subset):
    """P-n3-M1-17 (241 two-qubit gates) transpiled with its WP3 seed."""
    record = next(r for r in subset if r["name"] == "P-n3-M1-17")
    instance = SubsetSumInstance(items=record["items"], target=record["target"], name=record["name"])
    circuit = DGSSPSolver(DGConfig(iterations=record["r_opt"])).build_circuit(instance)
    cfg = ZNESamplingConfig(seed_min=record["best_seed"], seed_max=record["best_seed"] + 1)
    tqc, _ = transpile_once_seeded(circuit, device, cfg)
    return tqc


# ---------------------------------------------------------------------------
# Plan arithmetic (no circuits)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 76, 241, 1206])
def test_realised_scale_is_within_one_over_n(n):
    """|realised - nominal| <= 1/N for every nominal scale."""
    for k in range(0, 900):
        lam = 1.0 + k / 150.0
        assert abs(float(realised_scale(lam, n)) - lam) <= 1.0 / n + 1e-12


def test_odd_integers_need_no_partial_layer():
    """(k, m) = ((lam-1)/2, 0) for odd lam."""
    for lam in (1, 3, 5, 7, 9):
        assert fold_plan(lam, 241) == ((lam - 1) // 2, 0)
        assert realised_scale(lam, 241) == Fraction(lam)


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [1.0, 1.25, 1.5, 2.0, 2.37, 3.0, 4.2, 5.0])
def test_realised_scale_matches_the_two_qubit_count(isa, lam):
    """FoldResult.realised_scale is exactly N_2q(folded) / N_2q(input)."""
    result = fold_isa(isa, lam, seed=11)
    n2 = two_qubit_count(isa)
    assert result.realised_scale == Fraction(two_qubit_count(result.circuit), n2)
    assert abs(float(result.realised_scale) - lam) <= 1.0 / n2 + 1e-12
    assert result.realised_scale == realised_scale(lam, n2)


@pytest.mark.parametrize("lam", [1, 3, 5, 7])
def test_odd_integers_reproduce_the_previous_implementation(isa, lam):
    """Gate for gate, name included."""
    new = fold_transpiled(isa, lam, seed=123)
    old = _legacy_fold(isa, lam)
    assert new == old
    assert new.name == old.name
    assert [i.operation.name for i in new.data] == [i.operation.name for i in old.data]


def test_same_seed_same_circuit_different_seed_different_subset(isa):
    """Deterministic given (circuit, lam, seed)."""
    a = fold_transpiled(isa, 1.6, seed=5)
    b = fold_transpiled(isa, 1.6, seed=5)
    c = fold_transpiled(isa, 1.6, seed=6)
    assert a == b
    assert a != c
    assert two_qubit_count(a) == two_qubit_count(c)


def _two_qubit_per_pair(circuit):
    """Two-qubit gate count per (sorted) physical pair."""
    out: dict[tuple[int, ...], int] = {}
    for instruction in circuit.data:
        if instruction.operation.num_qubits == 2:
            pair = tuple(sorted(circuit.find_bit(q).index for q in instruction.qubits))
            out[pair] = out.get(pair, 0) + 1
    return out


def _sxdg_per_qubit(circuit):
    """Inverted-sx count per qubit: one per extra-folded sx below scale 3."""
    out: dict[int, int] = {}
    for instruction in circuit.data:
        if instruction.operation.name == "sxdg":
            q = circuit.find_bit(instruction.qubits[0]).index
            out[q] = out.get(q, 0) + 1
    return out


def test_partial_subsets_are_nested_across_scales(isa):
    """For one seed, a larger fractional scale folds a superset of the gates.

    Black-box consequence checked: the extra folds on every physical pair (and
    the extra sx folds on every qubit) never decrease from lam = 1.4 to 1.8.
    """
    small = fold_isa(isa, 1.4, seed=9)
    large = fold_isa(isa, 1.8, seed=9)
    assert large.n_two_q_extra > small.n_two_q_extra > 0

    small_pairs, large_pairs = _two_qubit_per_pair(small.circuit), _two_qubit_per_pair(large.circuit)
    assert all(large_pairs[p] >= small_pairs[p] for p in small_pairs)
    small_sx, large_sx = _sxdg_per_qubit(small.circuit), _sxdg_per_qubit(large.circuit)
    assert all(large_sx.get(q, 0) >= n for q, n in small_sx.items())

    # A different seed picks a different subset of the same size.
    other = fold_isa(isa, 1.4, seed=10)
    assert other.n_two_q_extra == small.n_two_q_extra
    assert _two_qubit_per_pair(other.circuit) != small_pairs


def test_single_qubit_noise_scales_alongside(isa):
    """The physical 1Q gates get the same extra-fold fraction; rz is never extra-folded."""
    result = fold_isa(isa, 2.0, seed=3)
    assert result.realised_scale_1q is not None
    assert abs(float(result.realised_scale_1q) - 2.0) <= 1.0 / result.n_one_q_physical + 1e-12


@pytest.mark.parametrize("lam", [1.37, 2.0, 3.0, 4.5])
def test_rebased_fold_is_isa_valid_without_retranspilation(device, isa, lam):
    """Every instruction is supported by the target on exactly its physical qubits."""
    folded = rebase_to_backend(fold_transpiled(isa, lam, seed=1), device)
    target = device.target
    for instruction in folded.data:
        name = instruction.operation.name
        if name == "barrier":
            continue
        qargs = tuple(folded.find_bit(q).index for q in instruction.qubits)
        assert target.instruction_supported(name, qargs), (name, qargs)
    assert folded.layout is not None
    assert transpiled_metrics(folded)["physical_qubits"] == transpiled_metrics(isa)["physical_qubits"]
    assert two_qubit_count(folded) == two_qubit_count(fold_transpiled(isa, lam, seed=1))


@pytest.mark.skipif(not mitiq_is_available(), reason="mitiq not installed")
def test_mitiq_folds_stay_on_the_layout_and_are_isa_valid(device, isa):
    """Mitiq's folds (through the Cirq wrapper) keep registers, layout and ISA validity."""
    from dgssp.mitigation.mitiq_baseline import mitiq_fold_isa

    folded = mitiq_fold_isa(isa, 2.0, seed=4)
    assert folded.num_qubits == isa.num_qubits
    assert folded.num_clbits == isa.num_clbits
    assert folded.layout is not None
    ratio = two_qubit_count(folded) / two_qubit_count(isa)
    assert 1.3 < ratio < 3.0

    rebased = rebase_to_backend(folded, device)
    for instruction in rebased.data:
        qargs = tuple(rebased.find_bit(q).index for q in instruction.qubits)
        assert device.target.instruction_supported(instruction.operation.name, qargs)

    measured = sorted(
        (isa.find_bit(i.qubits[0]).index, isa.find_bit(i.clbits[0]).index)
        for i in isa.data
        if i.operation.name == "measure"
    )
    measured_folded = sorted(
        (folded.find_bit(i.qubits[0]).index, folded.find_bit(i.clbits[0]).index)
        for i in folded.data
        if i.operation.name == "measure"
    )
    assert measured == measured_folded
    assert mitiq_fold_isa(isa, 2.0, seed=4) == folded


# ---------------------------------------------------------------------------
# Transpile-once reproduction of the WP3 circuits
# ---------------------------------------------------------------------------


def test_transpile_once_reproduces_every_hardware_subset_circuit(device, subset):
    """seed_min=best_seed, seed_max=best_seed+1 gives the recorded circuit."""
    assert len(subset) == 10
    for record in subset:
        instance = SubsetSumInstance(
            items=record["items"], target=record["target"], name=record["name"]
        )
        circuit = DGSSPSolver(DGConfig(iterations=record["r_opt"])).build_circuit(instance)
        cfg = ZNESamplingConfig(
            seed_min=record["best_seed"], seed_max=record["best_seed"] + 1, optimization_level=3
        )
        tqc, seed = transpile_once_seeded(circuit, device, cfg)
        sizes = transpiled_metrics(tqc)
        assert seed == record["best_seed"]
        assert sizes["two_q_count"] == record["two_q_count"], record["name"]
        assert sizes["two_q_depth"] == record["two_q_depth"], record["name"]
        assert sizes["physical_qubits"] == record["physical_qubits"], record["name"]
