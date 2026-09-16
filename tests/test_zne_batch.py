"""
Tests for the single-job ZNE path and the iteration sweep.

The hardware budget allows one job a month, so "one job" is a tested property,
not a convention: the sampler entry point is monkeypatched with a counter and
every path must call it exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dgssp import SubsetSumInstance
from dgssp.metrics import proportion_sigma
from dgssp.mitigation import mitiq_is_available
from dgssp.mitigation.zne import (
    ZNEItem,
    ZNESamplingConfig,
    analyse_zne_item,
    run_zne_batch,
    transpile_once_seeded,
)
from dgssp.solvers import DGConfig, DGSSPSolver, grover_success_probability
from dgssp.transpilation import two_qubit_count

REPO = Path(__file__).resolve().parents[1]
SUBSET_FILE = REPO / "paper" / "instances" / "hardware_subset.json"
SWEEP_FILE = REPO / "paper" / "instances" / "sweep.json"

SMALL = ("P-n2-M1-11", "P-n2-M0-06", "P-n2-M2-03")


@pytest.fixture(scope="module")
def device_pair():
    """fake_marrakesh and the Aer noise model derived from it."""
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    except Exception:  # pragma: no cover
        pytest.skip("FakeMarrakesh unavailable")
    device = FakeMarrakesh()
    return device, AerSimulator.from_backend(device)


@pytest.fixture(scope="module")
def small_items(device_pair):
    """Three n = 2 hardware-subset instances, transpiled with their WP3 seeds."""
    device, _ = device_pair
    records = {
        r["name"]: r
        for r in json.loads(SUBSET_FILE.read_text(encoding="utf-8"))["selected"]
    }
    out = []
    for name in SMALL:
        record = records[name]
        instance = SubsetSumInstance(items=record["items"], target=record["target"], name=name)
        circuit = DGSSPSolver(DGConfig(iterations=record["r_opt"])).build_circuit(instance)
        cfg = ZNESamplingConfig(seed_min=record["best_seed"], seed_max=record["best_seed"] + 1)
        tqc, _ = transpile_once_seeded(circuit, device, cfg)
        out.append((record, tqc))
    return out


def test_three_instances_three_scales_plus_mitiq_is_one_job(monkeypatch, device_pair, small_items):
    """One sample_counts call; every pub maps back to its instance and scale."""
    _, noisy = device_pair
    calls: list[list] = []

    def fake_sample_counts(mode, circuits, **kwargs):
        calls.append(list(circuits))
        # Pub i returns a counts dict that identifies it: {"00": i + 1}.
        return [{"00": i + 1} for i in range(len(circuits))]

    # Patch the globals of the function actually called: another test module
    # re-imports dgssp, so "the" dgssp.mitigation.zne in sys.modules need not
    # be the module run_zne_batch was defined in.
    monkeypatch.setitem(run_zne_batch.__globals__, "sample_counts", fake_sample_counts)

    include_mitiq = mitiq_is_available()
    cfg = ZNESamplingConfig(scales=[1, 1.5, 3], shots_per_scale=100, fold_seed=2)
    items = [
        ZNEItem(name=r["name"], circuit=tqc, solution_bitstrings=r["solution_bitstrings"])
        for r, tqc in small_items
    ]
    batch = run_zne_batch(items, noisy, cfg, include_mitiq=include_mitiq)

    assert len(calls) == 1, f"expected one job, got {len(calls)}"
    submitted = calls[0]
    per_item = 3 + (2 if include_mitiq else 0)  # Mitiq's scale 1 reuses the item's own
    assert len(submitted) == batch.n_circuits == 3 * per_item

    for (record, tqc), run in zip(small_items, batch.items, strict=True):
        assert run.name == record["name"]
        n2 = two_qubit_count(tqc)
        for j, lam in enumerate(run.realised_scales):
            index = run.pub_indices["dgssp"][j]
            assert run.counts_per_scale[lam] == {"00": index + 1}
            assert two_qubit_count(submitted[index]) == round(lam * n2)
        assert run.realised_scales[0] == 1.0
        assert abs(run.realised_scales[1] - 1.5) <= 1 / n2
        if include_mitiq:
            assert run.mitiq_scales == run.realised_scales
            assert run.mitiq_counts_per_scale[1.0] is run.counts_per_scale[1.0]
            for lam, index in zip(run.mitiq_scales[1:], run.pub_indices["mitiq"], strict=True):
                assert run.mitiq_counts_per_scale[lam] == {"00": index + 1}
    all_indices = sorted(i for run in batch.items for v in run.pub_indices.values() for i in v)
    assert all_indices == list(range(len(submitted)))


def test_batch_on_the_noise_model_and_offline_analysis(device_pair, small_items):
    """End to end on Aer: counts come back per scale and every estimator runs offline."""
    _, noisy = device_pair
    record, tqc = small_items[0]  # P-n2-M1-11, S ~ 0.75
    cfg = ZNESamplingConfig(
        scales=[1, 1.5, 2],
        shots_per_scale=2000,
        method="exponential",
        asymptote="p_uniform",
        seed_simulator=1234,
        n_bootstrap=200,
    )
    batch = run_zne_batch(
        [ZNEItem(name=record["name"], circuit=tqc)],
        noisy,
        cfg,
        include_mitiq=mitiq_is_available(),
    )
    run = batch.items[0]
    assert set(run.counts_per_scale) == set(run.realised_scales)
    assert all(sum(c.values()) == 2000 for c in run.counts_per_scale.values())

    from dgssp.execution import exact_index_distribution

    instance = SubsetSumInstance(items=record["items"], target=record["target"])
    reference = exact_index_distribution(instance, record["r_opt"])
    analysis = analyse_zne_item(
        run,
        cfg,
        solution_bitstrings=record["solution_bitstrings"],
        p_uniform=record["p_uniform"],
        reference=reference,
    )
    scalar = analysis["scalar"]
    assert scalar["scales"] == sorted(run.realised_scales)
    assert scalar["asymptote"] == record["p_uniform"]
    assert scalar["ci"][0] <= scalar["value"] <= scalar["ci"][1]
    assert scalar["resolved"] is True
    # Mitigation moves P_sol up, toward the ideal 1.0.
    assert scalar["value"] > scalar["value_at_min_scale"]
    assert analysis["pathology"]["tv_raw_min_scale"] > 0
    if mitiq_is_available():
        assert analysis["mitiq"]["scales"] == sorted(run.mitiq_scales)


def test_exact_reference_is_uniform_for_the_pathology_cases():
    """M = 0 and M/N = 1/2 at one iteration: the noiseless output is exactly uniform."""
    from dgssp.execution import exact_index_distribution

    records = {
        r["name"]: r
        for r in json.loads(SUBSET_FILE.read_text(encoding="utf-8"))["selected"]
    }
    for name in ("P-n2-M0-06", "P-n2-M2-03", "P-n3-M0-06"):
        record = records[name]
        instance = SubsetSumInstance(items=record["items"], target=record["target"])
        dist = exact_index_distribution(instance, record["r_opt"])
        k = 2 ** record["n_items"]
        assert len(dist) == k
        assert all(v == pytest.approx(1 / k, abs=1e-9) for v in dist.values()), name


def test_optimized_run_batch_is_one_job(monkeypatch):
    """run_batch(executor='optimized') on two instances, Mitiq included: one job."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    from dgssp import BatchConfig, run_batch
    from dgssp.experiments import run_zne_batch as batch_fn

    calls = []
    original = batch_fn.__globals__["sample_counts"]

    def spy(mode, circuits, **kwargs):
        calls.append(len(circuits))
        return original(mode, circuits, **kwargs)

    monkeypatch.setitem(batch_fn.__globals__, "sample_counts", spy)

    batch = run_batch(
        [
            SubsetSumInstance(items=[1, 2], target=3, name="a"),
            SubsetSumInstance(items=[1, 1], target=1, name="b"),
        ],
        BatchConfig(
            executor="optimized",
            backend=AerSimulator.from_backend(FakeManilaV2()),
            run_mitiq_baseline=mitiq_is_available(),
            zne_config=ZNESamplingConfig(
                scales=[1, 2, 3],
                shots_per_scale=500,
                seed_min=0,
                seed_max=2,
                seed_simulator=1234,
                n_bootstrap=100,
            ),
            bootstrap_samples=100,
        ),
    )
    assert len(calls) == 1
    for result in batch:
        assert result.zne_fit is not None and result.zne_fit["dof"] == 1
        assert result.pathology is not None
        assert result.best_seed in (0, 1)
        assert result.survival is not None
        record = result.to_record(executor="optimized", shots=500)
        assert record.schema_version == 2
        # Keys are the *realised* scales: odd integers exactly, 2 within 1/N_2q.
        keys = sorted(record.counts, key=float)
        assert keys[0] == "1" and keys[2] == "3"
        assert abs(float(keys[1]) - 2.0) <= 1.0 / result.two_qubit_count
        assert [float(k) for k in keys] == record.zne_fit["scales"]
        assert record.zne_fit["resolved"] in (True, False)
        if mitiq_is_available():
            assert result.mitiq_solution_probability is not None


def test_iteration_sweep_on_the_ideal_simulator_is_one_job(monkeypatch):
    """sin^2((2r+1) theta) within 4 sigma at every r, all r in one sample_counts call."""
    import dgssp.experiments as experiments

    calls = []
    original = experiments.run_iteration_sweep.__globals__["sample_counts"]

    def spy(mode, circuits, **kwargs):
        calls.append(len(circuits))
        return original(mode, circuits, **kwargs)

    monkeypatch.setitem(experiments.run_iteration_sweep.__globals__, "sample_counts", spy)

    spec = json.loads(SWEEP_FILE.read_text(encoding="utf-8"))["instances"][0]
    instance = SubsetSumInstance(items=spec["items"], target=spec["target"], name=spec["name"])
    shots = 8000
    out = experiments.run_iteration_sweep(
        instance,
        spec["iterations"],
        executor="ideal",
        shots=shots,
        seed_simulator=20261020,
        bootstrap_samples=200,
    )
    assert calls == [len(spec["iterations"])]
    assert out["M"] == spec["M"]
    for row in out["rows"]:
        analytic = grover_success_probability(spec["n_items"], spec["M"], row["iterations"])
        assert row["analytic"] == pytest.approx(analytic)
        sigma = proportion_sigma(analytic, shots)
        assert abs(row["p_sol"] - analytic) <= 4 * sigma, row
        assert row["p_sol_ci"][0] <= row["p_sol"] <= row["p_sol_ci"][1]
