"""Tests for the multi-instance batch runner."""

from __future__ import annotations

import json

import pytest

from dgssp import (
    BatchConfig,
    BatchResult,
    DGConfig,
    SubsetSumInstance,
    run_batch,
    run_random_batch,
)
from dgssp.experiments import prepare_instances


@pytest.fixture
def small_batch():
    """Three random 3-item instances run on the ideal simulator."""
    return run_random_batch(
        3,
        3,
        max_value=8,
        seed=11,
        config=BatchConfig(shots=2048, seed_simulator=1234),
    )


def test_returns_one_result_per_instance(small_batch):
    """The batch preserves cardinality and order."""
    assert isinstance(small_batch, BatchResult)
    assert len(small_batch) == 3
    assert len(list(small_batch)) == 3


def test_every_result_carries_ground_truth(small_batch):
    """DP solutions and the measured probability against them are present."""
    for result in small_batch:
        assert result.num_solutions >= 1
        assert result.solution_bitstrings
        assert 0.0 <= result.solution_probability <= 1.0
        assert result.distribution
        assert result.iterations >= 1


def test_all_circuits_ran_in_a_single_job(monkeypatch):
    """The ideal/noisy path submits one job for the whole batch."""
    import dgssp.experiments as experiments

    calls = []
    original = experiments.sample_counts

    def spy(mode, circuits, **kwargs):
        calls.append(len(circuits) if isinstance(circuits, (list, tuple)) else 1)
        return original(mode, circuits, **kwargs)

    monkeypatch.setattr(experiments, "sample_counts", spy)

    run_random_batch(
        4, 3, seed=5, config=BatchConfig(shots=512, seed_simulator=1234)
    )
    assert calls == [4], f"expected one 4-circuit job, got {calls}"


def test_auto_iterations_use_the_dp_solution_count():
    """'auto' picks each instance's own optimal iteration count."""
    instances = [
        SubsetSumInstance(items=[1, 2, 3], target=5, name="one"),
        SubsetSumInstance(items=[1, 2, 3], target=3, name="two"),
    ]
    metas = prepare_instances(instances, BatchConfig(dg_config=DGConfig(iterations="auto")))
    assert all(m.iterations >= 1 for m in metas)
    assert metas[1].solution_bitstrings == ["011", "100"]


def test_explicit_iterations_override_auto():
    """An integer in DGConfig is applied to every instance."""
    instances = [SubsetSumInstance(items=[1, 2, 3], target=5)]
    metas = prepare_instances(instances, BatchConfig(dg_config=DGConfig(iterations=2)))
    assert metas[0].iterations == 2


def test_summary_aggregates_the_batch(small_batch):
    """summary() reports counts and averaged probabilities."""
    summary = small_batch.summary()
    assert summary["n_instances"] == 3
    assert summary["executor"] == "ideal"
    assert 0.0 <= summary["mean_solution_probability"] <= 1.0
    assert 0.0 <= summary["top_outcome_success_rate"] <= 1.0


def test_ideal_run_is_accurate(small_batch):
    """On a noiseless simulator the modal outcome should usually be correct."""
    assert small_batch.summary()["top_outcome_success_rate"] == pytest.approx(1.0)


def test_to_json_roundtrips(small_batch, tmp_path):
    """The serialised batch reloads with all its fields intact."""
    path = small_batch.to_json(tmp_path / "batch.json")
    payload = json.loads(path.read_text())

    assert len(payload["results"]) == 3
    assert payload["config"]["executor"] == "ideal"
    first = payload["results"][0]
    assert {"items", "target", "counts", "solution_probability"} <= set(first)


def test_to_dataframe_shape(small_batch):
    """The tidy table has one row per instance and no dict-valued columns."""
    pd = pytest.importorskip("pandas")
    frame = small_batch.to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert "solution_probability" in frame.columns
    assert "counts" not in frame.columns


def test_empty_batch_rejected():
    """Running zero instances is a mistake, not an empty result."""
    with pytest.raises(ValueError):
        run_batch([], BatchConfig())


def test_unknown_executor_rejected():
    """A typo'd executor name fails loudly."""
    instances = [SubsetSumInstance(items=[1, 2], target=3)]
    with pytest.raises(ValueError):
        run_batch(instances, BatchConfig(executor="magic"))  # type: ignore[arg-type]


def test_negative_items_are_supported_end_to_end():
    """A batch containing negative-item instances runs and succeeds."""
    instances = [
        SubsetSumInstance(items=[3, -2, 4], target=1, name="neg1"),
        SubsetSumInstance(items=[2, 3, -5, 6], target=1, name="neg2"),
    ]
    batch = run_batch(instances, BatchConfig(shots=4096, seed_simulator=1234))
    for result in batch:
        assert result.solution_probability > 0.5
