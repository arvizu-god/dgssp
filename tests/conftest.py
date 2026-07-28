"""Shared pytest fixtures and helpers for the dgssp test suite."""

from __future__ import annotations

import pytest

from dgssp import SubsetSumInstance


@pytest.fixture
def positive_instance() -> SubsetSumInstance:
    """A small all-positive instance with exactly one exact solution."""
    return SubsetSumInstance(items=[1, 2, 3], target=5, name="pos")


@pytest.fixture
def negative_instance() -> SubsetSumInstance:
    """A small instance containing a negative item."""
    return SubsetSumInstance(items=[3, -2, 4], target=1, name="neg")


@pytest.fixture
def ideal_backend():
    """A seeded noiseless AerSimulator."""
    from dgssp import build_ideal_aer_backend

    return build_ideal_aer_backend(seed_simulator=1234)


def fake_noisy_backend():
    """
    Build a noisy AerSimulator from a bundled fake IBM device.

    Returns ``None`` when ``qiskit_ibm_runtime.fake_provider`` is unavailable,
    so calibration-dependent tests can skip rather than fail.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2

        return AerSimulator.from_backend(FakeManilaV2()), FakeManilaV2()
    except Exception:  # pragma: no cover - optional provider data
        return None
