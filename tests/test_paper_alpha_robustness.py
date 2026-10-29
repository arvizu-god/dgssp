"""
Tests for ``paper/scripts/alpha_robustness.py`` and
``paper/scripts/describe_layer_fidelity.py`` (offline parts only).

Pinned: the weighted regression and means, the residual grouping, the
monotonicity check, and that the layer fidelity frozen with the job of
record's execution calibration is read (no network).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("scipy")
pytest.importorskip("qiskit_ibm_runtime")
ar = pytest.importorskip("alpha_robustness")
dlf = pytest.importorskip("describe_layer_fidelity")


def test_wls_recovers_a_line_and_widens_for_scatter():
    x = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    y = 3.0 - 0.001 * x
    exact = ar.wls(x, y, np.full(5, 0.1))
    assert exact["slope"] == pytest.approx(-0.001)
    assert exact["intercept"] == pytest.approx(3.0)
    assert exact["covariance_scale"] == 1.0
    noisy = ar.wls(x, y + np.array([0.5, -0.5, 0.5, -0.5, 0.5]), np.full(5, 0.1))
    assert noisy["covariance_scale"] > 1.0
    assert noisy["slope_ci"][0] < noisy["slope"] < noisy["slope_ci"][1]
    assert ar.wls([1, 2], [1, 2], [1, 1]) is None


def test_weighted_mean_and_split():
    m = ar.weighted_mean([1.0, 3.0], [1.0, 1.0])
    assert m["mean"] == pytest.approx(2.0)
    assert m["se"] == pytest.approx(np.sqrt(2.0 / 2.0))  # chi2/dof = 2 widens sqrt(1/2)
    rows = [{"id": str(i), "two_q": q, "v": v, "s": 0.1}
            for i, (q, v) in enumerate([(71, 3.2), (112, 3.1), (475, 2.0), (749, 1.8)])]
    split = ar.split_test(rows, "v", "s")
    assert split["shallow"]["n"] == 2 and split["deep"]["n"] == 2
    assert split["difference"] == pytest.approx(1.9 - 3.15)
    assert split["post_hoc"] is True


def test_variant_groups():
    base = {"key": "P-n3-M1-17", "n_items": 3, "iterations": 2, "r_opt": 2}
    assert ar.variant_group(base) == "r = r_opt"
    assert ar.variant_group(base | {"key": "P-n3-M1-17_r1", "iterations": 1}) == "n = 3, r < r_opt"
    assert ar.variant_group(base | {"key": "P-n3-M1-17_r3", "iterations": 3}) == "n = 3, r > r_opt"
    assert ar.variant_group(base | {"key": "P-n3-M1-17_drift"}).startswith("drift")
    assert ar.base_key("P-n2-M1-11_drift") == "P-n2-M1-11"


def test_monotone_tolerates_shot_noise_only():
    falling = np.array([0.8, 0.7, 0.6, 0.5])
    assert ar.monotone(falling, 4096)
    assert ar.monotone(falling + np.array([0, 0.005, 0, 0]), 4096)
    assert not ar.monotone(np.array([0.3, 0.4, 0.5, 0.4]), 4096)


def test_layer_fidelity_of_the_run_day_is_in_the_frozen_calibration():
    path = REPO_ROOT / "paper/hardware/calibration_exec_2026-10-25.json"
    if not path.exists():
        pytest.skip("execution calibration not present")
    raw = json.loads(path.read_text(encoding="utf-8"))["raw_properties"]
    found = dlf.exposed(raw, {"clops_h": 1, "processor_type": {}})
    lf = found["layer_fidelity"]
    assert "100" in lf and len(lf["100"]["qubits"]) == 100
    assert 0 < lf["100"]["eplg"] < 0.05
    assert found["configuration"] == {}
    cmp = dlf.comparison(lf, 0.0026, {"alpha": 3.176, "ci_scaled": [3.07, 3.35]})
    assert cmp["chain_length"] == 100
    assert cmp["ratio_eplg_to_median_cz"] == pytest.approx(lf["100"]["eplg"] / 0.0026)
