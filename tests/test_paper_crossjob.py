"""
Tests for ``paper/scripts/hw_crossjob.py`` (its pure parts; the build reads the
real job files and is exercised by running the script).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("scipy")
pytest.importorskip("qiskit_ibm_runtime")
crossjob = pytest.importorskip("hw_crossjob")


def test_base_key_strips_layout_and_drift():
    assert crossjob.base_key("P-n3-M1-17_r1_L2") == "P-n3-M1-17_r1"
    assert crossjob.base_key("P-n2-M1-11_L1_drift") == "P-n2-M1-11"
    assert crossjob.base_key("P-n3-M2-17_drift") == "P-n3-M2-17"


def test_rms_block_leaves_only_the_layout_test_out():
    rows = [{"key": "P-n3-M2-17_L2", "deviation": 0.4},
            {"key": "P-n2-M1-11_L2", "deviation": 0.03},
            {"key": "P-n3-M1-17_L2", "deviation": -0.04}]
    block = crossjob.rms_block(rows)
    assert block["rms_without_layout_test"] == pytest.approx(((0.03**2 + 0.04**2) / 2) ** 0.5)
    assert block["rms"] > block["rms_without_layout_test"]
    assert block["max_abs"] == pytest.approx(0.4)


def test_separators_need_every_below_and_no_above():
    below = [{90, 91, 92}, {90, 91, 92, 98}, {91, 92, 93}]
    above = [{92, 110}, {98, 111}]
    assert crossjob.separators(below, above) == [91]
    assert crossjob.separators([], above) == []


def test_shifted_chi2_removes_one_shift_per_reference_day():
    rows = [{"day": "a", "difference": -0.5, "se_ref": 0.1, "se_day3": 0.0},
            {"day": "a", "difference": -0.5, "se_ref": 0.1, "se_day3": 0.0},
            {"day": "a", "difference": -0.5, "se_ref": 0.1, "se_day3": 0.0},
            {"day": "b", "difference": 0.2, "se_ref": 0.1, "se_day3": 0.0}]
    out = crossjob.shifted_chi2(rows)
    assert out["dof"] == 2
    assert out["chi2"] == pytest.approx(0.0)
    assert out["shifts"] == pytest.approx({"a": -0.5, "b": 0.2})
    rows[0]["difference"] = -0.2
    rows[1]["difference"] = -0.8
    assert crossjob.shifted_chi2(rows)["chi2_red"] == pytest.approx(9.0)


def test_between_chi2_is_zero_for_equal_values_and_needs_two_rows():
    same = [{"alpha_day3": 2.4, "se_day3": 0.1}, {"alpha_day3": 2.4, "se_day3": 0.2}]
    assert crossjob.between_chi2(same) == pytest.approx(0.0)
    assert crossjob.between_chi2(same[:1]) is None


def test_qpu_block_totals_every_job():
    usage = {"a": {"usage_seconds": 7}, "b": {"usage_seconds": 75}}
    assert crossjob.qpu_block(usage) == {"jobs": {"a": 7, "b": 75}, "total_seconds": 82}
