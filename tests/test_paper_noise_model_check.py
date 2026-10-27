"""
Tests for ``paper/scripts/noise_model_check.py`` (the hardware-vs-model check).

The service and the simulations are not exercised here (``--predict`` was
accepted against the committed dry run: identical counts on all 47 PUBs).
What is pinned: the alpha fit recovers a known scale, flags a grid edge and
widens its interval when the scatter exceeds shot noise; the PUB selection
drops instances with no signal; calibration diffs and labels.
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
nmc = pytest.importorskip("noise_model_check")

ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
SHOTS = 4096


def _curves(alphas, n_pub=12, seed=1):
    """P(alpha) = S**alpha * P_ideal + (1 - S**alpha) * P_u for a spread of S."""
    rng = np.random.default_rng(seed)
    survival = rng.uniform(0.3, 0.95, n_pub)
    p_ideal, p_u = 0.95, 0.125
    a = np.asarray(alphas)[:, None]
    return survival**a * p_ideal + (1 - survival**a) * p_u


def test_fit_recovers_a_known_alpha():
    """Hardware drawn at alpha = 2.7: the fit lands there, inside its interval."""
    sim = _curves(ALPHAS)
    truth = _curves([2.7])[0]
    hw = np.random.default_rng(5).binomial(SHOTS, truth) / SHOTS
    fit = nmc.fit_alpha(ALPHAS, sim, 10**7, hw, SHOTS, B=100)
    assert fit["alpha"] == pytest.approx(2.7, abs=0.1)
    assert fit["ci"][0] <= 2.7 <= fit["ci"][1]
    assert fit["ci_scaled"][0] <= fit["ci"][0] and fit["ci_scaled"][1] >= fit["ci"][1]
    assert fit["at_grid_edge"] is False
    assert fit["n_pubs"] == 12 and fit["dof"] == 11


def test_excess_scatter_widens_the_quoted_interval():
    """Noise beyond shot noise -> chi2_red > 1 -> ci_scaled wider than ci."""
    sim = _curves(ALPHAS)
    truth = _curves([2.7])[0]
    hw = np.clip(truth + np.random.default_rng(2).normal(0, 0.03, truth.size), 0, 1)
    fit = nmc.fit_alpha(ALPHAS, sim, 10**7, hw, SHOTS, B=100)
    assert fit["chi2_red"] > 1
    width = fit["ci"][1] - fit["ci"][0]
    scaled = fit["ci_scaled"][1] - fit["ci_scaled"][0]
    assert scaled == pytest.approx(width * np.sqrt(fit["chi2_red"]), rel=1e-9)


def test_an_answer_beyond_the_grid_is_flagged():
    """Truth at alpha = 7 on a grid ending at 5: the edge flag says widen."""
    sim = _curves(ALPHAS)
    hw = _curves([7.0])[0]
    fit = nmc.fit_alpha(ALPHAS, sim, 10**7, hw, 10**7, B=20)
    assert fit["at_grid_edge"] is True


def test_leave_one_out_has_one_entry_per_record():
    sim = _curves(ALPHAS, n_pub=6)
    hw = _curves([3.0], n_pub=6)[0]
    groups = ["a", "a", "b", "b", "c", "c"]
    fit = nmc.fit_alpha(ALPHAS, sim, 10**7, hw, 10**7, groups=groups, B=10)
    assert set(fit["loo"]) == {"a", "b", "c"}
    assert all(v == pytest.approx(3.0, abs=0.05) for v in fit["loo"].values())


def test_a_bad_grid_is_refused():
    with pytest.raises(ValueError, match="increasing"):
        nmc.fit_alpha([2.0, 1.0], np.zeros((2, 3)), 100, np.zeros(3), 100)
    with pytest.raises(ValueError, match="sim must be"):
        nmc.fit_alpha([1.0, 2.0], np.zeros((3, 3)), 100, np.zeros(3), 100)


def test_signal_pubs_drop_instances_without_signal():
    """M = 0 and M = N/2 records never enter the fit; everything else does."""
    manifest = json.loads(
        (REPO_ROOT / "paper/hardware/manifest_2026-10-25.json").read_text(encoding="utf-8")
    )
    pubs = nmc.signal_pubs(manifest)
    keys = {p["key"] for p in pubs}
    assert not {"P-n2-M0-06", "P-n3-M0-06", "P-n4-M0-09", "P-n2-M2-03"} & keys
    assert {"P-n2-M1-11", "P-n4-M1-10_r4", "P-n3-M2-17_drift"} <= keys
    assert len(pubs) == 35
    assert all(p["p_ideal"] > p["p_uniform"] for p in pubs)


def test_payload_differences_and_labels():
    a = {"dt": 1.0, "instructions": {"cz": [{"qubits": [0, 1], "error": 0.01, "duration": 1}]}}
    b = json.loads(json.dumps(a))
    assert nmc.payload_differences(a, b) == []
    b["instructions"]["cz"][0]["error"] = 0.02
    b["dt"] = 2.0
    assert nmc.payload_differences(a, b) == ["dt", "cz[0, 1]"]
    date = "2026-10-25"
    assert nmc.calibration_label(Path(f"calibration_exec_{date}.json"), date) == "exec"
    assert nmc.calibration_label(Path(f"calibration_{date}.json"), date) == "plan"


def test_parse_utc_needs_a_zone():
    assert nmc.parse_utc("2026-10-26T06:05:15Z").isoformat() == "2026-10-26T06:05:15+00:00"
    assert nmc.parse_utc("2026-10-25T23:36:48-06:00").hour == 5
    with pytest.raises(SystemExit):
        nmc.parse_utc("2026-10-26T06:05:15")
