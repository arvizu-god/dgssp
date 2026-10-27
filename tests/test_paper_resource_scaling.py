"""
Tests for ``paper/scripts/resource_scaling.py``.

The script is the source of every number in the C1 resource table, and its
expensive part (144 circuits x 3 devices x 32 transpiler seeds) is not something
CI should run. What CI *can* cheaply protect is everything around it: that the
grid it sweeps is the grid the module docstring describes, that the item
families really are what they claim to be, that the closed-form asserts fire on
the smallest cells, and that the fit and the survival estimate behave.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

resource_scaling = pytest.importorskip("resource_scaling")

from dgssp import build_encoding  # noqa: E402
from dgssp.solvers.draper_grover import two_adic_valuation  # noqa: E402

# ---------------------------------------------------------------------------
# Item families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b", resource_scaling.B_VALUES)
@pytest.mark.parametrize("n", resource_scaling.N_VALUES)
def test_item_families_land_in_the_documented_window(n, b):
    """Both families produce ``n`` positive items whose total gives width ``b``."""
    low, high = resource_scaling.sum_window(b)
    for name, make in resource_scaling.FAMILIES.items():
        items = make(n, b)
        if items is None:
            continue
        assert len(items) == n, name
        assert all(a >= 1 for a in items), name
        assert low <= sum(items) <= high, name
        assert build_encoding(items, 0).n_sum == b, name


@pytest.mark.parametrize("b", resource_scaling.B_VALUES)
@pytest.mark.parametrize("n", resource_scaling.N_VALUES)
def test_dense_is_all_odd_and_sparse_is_all_powers_of_two(n, b):
    """The two families bracket the 2-adic valuation, which is their whole point."""
    dense = resource_scaling.dense_items(n, b)
    if dense is not None:
        assert all(a % 2 == 1 for a in dense)
        assert all(two_adic_valuation(a) == 0 for a in dense)

    sparse = resource_scaling.sparse_items(n, b)
    if sparse is not None:
        assert all(a & (a - 1) == 0 for a in sparse)


def test_families_agree_on_which_cells_exist():
    """A cell is feasible for both families or for neither."""
    for n in resource_scaling.N_VALUES:
        for b in resource_scaling.B_VALUES:
            dense = resource_scaling.dense_items(n, b) is not None
            sparse = resource_scaling.sparse_items(n, b) is not None
            assert dense == sparse, f"(n={n}, b={b})"


def test_infeasible_cells_are_the_arithmetically_impossible_ones():
    """``n`` positive items force ``S >= n``, hence ``b >= ceil(log2(n+1)) + 1``."""
    for n in resource_scaling.N_VALUES:
        b_min = math.ceil(math.log2(n + 1)) + 1
        for b in resource_scaling.B_VALUES:
            feasible = resource_scaling.dense_items(n, b) is not None
            assert feasible == (b >= b_min), f"(n={n}, b={b})"


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_grid_is_deterministic_and_covers_every_feasible_cell():
    """Two builds give the same grid, and every feasible cell appears 4 times."""
    first = resource_scaling.build_grid()
    assert first == resource_scaling.build_grid()

    cells = {(point.n, point.b) for point in first}
    for n, b in cells:
        matching = [p for p in first if (p.n, p.b) == (n, b)]
        assert len(matching) == 4  # two families x {positive, negated}
        assert {(p.family, p.sign) for p in matching} == {
            (family, sign)
            for family in resource_scaling.FAMILIES
            for sign in ("positive", "negated")
        }

    assert len(first) == 4 * len(cells)
    assert all(point.name.startswith(f"n{point.n}_b{point.b}_") for point in first)


def test_grid_targets_are_reachable_subset_sums():
    """Every target is a subset sum, so the oracle always marks at least one state."""
    from dgssp import DPSSPSolver

    for point in resource_scaling.build_grid():
        if point.n > 5:  # the DP enumerates all subsets; keep CI quick
            continue
        result = DPSSPSolver().solve(point.instance())
        assert result.all_solutions, point.name


def test_negating_the_last_item_preserves_the_register_width():
    """The signed variant of a cell has the same ``(n, b)`` as its positive twin."""
    grid = resource_scaling.build_grid()
    by_key = {(p.n, p.b, p.family, p.sign): p for p in grid}
    for (n, b, family, sign), point in by_key.items():
        if sign != "negated":
            continue
        twin = by_key[(n, b, family, "positive")]
        assert point.items[:-1] == twin.items[:-1]
        assert point.items[-1] == -twin.items[-1]
        assert build_encoding(list(point.items), 0).n_sum == b


# ---------------------------------------------------------------------------
# Step 2, on the cheap end of the grid
# ---------------------------------------------------------------------------


def test_closed_forms_hold_on_the_small_cells():
    """``measure_logical`` asserts internally; this exercises those asserts."""
    small = [p for p in resource_scaling.build_grid() if p.n <= 4 and p.b <= 5]
    assert small, "the grid lost its small cells"
    for point in small:
        measured = resource_scaling.measure_logical(point)
        model = measured["resource_model"]
        assert model["n_ind"] == point.n
        assert model["n_sum"] == point.b
        assert measured["logical"]["two_q_count"] > 0
        assert measured["guard_bit_overhead"]["qubits"] == 1
        assert measured["signed_overhead"]["two_q_gates"] == 0
        expected_phases = 2 * point.b if point.sign == "negated" else 0
        assert measured["signed_overhead"]["phase_gates"] == expected_phases
        assert measured["r_opt"] >= 1


def test_borrowed_ancillas_never_cost_more_than_the_reference():
    """Qiskit's default synthesis is a lower bound on the ancilla-free count."""
    for point in resource_scaling.build_grid():
        if point.n > 4 or point.b > 5:
            continue
        measured = resource_scaling.measure_logical(point)["logical"]
        assert measured["two_q_count_hls"] <= measured["two_q_count"]


# ---------------------------------------------------------------------------
# Fits and survival
# ---------------------------------------------------------------------------


def test_fit_power_law_recovers_a_known_exponent():
    """A clean power law is recovered to within the reported standard error."""
    xs = [2.0, 3.0, 4.0, 5.0, 6.0]
    fit = resource_scaling.fit_power_law(xs, [7.0 * x**2.5 for x in xs])
    assert fit is not None
    assert fit["exponent"] == pytest.approx(2.5, abs=1e-6)
    assert fit["prefactor"] == pytest.approx(7.0, rel=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert fit["n_points"] == 5


def test_fit_power_law_refuses_an_underdetermined_fit():
    """Two points fit any power law exactly and leave no residual to judge."""
    assert resource_scaling.fit_power_law([2.0, 4.0], [1.0, 4.0]) is None
    assert resource_scaling.fit_power_law([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None


def test_survival_estimate_is_a_product_of_per_gate_survivals():
    """``S = (1-eps2)**n2 * (1-eps1)**n1``, with ``None`` where a median is missing."""
    errors = {"two_qubit": 0.01, "one_qubit": 0.001}
    estimate = resource_scaling.survival_estimate(100, 50, errors)
    assert estimate["survival_two_q"] == pytest.approx(0.99**100)
    assert estimate["survival_one_q"] == pytest.approx(0.999**50)
    assert estimate["survival"] == pytest.approx(0.99**100 * 0.999**50)

    partial = resource_scaling.survival_estimate(10, 10, {"two_qubit": 0.01})
    assert partial["survival_one_q"] is None
    assert partial["survival"] == pytest.approx(0.99**10)

    empty = resource_scaling.survival_estimate(10, 10, {})
    assert empty["survival"] is None


def test_rows_from_records_drops_the_untranspiled_rows():
    """A logical-only record carries no ISA metrics and must not reach the figure."""
    logical_block = {
        "two_q_count": 100,
        "one_q_count": 200,
        "depth": 10,
        "two_q_depth": 5,
        "two_q_count_hls": 90,
    }
    grid_block = {"n": 3, "b": 4, "family": "dense", "sign": "positive"}
    records = [
        {"backend": None, "transpiled": {}, "metrics": {"grid": grid_block, "logical": logical_block}},
        {
            "backend": "fake_marrakesh",
            "transpiled": {
                "two_q_count": 300,
                "two_q_depth": 120,
                "depth": 400,
                "one_q_count": 500,
                "survival": 0.5,
            },
            "metrics": {"grid": grid_block, "logical": logical_block},
        },
    ]
    rows = resource_scaling.rows_from_records(records)
    assert len(rows) == 1
    assert rows[0]["backend"] == "fake_marrakesh"
    assert rows[0]["two_q_count"] == 300
    assert rows[0]["logical_two_q"] == 100


# ---------------------------------------------------------------------------
# Table "closed forms vs measured" and the figure, from the run of record
# ---------------------------------------------------------------------------

RESOURCES = REPO_ROOT / "paper" / "data" / "resources.json"
needs_resources = pytest.mark.skipif(not RESOURCES.exists(), reason="resources.json not checked out")


@pytest.fixture(scope="module")
def stored_records():
    """The 432 records of the run of record."""
    import json

    return json.loads(RESOURCES.read_text(encoding="utf-8"))


def test_the_fake_twin_of_a_device():
    """The job ran on ibm_fez; the grid was transpiled on fake_fez."""
    assert resource_scaling.fake_twin("ibm_fez") == "fake_fez"
    assert resource_scaling.fake_twin(None) is None


@needs_resources
def test_every_closed_form_matches_every_stored_circuit(stored_records):
    """The table's "144/144" is re-derived from the records, not typed."""
    checks = resource_scaling.closed_form_checks(stored_records)
    assert checks.pop("points") == 144
    assert set(checks) == {key for key, _, _ in resource_scaling.CLOSED_FORMS}
    assert all(count == 144 for count in checks.values()), checks


@needs_resources
def test_a_wrong_count_is_caught(stored_records):
    """One circuit with an extra controlled phase is one disagreement."""
    import copy

    broken = copy.deepcopy(stored_records[:3])
    broken[0]["metrics"]["gate_level_counts"]["cp"] += 1
    checks = resource_scaling.closed_form_checks(broken)
    assert checks["points"] == 1
    assert checks["cp"] == 0 and checks["h"] == 1


@needs_resources
def test_the_table_quotes_the_measured_factors(stored_records):
    """Routing 1.64x and borrowed ancillas 0.78x (median), as in the note SS10."""
    text = resource_scaling.resources_table(stored_records)
    assert "& 144/144 \\\\" in text
    assert "1.25--2.08 (median 1.64)" in text
    assert "0.63--1.00 (median 0.78)" in text
    assert "$-15$--0 (median 0)" in text
    # The hardware cell (2, 3): reference 55, transpiled 82 on fez.
    assert "2 & 3 & 55 & 55 & 82 & 83 & 83 \\\\" in text


@needs_resources
def test_the_figure_draws_with_and_without_the_hardware_band(stored_records, tmp_path):
    """No summary: no band. With one: the band and the hardware points are drawn."""
    rows = resource_scaling.rows_from_records(stored_records)
    written = resource_scaling.make_figure(rows, directory=tmp_path / "plain")
    assert all(path.exists() for path in written)

    summary = {
        "config": {"backend": "ibm_fez"},
        "limit_band": {"lower": 71, "lower_key": "a", "upper": 194, "upper_key": "b",
                       "exceptions": []},
        "records": [
            {"n_items": 2, "two_q_count": 71, "zne_status": "resolved"},
            {"n_items": 3, "two_q_count": 194, "zne_status": "not identifiable"},
            {"n_items": 2, "two_q_count": 80, "zne_status": "no signal"},
        ],
    }
    written = resource_scaling.make_figure(rows, hardware=summary, directory=tmp_path / "band")
    assert all(path.exists() for path in written)
    assert resource_scaling.load_hardware_summary(tmp_path / "absent.json") is None
