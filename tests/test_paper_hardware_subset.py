"""
Tests for ``paper/scripts/select_hardware_subset.py`` and ``build_sweep.py``.

The expensive part -- 463 circuits x 32 SABRE seeds at optimization level 3,
plus a 20 000-shot noiseless run each -- is not something CI should redo.  What
CI can protect is the part that decides the paper's hardware plan: that the
ranking is a total order with no dependence on input order, that the edge
criterion is the mixture criterion and not the wrong one it is easy to slip
into, and that the committed subset file is internally consistent with the
family it claims to come from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"
INSTANCES_DIR = REPO_ROOT / "paper" / "instances"
SUBSET_FILE = INSTANCES_DIR / "hardware_subset.json"
SWEEP_FILE = INSTANCES_DIR / "sweep.json"
FAMILY_FILE = INSTANCES_DIR / "family.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

build_sweep = pytest.importorskip("build_sweep")
select_hardware_subset = pytest.importorskip("select_hardware_subset")

from dgssp.encoding import build_encoding  # noqa: E402
from dgssp.solvers.draper_grover import optimal_iterations  # noqa: E402


def row(
    name: str,
    *,
    n_items: int = 3,
    n_sum: int = 4,
    m_class: str = "1",
    M: int = 1,
    two_q_depth: int = 100,
    two_q_count: int = 150,
    survival: float = 0.5,
    p_ideal: float = 0.9,
    family: str = "P",
) -> dict[str, Any]:
    """Build a synthetic candidate row with the fields the selectors read."""
    p_uniform = M / 2**n_items
    p_est = survival * p_ideal + (1.0 - survival) * p_uniform
    return {
        "name": name,
        "family": family,
        "n_items": n_items,
        "n_sum": n_sum,
        "m_class": m_class,
        "M": M,
        "r_opt": optimal_iterations(n_items, M),
        "two_q_depth": two_q_depth,
        "two_q_count": two_q_count,
        "survival": survival,
        "p_ideal": p_ideal,
        "p_uniform": p_uniform,
        "p_est": p_est,
        "beats_random_by_2x": (
            None
            if M == 0
            else bool(p_est >= select_hardware_subset.EDGE_FACTOR * p_uniform)
        ),
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_rank_key_is_a_total_order_independent_of_input_order() -> None:
    """Equal depth and count are still ordered, by name."""
    rows = [
        row("b", two_q_depth=10, two_q_count=20),
        row("a", two_q_depth=10, two_q_count=20),
        row("c", two_q_depth=10, two_q_count=19),
    ]
    ranked = [r["name"] for r in sorted(rows, key=select_hardware_subset.rank_key)]
    assert ranked == ["c", "a", "b"]
    assert (
        [r["name"] for r in sorted(reversed(rows), key=select_hardware_subset.rank_key)]
        == ranked
    )


def test_select_cells_takes_the_shallowest_per_cell() -> None:
    """Depth wins, not count and not family."""
    rows = [
        row("deep-P", n_items=3, m_class="1", two_q_depth=300, two_q_count=100),
        row("shallow-S", n_items=3, m_class="1", two_q_depth=200, two_q_count=900,
            family="S"),
        row("other-cell", n_items=4, m_class="2", M=2, two_q_depth=50),
    ]
    picked = {r["name"] for r in select_hardware_subset.select_cells(rows)}
    assert picked == {"shallow-S", "other-cell"}


def test_select_cells_skips_empty_cells() -> None:
    """A cell with no candidates contributes nothing, and does not raise."""
    assert select_hardware_subset.select_cells([]) == []


def test_select_cells_covers_the_grid_when_it_is_populated() -> None:
    """Nine cells in, nine winners out."""
    rows = [
        row(f"x-{n}-{m}", n_items=n, m_class=m, M=0 if m == "0" else int(m))
        for n in select_hardware_subset.SUBSET_N_VALUES
        for m in select_hardware_subset.SUBSET_M_CLASSES
    ]
    assert len(select_hardware_subset.select_cells(rows)) == 9


# ---------------------------------------------------------------------------
# The edge criterion
# ---------------------------------------------------------------------------


def test_edge_uses_the_mixture_not_a_bare_survival_comparison() -> None:
    """
    The wrong test would pick the wide instance; the right one rejects it.

    At ``n = 6, M = 1`` the two criteria disagree on
    ``S in (P_uniform, P_uniform / (P_ideal - P_uniform))``, i.e.
    ``(0.015625, 0.016723)``.  ``wide`` sits inside that window: a bare
    ``S >= P_uniform`` test passes it -- and, because that threshold falls
    exponentially in ``n`` while ``S`` falls only as fast as the gate count
    grows, such a test keeps passing ever wider instances and degenerates into
    "return the biggest".  The mixture criterion rejects it.
    """
    wide = row("wide", n_items=6, M=1, survival=0.016, p_ideal=0.95)
    narrow = row("narrow", n_items=3, M=1, survival=0.6, p_ideal=0.95)

    assert wide["survival"] > wide["p_uniform"]  # the wrong test would pass it
    assert wide["beats_random_by_2x"] is False  # the right test does not
    assert narrow["beats_random_by_2x"] is True

    edge = select_hardware_subset.select_edge([wide, narrow])
    assert edge is not None and edge["name"] == "narrow"


def test_edge_takes_the_largest_passing_instance() -> None:
    """Among passers, width wins, then register width, then cost."""
    rows = [
        row("small", n_items=2, M=1, survival=0.9, p_ideal=0.99),
        row("big", n_items=4, M=1, survival=0.9, p_ideal=0.99),
        row("big-wider-b", n_items=4, n_sum=5, M=1, survival=0.9, p_ideal=0.99),
    ]
    edge = select_hardware_subset.select_edge(rows)
    assert edge is not None and edge["name"] == "big-wider-b"


def test_edge_excludes_infeasible_instances() -> None:
    """M=0 makes the criterion vacuous, so it must not be edge-eligible."""
    infeasible = row("infeasible", n_items=6, M=0, survival=1e-9, p_ideal=0.0)
    assert infeasible["p_uniform"] == 0.0
    assert infeasible["p_est"] == 0.0
    assert infeasible["p_est"] >= 2 * infeasible["p_uniform"]  # vacuously
    assert infeasible["beats_random_by_2x"] is None
    assert select_hardware_subset.select_edge([infeasible]) is None


def test_edge_excludes_single_iteration_instances() -> None:
    """
    A wide one-iteration instance cannot be the edge, however well it scores.

    ``n = 6, M = 11`` has ``r_opt = 1``: it beats guessing because one
    oracle-plus-diffuser is cheap, not because search survives at that width.
    """
    one_shot = row("one-shot", n_items=6, M=11, m_class="3+", survival=0.9, p_ideal=0.95)
    search = row("search", n_items=3, M=1, survival=0.6, p_ideal=0.95)
    assert one_shot["r_opt"] == 1 and one_shot["beats_random_by_2x"] is True
    assert search["r_opt"] >= select_hardware_subset.EDGE_MIN_R_OPT

    edge = select_hardware_subset.select_edge([one_shot, search])
    assert edge is not None and edge["name"] == "search"
    assert select_hardware_subset.select_edge([one_shot]) is None


def test_edge_returns_none_when_nothing_passes() -> None:
    """A family too noisy to beat guessing is a reportable result, not a crash."""
    assert (
        select_hardware_subset.select_edge(
            [row("hopeless", n_items=5, M=1, survival=1e-6, p_ideal=0.95)]
        )
        is None
    )


# ---------------------------------------------------------------------------
# The committed subset file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def subset() -> dict[str, Any]:
    """The committed hardware subset, or a skip."""
    if not SUBSET_FILE.exists():
        pytest.skip(
            f"{SUBSET_FILE} not generated; run "
            "`python paper/scripts/select_hardware_subset.py`."
        )
    return json.loads(SUBSET_FILE.read_text(encoding="utf-8"))


def test_subset_matches_the_committed_family(subset: dict[str, Any]) -> None:
    """The subset was built from the family file that is checked in."""
    if not FAMILY_FILE.exists():
        pytest.skip("family file not generated")
    family = json.loads(FAMILY_FILE.read_text(encoding="utf-8"))
    assert subset["family_payload_sha256"] == family["payload_sha256"]
    assert len(subset["candidates"]) == len(family["instances"])


def test_subset_size_is_in_the_planned_range(subset: dict[str, Any]) -> None:
    """Eight to twelve instances: what one monthly job can carry."""
    assert 8 <= len(subset["selected"]) <= 12


def test_subset_covers_the_grid(subset: dict[str, Any]) -> None:
    """Every populated (n, M-class) cell of the grid is represented."""
    covered = {
        (r["n_items"], r["m_class"])
        for r in subset["selected"]
        if r["n_items"] in select_hardware_subset.SUBSET_N_VALUES
        and r["m_class"] in select_hardware_subset.SUBSET_M_CLASSES
    }
    expected = {
        (n, m)
        for n in select_hardware_subset.SUBSET_N_VALUES
        for m in select_hardware_subset.SUBSET_M_CLASSES
    }
    assert covered == expected


def test_each_cell_winner_really_is_the_shallowest(subset: dict[str, Any]) -> None:
    """Re-derive the selection from the recorded candidates."""
    rederived = {r["name"] for r in select_hardware_subset.select_cells(subset["candidates"])}
    named = {
        r["name"]
        for r in subset["selected"]
        if r["selected_for"].startswith("cell ")
    }
    assert named == rederived


def test_edge_is_reproducible_from_the_recorded_candidates(
    subset: dict[str, Any],
) -> None:
    """The stored edge is what the criterion picks from the stored table."""
    edge = select_hardware_subset.select_edge(subset["candidates"])
    assert (edge["name"] if edge else None) == subset["edge_instance"]


def test_p_est_is_the_recorded_mixture(subset: dict[str, Any]) -> None:
    """``p_est`` is reconstructible from ``survival``, ``p_ideal``, ``p_uniform``."""
    for candidate in subset["candidates"]:
        s, p_i, p_u = candidate["survival"], candidate["p_ideal"], candidate["p_uniform"]
        assert candidate["p_est"] == pytest.approx(s * p_i + (1.0 - s) * p_u)


def test_p_uniform_is_M_over_2_to_the_n(subset: dict[str, Any]) -> None:
    """No hidden normalisation: the null model is a uniform index register."""
    for candidate in subset["candidates"]:
        assert candidate["p_uniform"] == candidate["M"] / 2 ** candidate["n_items"]


def test_survival_was_computed_at_r_opt_not_at_one_iteration(
    subset: dict[str, Any],
) -> None:
    """
    The circuits behind the survival figures are the ``r_opt`` circuits.

    The two-qubit count grows linearly in ``r``, so an instance with
    ``r_opt > 1`` must carry a count well above the ``r = 1`` cost of the same
    encoding.  A regression to WP2's ``r = 1`` numbers would collapse this.
    """
    multi = [c for c in subset["candidates"] if c["r_opt"] > 1]
    assert multi, "no multi-iteration instances to check"
    for candidate in multi:
        singles = [
            c
            for c in subset["candidates"]
            if c["r_opt"] == 1
            and c["n_items"] == candidate["n_items"]
            and c["n_sum"] == candidate["n_sum"]
        ]
        if not singles:
            continue
        cheapest_single = min(c["two_q_count"] for c in singles)
        assert candidate["two_q_count"] > cheapest_single, candidate["name"]


def test_recorded_encoding_agrees_with_the_library(subset: dict[str, Any]) -> None:
    """The subset did not re-derive register widths of its own."""
    for candidate in subset["candidates"]:
        enc = build_encoding(candidate["items"], candidate["target"])
        assert candidate["n_sum"] == enc.n_sum
        assert candidate["n_qubits"] == enc.n_ind + enc.n_sum
        assert candidate["r_opt"] == optimal_iterations(
            candidate["n_items"], candidate["M"]
        )


def test_infeasible_candidates_have_zero_ideal_probability(
    subset: dict[str, Any],
) -> None:
    """An M=0 instance has no marked state, on hardware or anywhere else."""
    for candidate in subset["candidates"]:
        if candidate["M"] == 0:
            assert candidate["p_ideal"] == 0.0
            assert candidate["p_uniform"] == 0.0
            assert "M=0" in candidate["p_ideal_source"]


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def test_survival_is_charged_to_physical_gates_only(subset: dict[str, Any]) -> None:
    """
    The stored survival uses the counts the median describes: no virtual ``rz``.

    Recomputed from ``one_q_physical`` / ``two_q_physical`` and the stored
    medians, and the physical count must be strictly below the inventory,
    because every transpiled D-G circuit contains ``rz``.
    """
    errors = subset["median_gate_errors"]
    assert "rz" in errors["virtual_gates"]
    eps1, eps2 = errors["one_qubit"], errors["two_qubit"]
    for candidate in subset["candidates"]:
        expected = (1 - eps2) ** candidate["two_q_physical"] * (1 - eps1) ** candidate[
            "one_q_physical"
        ]
        assert candidate["survival"] == pytest.approx(expected, rel=1e-12), candidate["name"]
        assert candidate["two_q_physical"] == candidate["two_q_count"], candidate["name"]
        assert candidate["one_q_physical"] < candidate["one_q_count"], candidate["name"]
        assert (
            candidate["one_q_physical"] + candidate["virtual_gate_count"]
            == candidate["one_q_count"]
        ), candidate["name"]


def test_edge_instance_is_a_genuine_search(subset: dict[str, Any]) -> None:
    """The recorded edge needs at least ``EDGE_MIN_R_OPT`` iterations."""
    if subset["edge_instance"] is None:
        pytest.skip("no edge instance in this subset")
    edge = next(c for c in subset["candidates"] if c["name"] == subset["edge_instance"])
    assert edge["r_opt"] >= select_hardware_subset.EDGE_MIN_R_OPT
    assert edge["M"] > 0 and edge["beats_random_by_2x"] is True


def test_sweep_selection_prefers_distinct_widths() -> None:
    """Two instances at the same ``n`` lose to one at each of two widths."""
    candidates = [
        row("n3-shallow", n_items=3, M=1, two_q_depth=100),
        row("n3-deeper", n_items=3, M=1, two_q_depth=110),
        row("n4", n_items=4, M=1, two_q_depth=500),
    ]
    chosen = [r["name"] for r in build_sweep.choose(candidates)]
    assert chosen == ["n3-shallow", "n4"]


def test_sweep_qualification_requires_M1_and_a_real_interior() -> None:
    """M != 1 or r_opt == 1 disqualifies."""
    subset = {
        "selected": [
            row("good", n_items=3, M=1),
            row("degenerate", n_items=3, M=2, m_class="2"),
            row("no-interior", n_items=2, M=1),
        ]
    }
    names = [r["name"] for r in build_sweep.qualifying(subset)]
    assert names == ["good"]


@pytest.fixture(scope="module")
def sweep() -> dict[str, Any]:
    """The committed sweep file, or a skip."""
    if not SWEEP_FILE.exists():
        pytest.skip(
            f"{SWEEP_FILE} not generated; run `python paper/scripts/build_sweep.py`."
        )
    return json.loads(SWEEP_FILE.read_text(encoding="utf-8"))


def test_sweep_file_is_two_instances_from_the_subset(
    sweep: dict[str, Any], subset: dict[str, Any]
) -> None:
    """The sweep introduces no circuit the hardware plan has not costed."""
    assert len(sweep["instances"]) == build_sweep.N_SWEEP_INSTANCES
    selected = {r["name"] for r in subset["selected"]}
    for record in sweep["instances"]:
        assert record["name"] in selected


def test_sweep_instances_satisfy_their_stated_rule(sweep: dict[str, Any]) -> None:
    """M = 1, r_opt >= 2, r_max = r_opt + 1, iterations enumerated."""
    for record in sweep["instances"]:
        assert record["M"] == 1
        assert record["r_opt"] >= build_sweep.MIN_R_OPT
        assert record["r_max"] == record["r_opt"] + 1
        assert record["iterations"] == list(range(1, record["r_max"] + 1))
        assert record["r_opt"] == optimal_iterations(record["n_items"], record["M"])


def test_sweep_instances_have_distinct_widths(sweep: dict[str, Any]) -> None:
    """The pair brackets the width range rather than sampling one width twice."""
    widths = [r["n_items"] for r in sweep["instances"]]
    assert len(set(widths)) == len(widths)
