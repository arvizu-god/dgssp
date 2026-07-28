"""Packaging smoke tests: does `import dgssp` actually work?"""

from __future__ import annotations

import importlib

import pytest


def test_version_is_exposed():
    """The package reports its version."""
    import dgssp

    assert dgssp.__version__ == "0.1.0"


def test_all_exports_resolve():
    """Every name in __all__ exists -- catches stale exports like solve_ssp."""
    import dgssp

    missing = [name for name in dgssp.__all__ if not hasattr(dgssp, name)]
    assert missing == [], f"__all__ lists undefined names: {missing}"


def test_star_import_is_clean():
    """`from dgssp import *` binds every advertised name."""
    namespace: dict = {}
    exec("from dgssp import *", namespace)  # noqa: S102 - deliberate

    import dgssp

    missing = [n for n in dgssp.__all__ if n not in namespace]
    assert missing == []


def test_removed_symbols_are_gone():
    """solve_ssp and the QPE solver were deleted, not left dangling."""
    import dgssp

    assert not hasattr(dgssp, "solve_ssp")
    assert "solve_ssp" not in dgssp.__all__
    assert not hasattr(dgssp, "QPESSPSolver")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dgssp.solvers.qpe")


@pytest.mark.parametrize(
    "module",
    [
        "dgssp.instance",
        "dgssp.encoding",
        "dgssp.decoding",
        "dgssp.runtime",
        "dgssp.backends",
        "dgssp.transpilation",
        "dgssp.execution",
        "dgssp.experiments",
        "dgssp.api",
        "dgssp.cli",
        "dgssp.solvers",
        "dgssp.solvers.base",
        "dgssp.solvers.classical",
        "dgssp.solvers.draper_grover",
        "dgssp.mitigation",
        "dgssp.mitigation.zne",
        "dgssp.mitigation.mitiq_baseline",
    ],
)
def test_every_module_imports(module):
    """No module has a broken or undeclared import."""
    importlib.import_module(module)


def test_py_typed_marker_is_shipped():
    """PEP 561 marker is present so downstream users get the type hints."""
    import pathlib

    import dgssp

    marker = pathlib.Path(dgssp.__file__).parent / "py.typed"
    assert marker.exists()


def test_mitiq_baseline_fails_helpfully_without_mitiq():
    """The optional extra is reported by name when it is missing."""
    from dgssp.mitigation.mitiq_baseline import mitiq_is_available

    if mitiq_is_available():
        pytest.skip("mitiq is installed")

    from dgssp.mitigation.mitiq_baseline import _require_mitiq

    with pytest.raises(ImportError, match="dgssp\\[mitiq\\]"):
        _require_mitiq()


def test_cli_help_runs():
    """The console entry point builds its parser without importing hardware."""
    from dgssp.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
