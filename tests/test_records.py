"""
Tests for the frozen results schema (``dgssp.records``).

The schema promise: every file ever written loads (old keys only, or new keys
this version does not know), and every file written is strict JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dgssp.records import SCHEMA_VERSION, ResultRecord, json_safe, scale_key

DATA_DIR = Path(__file__).resolve().parents[1] / "paper" / "data"


def _minimal() -> dict:
    return {
        "instance": {"items": [1, 2, 3], "target": 5, "name": "t"},
        "n_items": 3,
        "n_sum": 4,
        "iterations": 1,
        "executor": "ideal",
        "mitigation_mode": "none",
        "backend": "aer_simulator",
        "shots": 100,
        "counts": {"1": {"101": 90, "000": 10}},
    }


def test_new_fields_default_and_old_payloads_load():
    """A WP1-WP3 payload (no new keys) loads with schema_version 1 and None blocks."""
    record = ResultRecord.from_dict(_minimal())
    assert record.schema_version == 1
    assert record.zne_fit is None and record.pathology is None and record.mitiq is None
    assert record.fold_seed is None and record.renorm is None


def test_unknown_keys_are_ignored():
    """A file from a later schema still loads."""
    payload = _minimal() | {"field_added_in_wp9": 1, "_source": "x.json"}
    assert ResultRecord.from_dict(payload).n_items == 3


@pytest.mark.parametrize(
    "path", sorted(DATA_DIR.glob("smoke*.json")), ids=lambda p: p.name
)
def test_committed_smoke_files_load(path):
    """Every smoke file in paper/data loads into the current schema."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload if isinstance(payload, list) else [payload]:
        record = ResultRecord.from_dict(item)
        assert record.counts
        assert record.schema_version in (1, SCHEMA_VERSION)


def test_to_dict_is_strict_json():
    """NaN / inf become null; numpy values become Python ones."""
    record = ResultRecord.from_dict(_minimal())
    record.zne_fit = {"value": float("nan"), "cov": np.array([[1.0, np.inf], [0.0, 2.0]])}
    record.metrics = {"x": np.float64(0.5), "n": np.int64(3)}
    text = json.dumps(record.to_dict(), allow_nan=False)
    back = json.loads(text)
    assert back["zne_fit"]["value"] is None
    assert back["zne_fit"]["cov"] == [[1.0, None], [0.0, 2.0]]
    assert back["metrics"] == {"x": 0.5, "n": 3}


def test_round_trip():
    """A record loaded back equals the record saved."""
    record = ResultRecord.from_dict(_minimal())
    record.schema_version = SCHEMA_VERSION
    record.zne_fit = {"method": "linear", "scales": [1.0, 1.4989247311827957]}
    assert ResultRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record


@pytest.mark.parametrize("scale", [1, 3.0, 5, 1.4989247311827957, 2.0041493775933608])
def test_scale_key_round_trips_exactly(scale):
    """Integers keep the historical form; everything else round-trips exactly."""
    key = scale_key(scale)
    assert float(key) == float(scale)
    if float(scale).is_integer():
        assert key == str(int(scale))


def test_json_safe_leaves_ordinary_values_alone():
    """Strings, ints, bools and None pass through."""
    value = {"a": "s", "b": 1, "c": True, "d": None, "e": [1.5, math.pi]}
    assert json_safe(value) == value
