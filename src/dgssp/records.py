"""
records.py

The frozen results schema: one :class:`ResultRecord` per executed
configuration (an instance run one way, on one backend).

The schema lives in the library, not in ``paper/scripts``, because the library
must be able to produce records (:meth:`dgssp.experiments.InstanceResult.
to_record`) and the library cannot import the paper scripts.
``paper/scripts/_common.Record`` subclasses :class:`ResultRecord` and adds only
the provenance stamp.

Compatibility rule: a file written by an older schema always loads (new keys
take their defaults), and a file written by a newer schema loads too (unknown
keys are ignored).  ``schema_version`` defaults to ``1`` so that an old file
loaded back is not mislabelled as current; :meth:`InstanceResult.to_record`
writes :data:`SCHEMA_VERSION`.

Serialised records are strict JSON: NaN and infinities become ``null``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

#: Version written by the current code.  1 = WP1-WP3 records; 2 adds the ZNE
#: fit, pathology, Mitiq block, physical gate counts and fold settings.
SCHEMA_VERSION: int = 2


def scale_key(scale: float) -> str:
    """
    The string a noise scale is stored under in ``ResultRecord.counts``.

    Integers keep the historical form (``"1"``, ``"3"``); other scales use
    Python's shortest round-trip ``repr``, so ``float(scale_key(x)) == x``
    exactly for every float.

    Parameters
    ----------
    scale:
        A (realised) noise scale.

    Returns
    -------
    str
        The key.
    """
    value = float(scale)
    return str(int(value)) if value.is_integer() else repr(value)


def json_safe(value: Any) -> Any:
    """
    Recursively replace NaN / inf with ``None`` and tuples with lists.

    Parameters
    ----------
    value:
        Any JSON-like structure (NumPy scalars are converted to Python ones).

    Returns
    -------
    Any
        A structure ``json.dumps`` writes as strict JSON.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    # NumPy arrays and scalars (np.float64 included) expose tolist().
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        return json_safe(tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class ResultRecord:
    """
    One executed configuration: an instance run one way, on one backend.

    A batch of instances becomes a *list* of records, not one record with
    lists inside it, so that a single row can always be traced to a single
    circuit execution.

    Attributes
    ----------
    instance:
        ``{"items", "target", "name"}`` -- the problem, verbatim.
    n_items:
        Index-register width (one qubit per item).
    n_sum:
        Sum-register width, from :func:`dgssp.encoding.build_encoding`.
    iterations:
        Grover iterations actually applied.
    executor:
        ``"ideal"``, ``"noisy"``, ``"optimized"`` or ``"hardware"``.
    mitigation_mode:
        ``"none"``, ``"zne_scalar"``, ``"zne_pointwise"`` or ``"mitiq"``.
    backend:
        Name of the backend the circuit ran on.
    shots:
        Shots per circuit, per noise scale.
    counts:
        Raw counts keyed by :func:`scale_key` of the **realised** scale.  An
        unmitigated run has the single key ``"1"``.  These are the only
        irrecoverable numbers in the file.
    metrics:
        Derived scalars with their confidence intervals: ``solution_probability``
        and ``solution_probability_ci``, ``mitigated_solution_probability``
        (and ``_ci``), distances to the reference (``tv_to_reference``,
        ``classical_fidelity``, ``hellinger`` with ``_ci`` where bootstrapped),
        ``p_uniform``, ``reference_distribution``, ...
    transpiled:
        ``{"two_q_count", "depth", "two_q_depth", "physical_qubits",
        "one_q_physical", "two_q_physical", "virtual_gate_count", "survival",
        "best_seed"}``.
    job_id:
        Runtime job id, else ``None``.
    provenance:
        Filled by ``paper/scripts/_common.record_provenance``.
    schema_version:
        ``1`` for records written before WP4; :data:`SCHEMA_VERSION` otherwise.
    calibration_timestamp:
        Calibration snapshot of the backend (ISO-8601), when known.
    zne_fit:
        The scalar fit: ``method``, ``scales`` (realised) and
        ``realised_scales_exact``, ``params``, ``cov``, ``dof``, ``chi2_red``,
        ``resolved``, ``reported_value``, ``ci``, ...
    pathology:
        :func:`dgssp.mitigation.zne.pathology_diagnostic` output.
    mitiq:
        Mitiq baseline: scales, realised two-qubit scales, per-scale values,
        extrapolation and interval.
    pointwise:
        The pointwise estimates (``clip`` and ``simplex``) with bootstrap
        intervals.
    renorm:
        Pointwise renormalisation reported (``"clip"`` / ``"simplex"``).
    fold_seed:
        Seed of the partial-folding (and Mitiq-folding) selection.
    """

    instance: dict[str, Any]
    n_items: int
    n_sum: int
    iterations: int
    executor: str
    mitigation_mode: str
    backend: str | None
    shots: int
    counts: dict[str, dict[str, int]]
    metrics: dict[str, Any] = field(default_factory=dict)
    transpiled: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    calibration_timestamp: str | None = None
    zne_fit: dict[str, Any] | None = None
    pathology: dict[str, Any] | None = None
    mitiq: dict[str, Any] | None = None
    pointwise: dict[str, Any] | None = None
    renorm: str | None = None
    fold_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Strict-JSON-serialisable view of the record.

        Returns
        -------
        dict
            All fields, with nothing dropped; NaN / inf written as ``None``.
        """
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResultRecord:
        """
        Rebuild a record from a loaded JSON object.

        Parameters
        ----------
        payload:
            A dictionary as written by ``save_record``.

        Returns
        -------
        ResultRecord
            (Or the subclass it is called on.)  Unknown keys are ignored and
            missing optional keys take their defaults, so files from any
            schema version load.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


__all__ = ["SCHEMA_VERSION", "ResultRecord", "scale_key", "json_safe"]
