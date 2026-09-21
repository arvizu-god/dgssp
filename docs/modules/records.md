# `dgssp.records`

## What the module does

The frozen results schema. One `ResultRecord` per executed configuration — an
instance run one way, on one backend — so a row can always be traced to a
single execution. A batch is a *list* of records.

The schema lives in the library, not in `paper/scripts`, because the library
produces records (`InstanceResult.to_record()`) and cannot import the paper
scripts. `paper/scripts/_common.Record` is a subclass that adds only the
provenance stamp.

**Compatibility.** Every file ever written loads: missing keys take their
defaults (old `smoke*.json` files load with `schema_version = 1` and `None`
blocks), unknown keys are ignored (a file from a later schema loads).
`to_record()` writes `SCHEMA_VERSION = 2`.

**Strict JSON.** `to_dict()` replaces NaN/inf with `null` and NumPy values with
Python ones.

## Contents

| Name | Kind |
|---|---|
| `SCHEMA_VERSION` | constant (`2`) |
| `ResultRecord` | dataclass |
| `scale_key` | function |
| `json_safe` | function |

---

### `ResultRecord` — dataclass

**Schema-1 fields** (WP1–WP3): `instance` (`{"items", "target", "name"}`),
`n_items`, `n_sum` (from `encoding.build_encoding`), `iterations`, `executor`,
`mitigation_mode` (`"none"`, `"zne_scalar"`, `"zne_pointwise"`, `"mitiq"`),
`backend` (name), `shots` (per circuit, per scale), `counts` (raw counts keyed
by `scale_key` of the **realised** scale — the only irrecoverable numbers),
`metrics`, `transpiled`, `job_id`, `provenance`.

**Schema-2 fields** (WP4, all defaulted):

- `schema_version: int = 1`
- `calibration_timestamp: str | None` — the backend's calibration snapshot.
- `zne_fit: dict | None` — the scalar fit: `method`, `scales` (realised),
  `realised_scales_exact` (`[numerator, denominator]`), `nominal_scales`,
  `observed`, `sigma`, `params`, `param_names`, `cov`, `residuals`, `dof`,
  `chi2`, `chi2_red` (`null` when `dof == 0`), `asymptote`, `null_value`,
  `resolved` / `resolved_scales` / `n_resolved_scales`, `identifiable` /
  `informative_scales` / `n_informative_scales`, `rate_at_bound`,
  `exp_rate_max`, `reported_value` / `reported_source`, `ci`, `bootstrap_se`,
  `fold_seed`. A fit that is unresolved or unidentified keeps its raw `value`
  but reports `value_at_min_scale`; see
  [`mitigation.zne`](mitigation.zne.md).
- `pathology: dict | None` — `zne.pathology_diagnostic`.
- `mitiq: dict | None` — Mitiq's scales, realised two-qubit scales, per-scale
  values, extrapolation and interval.
- `pointwise: dict | None` — clip and simplex distributions with bootstrap
  intervals.
- `renorm: str | None`, `fold_seed: int | None`.

`metrics` (schema 2) carries `solution_probability` and
`solution_probability_ci`, `mitigated_solution_probability(_ci)`,
`mitiq_solution_probability(_ci)`, `tv_to_reference` (`{point, lo, hi}`),
`classical_fidelity`, `hellinger`, `p_uniform`, `reference_distribution`.
`transpiled` carries `two_q_count`, `depth`, `two_q_depth`,
`physical_qubits`, `one_q_physical`, `two_q_physical`, `virtual_gate_count`,
`survival`, `best_seed`.

**Methods**

- `to_dict() -> dict` — strict-JSON view, nothing dropped.
- `from_dict(payload) -> ResultRecord` — classmethod; unknown keys ignored,
  missing optional keys defaulted; returns the subclass it is called on.

### `scale_key(scale) -> str`

Integers keep the historical form (`"1"`, `"3"`); other scales use Python's
shortest round-trip `repr`, so `float(scale_key(x)) == x` exactly.

### `json_safe(value) -> Any`

Recursive: NaN/inf → `None`, tuples → lists, NumPy arrays/scalars → Python.
