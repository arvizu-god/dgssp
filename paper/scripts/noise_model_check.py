"""
noise_model_check.py

How far is ibm_fez from its own noise model, and is the gap calibration age?

The WP6 job's signal at ``lambda = 1`` was roughly the *cube* of what the noise
model built from the plan's calibration predicted.  Two follow-ups settle what
that means, both from the frozen WP6 inputs (manifest, QPY circuits, hardware
counts) and neither using QPU time:

``--fetch-calibration``  (needs ``QISKIT_IBM_TOKEN``; no QPU time)
    Fetch the calibration that was in force while the job *ran*
    (``backend.target_history(t)``: the properties "closest to, but older
    than" ``t``; ``t`` defaults to the job's ``running`` timestamp in
    ``paper/hardware/usage.json``) and freeze it as
    ``paper/hardware/calibration_<label>_<date>.json`` (label ``exec``).
    The method is checked first: ``target_history`` just after the plan's
    calibration stamp must rebuild the plan's snapshot exactly, so the new
    file differs from the old one only by what the device recalibrated.  A
    report ``calibration_<label>_<date>_fetch.json`` records both checks, the
    ISA check of the 47 circuits against the new target, and the errors on the
    qubits the job used.

``--predict``  (offline)
    The manifest's circuits, same shots, same options, same seed, on an
    ``AerSimulator`` built from ``--calibration`` (default: the ``exec``
    snapshot), optionally with every calibrated error times ``--alpha``.
    Records (``executor="noise_model_prediction"``, with a
    ``metrics["prediction"]`` block) go to
    ``paper/data/hw_predict_<label>_<date>/``.  With the plan's calibration and
    ``--alpha 1`` this reproduces the dry run record for record -- the
    acceptance check of this mode.

``--fit-alpha``  (offline)
    Simulate the job at every ``--alphas`` value (counts cached per alpha in
    ``paper/data/hw_alpha_<label>_<date>/``, so an interrupted run resumes),
    then find the ``alpha`` whose prediction matches the hardware's
    ``P(solution)``.  Fitted separately on the ``lambda = 1`` PUBs (the
    headline: "the device behaves like its calibration with every error times
    alpha") and on ``lambda = 3`` / ``lambda = 5`` (if one alpha explains the
    folded circuits too, the model is consistent; if not, the extra noise is
    not a uniform rescaling).  Writes ``fit.json`` there and prints the table.

What alpha scales: every instruction's calibrated ``error`` (gates, readout,
reset); durations and ``T1`` / ``T2`` stay as calibrated (see
:func:`dgssp.backends.scale_calibration_errors`).  ``--scale-only
{2q,1q,readout}`` scales one error class alone and keeps the others as
calibrated (default ``all``, the paper's fit); its counts are cached in
``hw_alpha_<label>_<class>_<date>/`` and its predictions go to
``hw_predict_<label>_x<alpha>_<class>_<date>/``, so the ``all`` files are never
touched.  Only PUBs whose instance has a signal to lose are fitted: ``M = 0``
and ``M = N/2`` sit at ``P_uniform`` whatever the noise.

``--fit-alpha`` never replaces an existing ``fit.json``: it refits from the
cache and reports whether the stored fit is reproduced exactly (``--force``
to replace it).

Run::

    python paper/scripts/noise_model_check.py --fetch-calibration
    python paper/scripts/noise_model_check.py --predict
    python paper/scripts/noise_model_check.py --fit-alpha
    python paper/scripts/noise_model_check.py --fit-alpha --scale-only 2q
    python paper/scripts/noise_model_check.py --predict --calibration paper/hardware/calibration_2026-10-25.json --label plan
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from qiskit import QuantumCircuit  # noqa: E402

from _common import DATA_DIR, HARDWARE_DIR, record_provenance, utc_timestamp  # noqa: E402
from dgssp.backends import (  # noqa: E402
    ERROR_CLASSES,
    CalibrationSnapshotBackend,
    ErrorClass,
    calibration_payload,
    load_noise_model,
    median_gate_errors,
    save_noise_model,
    scale_calibration_errors,
)
from dgssp.execution import exact_index_distribution  # noqa: E402
from dgssp.instance import SubsetSumInstance  # noqa: E402
from dgssp.metrics import distribution_solution_probability, solution_probability  # noqa: E402
from dgssp.mitigation.zne import signal_direction  # noqa: E402
from dgssp.runtime import counts_from_result, submit_sampler_job  # noqa: E402
from hardware_batch import (  # noqa: E402
    INSTANCE_ENV,
    Paths,
    build_records,
    connect,
    decode_result_file,
    job_inputs,
    json_plain,
    latest_manifest_date,
    load_manifest,
    print_records_summary,
    rel,
    resolve,
    sha256_file,
    write_json,
    write_records,
)
from sim_campaign import SEED_SIMULATOR, throttle  # noqa: E402

FIT_SCHEMA = "dgssp/paper/hardware-alpha-fit/1"
FETCH_SCHEMA = "dgssp/paper/calibration-fetch/1"

#: Default alpha grid: the calibration itself up to 5x, finer where the
#: WP6 cube rule (signal^3, i.e. roughly 3x per gate) puts the answer.
DEFAULT_ALPHAS: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)

#: Resolution of the alpha search between grid points (monotone cubic
#: interpolation of each PUB's predicted P(solution)).
ALPHA_STEP = 0.001

#: Bootstrap of the alpha fit: hardware and simulated counts resampled.
FIT_BOOTSTRAP_B = 500
FIT_BOOTSTRAP_SEED = 0

#: ``target_history`` returns the properties *older* than the time asked;
#: the plan's stamp plus this lands on the plan's own calibration.
PLAN_STAMP_OFFSET = timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def parse_utc(text: str) -> datetime:
    """An ISO-8601 time (``Z`` or an offset) as an aware UTC datetime."""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SystemExit(f"--at {text!r} has no time zone; add Z or an offset.")
    return parsed.astimezone(timezone.utc)


def iso(value: Any) -> str | None:
    """A datetime (or anything) as an ISO string."""
    if value is None:
        return None
    fmt = getattr(value, "isoformat", None)
    return fmt() if callable(fmt) else str(value)


def job_running_time(paths: Paths) -> tuple[str, datetime]:
    """
    The hardware job's id and the moment it started running.

    Read from ``usage.json`` (written by ``hardware_batch.py --collect`` from
    ``job.metrics()``), for the job of ``jobs_<date>.json``.
    """
    jobs = json.loads(paths.jobs.read_text(encoding="utf-8"))
    job_id = jobs["jobs"][0]["job_id"]
    usage = json.loads(paths.usage_json.read_text(encoding="utf-8"))
    running = (usage.get(job_id) or {}).get("metrics", {}).get("timestamps", {}).get("running")
    if running is None:
        raise SystemExit(f"No 'running' timestamp for {job_id} in {rel(paths.usage_json)}; "
                         "pass --at.")
    return job_id, parse_utc(running)


def used_qubits(manifest: Mapping[str, Any]) -> list[int]:
    """Every physical qubit any circuit of the job touches."""
    return sorted({int(q) for row in manifest["circuits"] for q in row["physical_qubits"]})


def used_qubit_errors(payload: Mapping[str, Any], qubits: Sequence[int]) -> dict[str, Any]:
    """
    Median calibrated errors on the qubits (and couplers) the job used.

    Parameters
    ----------
    payload:
        :func:`dgssp.backends.calibration_payload` of a device.
    qubits:
        :func:`used_qubits`.

    Returns
    -------
    dict
        ``{name: {"median", "max", "n"}}`` for ``cz``, ``sx``, ``measure``,
        plus the median ``T1`` / ``T2`` (seconds).
    """
    wanted = set(qubits)
    out: dict[str, Any] = {}
    for name in ("cz", "sx", "x", "measure"):
        errs = [
            float(row["error"])
            for row in payload["instructions"].get(name, [])
            if row["error"] is not None and row["qubits"] is not None
            and set(row["qubits"]) <= wanted
        ]
        if errs:
            out[name] = {"median": float(np.median(errs)), "max": max(errs), "n": len(errs)}
    props = payload.get("qubit_properties") or []
    for key in ("t1", "t2"):
        vals = [props[q][key] for q in qubits if q < len(props) and props[q].get(key)]
        if vals:
            out[f"{key}_median_s"] = float(np.median(vals))
    return out


def payload_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Where two calibration payloads differ (empty = identical)."""
    diffs: list[str] = []
    for key in sorted(set(a) | set(b)):
        if key == "instructions":
            continue
        if a.get(key) != b.get(key):
            diffs.append(key)
    ia, ib = a.get("instructions", {}), b.get("instructions", {})
    for name in sorted(set(ia) | set(ib)):
        rows_a = {json.dumps(r["qubits"]): r for r in ia.get(name, [])}
        rows_b = {json.dumps(r["qubits"]): r for r in ib.get(name, [])}
        for qargs in sorted(set(rows_a) | set(rows_b)):
            if rows_a.get(qargs) != rows_b.get(qargs):
                diffs.append(f"{name}{qargs}")
    return diffs


def isa_violations(target: Any, circuits: Sequence[QuantumCircuit]) -> list[str]:
    """Instructions of the circuits the target does not support, as text."""
    bad: list[str] = []
    for index, circuit in enumerate(circuits):
        for instruction in circuit.data:
            name = instruction.operation.name
            if name == "barrier":
                continue
            qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
            if not target.instruction_supported(name, qargs):
                bad.append(f"PUB {index}: {name}{list(qargs)}")
    return bad


def signal_pubs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Every PUB whose instance has a signal to lose, with what the fit needs.

    Returns
    -------
    list[dict]
        ``{"pub", "key", "scale", "solution_bitstrings", "p_uniform",
        "p_ideal"}`` in PUB order; ``M = 0`` and ``M = N/2`` excluded.
    """
    scale_of = {int(c["pub_index"]): float(c["nominal_scale"]) for c in manifest["circuits"]}
    out = []
    for row in manifest["records"]:
        instance = SubsetSumInstance(items=list(row["items"]), target=row["target"],
                                     name=row["instance"])
        reference = exact_index_distribution(instance, int(row["iterations"]))
        sols = list(row["solution_bitstrings"])
        p_ideal = distribution_solution_probability(reference, sols)
        if signal_direction(p_ideal, float(row["p_uniform"])) == 0:
            continue
        for pub in row["pub_indices"]:
            out.append({
                "pub": int(pub),
                "key": row["key"],
                "scale": scale_of[int(pub)],
                "solution_bitstrings": sols,
                "p_uniform": float(row["p_uniform"]),
                "p_ideal": p_ideal,
            })
    return sorted(out, key=lambda r: r["pub"])


def hardware_counts(paths: Paths) -> list[dict[str, int]]:
    """The job's counts per PUB, decoded from the raw result ``--collect`` saved."""
    jobs = json.loads(paths.jobs.read_text(encoding="utf-8"))
    counts: list[dict[str, int]] = []
    for entry in jobs["jobs"]:
        raw = paths.hardware_out_dir / "raw" / f"result_{entry['job_id']}.json"
        if not raw.exists():
            raise SystemExit(f"{rel(raw)} not found; run hardware_batch.py --collect first.")
        counts.extend(counts_from_result(decode_result_file(raw)))
    return counts


def simulate(
    manifest: Mapping[str, Any],
    circuits: Sequence[QuantumCircuit],
    device: Any,
    *,
    max_threads: int | None,
    meta: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[str]]:
    """
    The manifest's circuits on ``AerSimulator.from_backend(device)``.

    Exactly the dry run's path -- same shots, options, seed and
    :func:`dgssp.runtime.submit_sampler_job` -- with another noise source.

    Returns
    -------
    tuple
        Counts per PUB, and the options the local path ignored.
    """
    from qiskit_aer import AerSimulator

    circuits, shots, options = job_inputs(manifest, circuits)
    simulator = throttle(AerSimulator.from_backend(device), max_threads)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        job = submit_sampler_job(simulator, circuits, shots=shots, options=options,
                                 seed_simulator=SEED_SIMULATOR, run_log=None, meta=dict(meta))
        result = job.result()
    ignored = sorted({str(w.message) for w in caught if "local testing mode" in str(w.message)})
    return counts_from_result(result), ignored


def noise_device(
    calibration: Path, alpha: float, which: ErrorClass = "all"
) -> tuple[Any, Any, dict[str, Any]]:
    """
    The snapshot of ``calibration``, and the device to simulate (scaled if ``alpha != 1``).

    ``which`` selects the errors ``alpha`` multiplies
    (:func:`dgssp.backends.scale_calibration_errors`); at ``alpha = 1`` every
    choice is the calibration itself.

    Returns
    -------
    tuple
        ``(snapshot, device, scale_info)``.
    """
    snapshot = load_noise_model(calibration)
    if alpha == 1.0:
        return snapshot, snapshot.backend, {"alpha": 1.0, "n_scaled": 0, "n_capped": 0}
    device, info = scale_calibration_errors(snapshot.backend, alpha, which=which)
    return snapshot, device, info


def median_errors(calibration: Path, alpha: float = 1.0, which: ErrorClass = "all") -> dict[str, Any]:
    """
    The device's median gate errors on ``calibration``, the ``which`` errors times ``alpha``.

    What :func:`dgssp.metrics.survival_estimate` takes.  Built from the same
    snapshot and the same scaling (with its cap) as :func:`noise_device`, so a
    survival at ``alpha`` describes the device the alpha fit describes.

    Returns
    -------
    dict
        :func:`dgssp.backends.median_gate_errors` of the (scaled) device.
    """
    _, device, _ = noise_device(calibration, alpha, which)
    return median_gate_errors(device)


def class_suffix(which: str) -> str:
    """``""`` for ``all`` (the paper's file names), ``"_<class>"`` otherwise."""
    return "" if which == "all" else f"_{which}"


def alpha_cache_dir(data_dir: Path, label: str, which: str, date: str) -> Path:
    """``hw_alpha_<label>[_<class>]_<date>``: one cache per calibration and error class."""
    return data_dir / f"hw_alpha_{label}{class_suffix(which)}_{date}"


def what_alpha_scales(which: str) -> str:
    """The sentence ``fit.json`` records for ``which``."""
    if which == "all":
        return ("every calibrated instruction error (gates, readout, reset); "
                "durations and T1/T2 as calibrated")
    return (f"only the '{which}' errors (dgssp.backends.error_class); every other error, "
            "durations and T1/T2 as calibrated")


def default_calibration(paths: Paths) -> Path:
    """The execution-time snapshot ``--fetch-calibration`` writes."""
    return paths.hardware_dir / f"calibration_exec_{paths.date}.json"


def calibration_label(calibration: Path, date: str) -> str:
    """``calibration_exec_<date>.json`` -> ``exec``; the plan's own file -> ``plan``."""
    stem = calibration.stem.removeprefix("calibration_").removesuffix(date).strip("_")
    return stem or "plan"


# ---------------------------------------------------------------------------
# --fetch-calibration
# ---------------------------------------------------------------------------


def run_fetch(args: argparse.Namespace, paths: Paths) -> int:
    """``--fetch-calibration``: freeze the calibration in force at ``--at``."""
    from qiskit_aer.noise import NoiseModel

    manifest, _, circuits = load_manifest(paths.manifest)
    if args.at:
        job_id, at = None, parse_utc(args.at)
    else:
        job_id, at = job_running_time(paths)
    out = paths.hardware_dir / f"calibration_{args.label}_{paths.date}.json"
    if out.exists() and not args.force:
        print(f"{rel(out)} exists (a snapshot is evidence); --force to replace it.")
        return 1

    name = manifest["backend"]["name"]
    _, backend, _ = connect(name, offline=False, instance=args.instance)
    print(f"{name}: calibration in force at {at.isoformat()}"
          + (f" (job {job_id} started running)" if job_id else ""))

    # 1. The method: target_history just after the plan's stamp must rebuild
    #    the plan's snapshot, or the new file is not comparable with it.
    plan_snapshot = load_noise_model(resolve(manifest["backend"]["calibration_file"]))
    plan_at = parse_utc(manifest["backend"]["calibration_timestamp"]) + PLAN_STAMP_OFFSET
    plan_props = backend.properties(datetime=plan_at)
    plan_target = backend.target_history(datetime=plan_at)
    plan_rebuilt = CalibrationSnapshotBackend(
        plan_target, name=name, calibration_timestamp=iso(plan_props.last_update_date)
    )
    method_diffs = payload_differences(
        json_plain(calibration_payload(plan_snapshot.backend)),
        json_plain(calibration_payload(plan_rebuilt)),
    )
    method_ok = not method_diffs
    print(f"method check (history at plan stamp + {PLAN_STAMP_OFFSET} == plan snapshot): "
          f"{'PASS' if method_ok else f'FAIL, {len(method_diffs)} entries differ'}")

    # 2. The calibration in force while the job ran.
    props = backend.properties(datetime=at)
    if props is None:
        print("The service returned no properties for that time.")
        return 1
    target = backend.target_history(datetime=at)
    snapshot = CalibrationSnapshotBackend(
        target, name=name, calibration_timestamp=iso(props.last_update_date)
    )
    save_noise_model(out, NoiseModel.from_backend(snapshot), snapshot, overwrite=args.force,
                     raw_properties=json_plain(props.to_dict()))
    exec_snapshot = load_noise_model(out)
    bad = isa_violations(exec_snapshot.backend.target, circuits)

    qubits = used_qubits(manifest)
    plan_payload = json_plain(calibration_payload(plan_snapshot.backend))
    exec_payload = json_plain(calibration_payload(exec_snapshot.backend))
    changed = payload_differences(plan_payload, exec_payload)
    report = {
        "schema": FETCH_SCHEMA,
        "date": paths.date,
        "backend": name,
        "requested_at_utc": at.isoformat(),
        "job_id": job_id,
        "calibration_file": rel(out),
        "calibration_sha256": sha256_file(out),
        "calibration_timestamp": exec_snapshot.calibration_timestamp,
        "plan_calibration_timestamp": plan_snapshot.calibration_timestamp,
        "method_check": {
            "requested_at_utc": plan_at.isoformat(),
            "returned_timestamp": iso(plan_props.last_update_date),
            "identical_to_plan_snapshot": method_ok,
            "differences": method_diffs[:50],
            "n_differences": len(method_diffs),
        },
        "isa_check": {"ok": not bad, "violations": bad[:50], "n_violations": len(bad)},
        "entries_changed_since_plan": len(changed),
        "used_qubits": qubits,
        "used_qubit_errors": {
            "plan": used_qubit_errors(plan_payload, qubits),
            "exec": used_qubit_errors(exec_payload, qubits),
        },
        "created_utc": utc_timestamp(),
        "provenance": record_provenance(),
    }
    report_path = write_json(out.with_name(f"{out.stem}_fetch.json"), report)

    print(f"calibration stamp: {exec_snapshot.calibration_timestamp} "
          f"(plan: {plan_snapshot.calibration_timestamp}); "
          f"{len(changed)} calibration entries changed since the plan")
    print(f"ISA check of the {len(circuits)} circuits: "
          f"{'ok' if not bad else f'{len(bad)} unsupported instructions'}")
    print(f"\nerrors on the {len(qubits)} qubits the job used (median / max):")
    for key in ("cz", "sx", "measure"):
        p, e = report["used_qubit_errors"]["plan"].get(key), report["used_qubit_errors"]["exec"].get(key)
        if p and e:
            print(f"  {key:<8} plan {p['median']:.2e} / {p['max']:.2e}   "
                  f"exec {e['median']:.2e} / {e['max']:.2e}")
    for key in ("t1_median_s", "t2_median_s"):
        p, e = report["used_qubit_errors"]["plan"].get(key), report["used_qubit_errors"]["exec"].get(key)
        if p and e:
            print(f"  {key:<12} plan {p * 1e6:.0f} us   exec {e * 1e6:.0f} us")
    print(f"\nwrote {rel(out)}\nwrote {rel(report_path)}")
    print("\nNext (offline, no token): python paper/scripts/noise_model_check.py --predict")
    return 0 if method_ok and not bad else 1


# ---------------------------------------------------------------------------
# --predict
# ---------------------------------------------------------------------------


def signal_kept(p_sol: float | None, p_ideal: float, p_uniform: float) -> float | None:
    """``(P_sol - P_uniform) / (P_ideal - P_uniform)``: 1 perfect, 0 random."""
    if p_sol is None or math.isclose(p_ideal, p_uniform):
        return None
    return (p_sol - p_uniform) / (p_ideal - p_uniform)


def scale1_p_sol(directory: Path, key: str) -> float | None:
    """``P(solution)`` at ``lambda = 1`` of one stored record, or ``None``."""
    path = directory / f"{key}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    value = record.get("metrics", {}).get("solution_probability")
    return float(value) if value is not None else None


def run_predict(args: argparse.Namespace, paths: Paths) -> int:
    """``--predict``: the job on another calibration, optionally scaled."""
    manifest, manifest_sha, circuits = load_manifest(paths.manifest)
    calibration = Path(args.calibration) if args.calibration else default_calibration(paths)
    if not calibration.exists():
        print(f"{rel(calibration)} not found; run --fetch-calibration first.")
        return 2
    alpha = float(args.alpha)
    which = args.scale_only
    label = args.label or calibration_label(calibration, paths.date)
    if alpha != 1.0:
        label = f"{label}_x{alpha:g}{class_suffix(which)}"
    out_dir = paths.data_dir / f"hw_predict_{label}_{paths.date}"

    snapshot, device, scale_info = noise_device(calibration, alpha, which)
    print(f"predict: {len(circuits)} circuits on the {snapshot.backend_name} calibration of "
          f"{snapshot.calibration_timestamp}, "
          f"{'errors' if which == 'all' else f'{which} errors'} x {alpha:g}"
          + (f" ({scale_info['n_capped']} capped at 0.5)" if scale_info.get("n_capped") else ""))
    counts, ignored = simulate(manifest, circuits, device, max_threads=args.max_threads,
                               meta={"stage": "wp6_predict", "label": label})

    prediction = {
        "calibration_file": rel(calibration),
        "calibration_sha256": sha256_file(calibration),
        "calibration_timestamp": snapshot.calibration_timestamp,
        "alpha": alpha,
        "scale_info": scale_info,
        "seed_simulator": SEED_SIMULATOR,
        "plan_calibration_file": manifest["backend"]["calibration_file"],
    }
    # The records describe *this* calibration, not the plan's.
    stamped = copy.deepcopy(dict(manifest))
    stamped["backend"]["calibration_file"] = rel(calibration)
    stamped["backend"]["calibration_timestamp"] = snapshot.calibration_timestamp
    jobs_doc = {"source": "prediction", "execution_mode": "local", "ignored_options": ignored}
    job_id = f"predict-{label}-{paths.date}"
    records = build_records(stamped, manifest_sha, circuits, counts, [job_id] * len(counts),
                            jobs_doc, device, executor="noise_model_prediction",
                            prediction=prediction)
    written, unchanged, conflicts = write_records(out_dir, records, force=args.force)
    print_records_summary([s for _, _, s in records], f"prediction '{label}':")

    print("\nsignal kept at lambda = 1 (1 = perfect, 0 = random): "
          "plan model / this model / hardware")
    for _, _, s in records:
        if s["kind"] != "zne":
            continue
        p_u = float(next(r["p_uniform"] for r in manifest["records"] if r["key"] == s["key"]))
        if signal_direction(s["p_ideal"], p_u) == 0:
            continue
        cells = [
            signal_kept(scale1_p_sol(paths.dryrun_dir, s["key"]), s["p_ideal"], p_u),
            signal_kept(s["p_sol"], s["p_ideal"], p_u),
            signal_kept(scale1_p_sol(paths.hardware_out_dir, s["key"]), s["p_ideal"], p_u),
        ]
        text = "  ".join("   -  " if c is None else f"{c:6.3f}" for c in cells)
        print(f"  {s['key']:<16} {text}")
    print(f"\n{written} record(s) written, {unchanged} unchanged -> {rel(out_dir)}")
    if conflicts:
        print("CONFLICT (kept the old files; --force to replace):\n  " + "\n  ".join(conflicts))
        return 1
    return 0


# ---------------------------------------------------------------------------
# --fit-alpha
# ---------------------------------------------------------------------------


def _sigma2(p: np.ndarray, shots: float) -> np.ndarray:
    """Binomial variance, Jeffreys-smoothed as :func:`dgssp.metrics.proportion_sigma`."""
    smoothed = (p * shots + 0.5) / (shots + 1.0)
    return smoothed * (1.0 - smoothed) / shots


def _alpha_argmin(
    alphas: np.ndarray, sim: np.ndarray, sim_shots: float, hw: np.ndarray, hw_shots: float
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """``(alpha_hat, chi2_min, fine_grid, chi2_profile)`` for one data set."""
    from scipy.interpolate import PchipInterpolator

    fine = np.arange(alphas[0], alphas[-1] + ALPHA_STEP / 2, ALPHA_STEP)
    curves = PchipInterpolator(alphas, sim, axis=0)(fine)  # (n_fine, n_pub)
    chi2 = (((hw - curves) ** 2) / (_sigma2(hw, hw_shots) + _sigma2(curves, sim_shots))).sum(axis=1)
    best = int(np.argmin(chi2))
    return float(fine[best]), float(chi2[best]), fine, chi2


def fit_alpha(
    alphas: Sequence[float],
    sim: np.ndarray,
    sim_shots: int,
    hw: np.ndarray,
    hw_shots: int,
    *,
    groups: Sequence[str] | None = None,
    B: int = FIT_BOOTSTRAP_B,
    seed: int = FIT_BOOTSTRAP_SEED,
    keep_draws: bool = False,
) -> dict[str, Any]:
    """
    The noise scale that best matches hardware ``P(solution)`` to prediction.

    Each PUB's predicted ``P(solution)`` is interpolated in ``alpha`` (monotone
    cubic between the simulated grid points) and ``alpha`` minimises
    ``chi2 = sum (hw - sim(alpha))^2 / (var_hw + var_sim)``.

    Parameters
    ----------
    alphas:
        The simulated grid, increasing.
    sim:
        ``(len(alphas), n_pub)`` predicted ``P(solution)``.
    sim_shots, hw_shots:
        Shots behind each simulated / hardware value.
    hw:
        ``(n_pub,)`` hardware ``P(solution)``.
    groups:
        A record key per PUB, for the leave-one-record-out spread.
    B, seed:
        Bootstrap resamples (hardware *and* simulated counts, binomially) and
        seed.
    keep_draws:
        Also return the ``B`` bootstrap values of ``alpha`` (``draws``), e.g.
        to bootstrap a difference of two independent fits.

    Returns
    -------
    dict
        ``alpha``, ``ci`` (bootstrap percentile 95 %: shot noise only),
        ``ci_scaled`` (``ci`` widened about ``alpha`` by
        ``sqrt(max(1, chi2_red))``, the usual correction when the scatter
        exceeds shot noise -- the interval to quote), ``bootstrap_se``,
        ``chi2_min``, ``dof``, ``chi2_red``, ``at_grid_edge``, ``n_pubs``,
        ``loo`` (``{record: alpha}`` with that record left out, when
        ``groups`` is given) and ``profile`` (``chi2`` at the grid points).
    """
    a = np.asarray(alphas, dtype=float)
    s = np.asarray(sim, dtype=float)
    h = np.asarray(hw, dtype=float)
    if a.ndim != 1 or a.size < 2 or np.any(np.diff(a) <= 0):
        raise ValueError("alphas must be an increasing grid of at least two values.")
    if s.shape != (a.size, h.size):
        raise ValueError(f"sim must be {(a.size, h.size)}, got {s.shape}.")
    alpha_hat, chi2_min, fine, chi2 = _alpha_argmin(a, s, sim_shots, h, hw_shots)
    edge = bool(math.isclose(alpha_hat, a[0]) or math.isclose(alpha_hat, a[-1], abs_tol=ALPHA_STEP))

    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        h_b = rng.binomial(hw_shots, h) / hw_shots
        s_b = rng.binomial(sim_shots, s) / sim_shots
        boot[b] = _alpha_argmin(a, s_b, sim_shots, h_b, hw_shots)[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    loo: dict[str, float] | None = None
    if groups is not None:
        keys = list(dict.fromkeys(groups))
        loo = {}
        for key in keys:
            keep = np.array([g != key for g in groups])
            if keep.sum() >= 1:
                loo[key] = _alpha_argmin(a, s[:, keep], sim_shots, h[keep], hw_shots)[0]

    dof = int(h.size - 1)
    chi2_red = chi2_min / dof if dof > 0 else None
    widen = math.sqrt(max(1.0, chi2_red or 1.0))
    grid_chi2 = [float(chi2[int(np.argmin(np.abs(fine - x)))]) for x in a]
    out: dict[str, Any] = {
        "alpha": alpha_hat,
        "ci": [float(lo), float(hi)],
        "ci_scaled": [alpha_hat - widen * (alpha_hat - float(lo)),
                      alpha_hat + widen * (float(hi) - alpha_hat)],
        "bootstrap_se": float(np.std(boot, ddof=1)),
        "B": int(B),
        "seed": int(seed),
        "chi2_min": chi2_min,
        "dof": dof,
        "chi2_red": chi2_red,
        "at_grid_edge": edge,
        "n_pubs": int(h.size),
        "loo": loo,
        "profile": {"alphas": a.tolist(), "chi2": grid_chi2},
    }
    if keep_draws:
        out["draws"] = boot.tolist()
    return out


def cached_counts(
    cache: Path,
    alpha: float,
    calibration: Path,
    manifest: Mapping[str, Any],
    circuits: Sequence[QuantumCircuit],
    max_threads: int | None,
    which: ErrorClass = "all",
) -> list[dict[str, int]]:
    """
    The job's counts at one alpha: from the cache, or simulated and cached.

    A cached file is used only if its calibration, simulator seed and error
    class (``which``; files without the key are ``all``) match.
    """
    path = cache / f"counts_alpha_{alpha:g}.json"
    sha = sha256_file(calibration)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if (stored.get("calibration_sha256") == sha and stored.get("seed") == SEED_SIMULATOR
                and stored.get("which", "all") == which):
            return [dict(c) for c in stored["counts"]]
        raise SystemExit(f"{rel(path)} is from another calibration, seed or error class; "
                         "a cached simulation is never overwritten (move it away to redo it).")
    snapshot, device, info = noise_device(calibration, alpha, which)
    print(f"  alpha = {alpha:g} ({which}): simulating {len(circuits)} circuits"
          + (f" ({info['n_capped']} errors capped at 0.5)" if info.get("n_capped") else ""),
          flush=True)
    counts, _ = simulate(manifest, circuits, device, max_threads=max_threads,
                         meta={"stage": "wp6_alpha_fit", "alpha": alpha, "which": which})
    write_json(path, {
        "alpha": alpha,
        "which": which,
        "calibration_file": rel(calibration),
        "calibration_sha256": sha,
        "seed": SEED_SIMULATOR,
        "scale_info": info,
        "counts": counts,
        "created_utc": utc_timestamp(),
    })
    return counts


def default_alpha_grid(*caches: Path) -> list[float]:
    """The grid of the first cache holding a ``fit.json``, else :data:`DEFAULT_ALPHAS`."""
    for cache in caches:
        fit = cache / "fit.json"
        if fit.exists():
            return [float(a) for a in json.loads(fit.read_text(encoding="utf-8"))["alphas"]]
    return list(DEFAULT_ALPHAS)


#: Keys of a stored fit that a refit from the same cache must reproduce exactly.
REPRODUCED_KEYS: tuple[str, ...] = ("alpha", "ci", "ci_scaled", "bootstrap_se", "chi2_min",
                                    "dof", "n_pubs", "loo", "profile")


def fit_differences(stored: Mapping[str, Any], refit: Mapping[str, Any]) -> list[str]:
    """``"<subset>.<key>"`` wherever a refit's ``fits`` differ from the stored ones."""
    diffs: list[str] = []
    old, new = stored.get("fits", {}), refit.get("fits", {})
    for name in sorted(set(old) | set(new)):
        if name not in old or name not in new:
            diffs.append(f"{name} (only in {'stored' if name in old else 'refit'})")
            continue
        diffs.extend(f"{name}.{key}" for key in REPRODUCED_KEYS
                     if json_plain(old[name].get(key)) != json_plain(new[name].get(key)))
    return diffs


def run_fit_alpha(args: argparse.Namespace, paths: Paths) -> int:
    """``--fit-alpha``: simulate the alpha grid, fit it to the hardware."""
    manifest, _, circuits = load_manifest(paths.manifest)
    calibration = Path(args.calibration) if args.calibration else default_calibration(paths)
    if not calibration.exists():
        print(f"{rel(calibration)} not found; run --fetch-calibration first, or pass "
              "--calibration.")
        return 2
    label = args.label or calibration_label(calibration, paths.date)
    which = args.scale_only
    cache = alpha_cache_dir(paths.data_dir, label, which, paths.date)
    alphas = sorted(float(a) for a in (
        args.alphas or default_alpha_grid(cache, alpha_cache_dir(paths.data_dir, label, "all",
                                                                 paths.date))))
    shots = int(manifest["settings"]["shots"])

    pubs = signal_pubs(manifest)
    hw_counts = hardware_counts(paths)
    hw = np.array([solution_probability(hw_counts[p["pub"]], p["solution_bitstrings"])
                   for p in pubs])
    print(f"alpha fit ({which} errors): {len(pubs)} PUBs with signal, alphas {alphas}, "
          f"calibration {rel(calibration)}")
    sim = np.empty((len(alphas), len(pubs)))
    for i, alpha in enumerate(alphas):
        counts = cached_counts(cache, alpha, calibration, manifest, circuits, args.max_threads,
                               which)
        sim[i] = [solution_probability(counts[p["pub"]], p["solution_bitstrings"]) for p in pubs]

    # One subset per fold scale the job ran, then every PUB.  On the odd grid
    # this is exactly lambda1 / lambda3 / lambda5 / all.
    scales_run = sorted({p["scale"] for p in pubs})
    subsets = {f"lambda{s:g}": [s] for s in scales_run} | {"all": scales_run}
    fits: dict[str, Any] = {}
    for name, scales in subsets.items():
        keep = np.array([p["scale"] in scales for p in pubs])
        if keep.sum() < 2:
            continue
        fits[name] = fit_alpha(alphas, sim[:, keep], shots, hw[keep], shots,
                               groups=[p["key"] for p, k in zip(pubs, keep, strict=True) if k])

    per_pub = [
        {
            **{k: p[k] for k in ("pub", "key", "scale", "p_uniform", "p_ideal")},
            "hardware": float(hw[j]),
            "predicted": {f"{a:g}": float(sim[i, j]) for i, a in enumerate(alphas)},
        }
        for j, p in enumerate(pubs)
    ]
    payload = {
        "schema": FIT_SCHEMA,
        "date": paths.date,
        "calibration_file": rel(calibration),
        "calibration_sha256": sha256_file(calibration),
        "alphas": alphas,
        "scale_only": which,
        "what_alpha_scales": what_alpha_scales(which),
        "pubs_excluded": "instances with P_ideal == P_uniform (M = 0, M = N/2)",
        "shots": shots,
        "seed_simulator": SEED_SIMULATOR,
        "fits": fits,
        "per_pub": per_pub,
        "created_utc": utc_timestamp(),
        "provenance": record_provenance(),
    }

    print(f"\n{'fitted on':<10} {'alpha':>6}  {'95% CI (shots)':<15} "
          f"{'95% CI (scaled)':<16} {'chi2/dof':>8}  {'PUBs':>4}  leave-one-record-out")
    for name, fit in fits.items():
        loo = list((fit["loo"] or {}).values())
        loo_text = f"{min(loo):.2f} .. {max(loo):.2f}" if loo else "-"
        red = f"{fit['chi2_red']:.2f}" if fit["chi2_red"] is not None else "-"
        flag = "  (at the grid edge: widen --alphas)" if fit["at_grid_edge"] else ""
        ci, wide = fit["ci"], fit["ci_scaled"]
        print(f"{name:<10} {fit['alpha']:6.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]     "
              f"[{wide[0]:.2f}, {wide[1]:.2f}]     {red:>8}  {fit['n_pubs']:>4}  "
              f"{loo_text}{flag}")

    out = cache / "fit.json"
    if out.exists() and not args.force:
        diffs = fit_differences(json.loads(out.read_text(encoding="utf-8")), payload)
        if diffs:
            print(f"\n{rel(out)} exists and this refit DIFFERS in {len(diffs)} entries "
                  f"(kept the stored file; --force to replace it):\n  " + "\n  ".join(diffs[:20]))
            return 1
        print(f"\n{rel(out)} exists; refit from the cache reproduces it exactly "
              f"({len(fits)} subsets x {', '.join(REPRODUCED_KEYS)}). Not rewritten.")
        return 0
    write_json(out, payload)
    print(f"\nwrote {rel(out)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="WP6 follow-up: the hardware vs its noise model.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--fetch-calibration", action="store_true",
                        help="Freeze the calibration in force when the job ran (needs the token).")
    action.add_argument("--predict", action="store_true",
                        help="Simulate the job on --calibration (x --alpha).")
    action.add_argument("--fit-alpha", action="store_true",
                        help="Fit the error scale alpha to the hardware P(solution).")
    parser.add_argument("--date", default=None, help="Job date; default the newest manifest.")
    parser.add_argument("--at", default=None,
                        help="--fetch-calibration: ISO time; default the job's 'running' time.")
    parser.add_argument("--label", default=None,
                        help="--fetch-calibration: file label (default exec). Otherwise: the "
                             "output label (default from the calibration file name).")
    parser.add_argument("--calibration", default=None,
                        help="Snapshot to simulate; default calibration_exec_<date>.json.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="--predict: multiply every calibrated error by this.")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="--fit-alpha: the simulated grid (default: the grid of this "
                             "cache's fit.json, else of the 'all' fit, else DEFAULT_ALPHAS).")
    parser.add_argument("--scale-only", choices=ERROR_CLASSES, default="all",
                        help="--predict / --fit-alpha: scale only this error class "
                             "(default all, the paper's fit).")
    parser.add_argument("--instance", default=None, help=f"Instance CRN; default ${INSTANCE_ENV}.")
    parser.add_argument("--max-threads", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing snapshot or differing records.")
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if args.fetch_calibration and args.label is None:
        args.label = "exec"
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the chosen mode."""
    args = parse_args(argv)
    date = args.date or latest_manifest_date(args.hardware_dir)
    if date is None:
        print(f"No manifest in {args.hardware_dir}.")
        return 2
    paths = Paths(date, args.hardware_dir, args.data_dir)
    if args.fetch_calibration:
        return run_fetch(args, paths)
    if args.predict:
        return run_predict(args, paths)
    return run_fit_alpha(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
