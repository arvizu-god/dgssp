"""
hardware_batch.py

The paper's single monthly hardware job (WP6): plan it, dry-run it, submit it,
collect it.

Modes
-----

``--plan``
    Connect to IBM Quantum (token from ``QISKIT_IBM_TOKEN``, never from a
    file), transpile every instance of ``paper/instances/hardware_subset.json``
    ONCE on the live device (SABRE seed search, opt level 3), fold each ISA
    circuit at ``[1, 3, 5]`` and rebase it, add the iteration-sweep variants of
    ``paper/instances/sweep.json`` on the *same* physical layout, and append one
    repeat of the two smallest ``M > 0`` instances at scale 1 (drift check).
    Scale 1 is the unmitigated run; nothing else is added.  Writes:

    * ``paper/hardware/manifest_<date>.json`` -- every circuit, the totals, the
      sampler options, the backend and calibration stamp, the pre-flight
      result and the ESTIMATED QPU TIME;
    * ``paper/hardware/circuits_<date>.qpy`` -- the exact circuits, so the dry
      run and the submission send the same objects;
    * ``paper/hardware/calibration_<date>.json`` -- the noise model's
      calibration (BackendV2 target) plus the raw ``properties()``.

    Nothing is submitted.  ``--offline-backend fake_marrakesh`` does the same
    against a bundled fake device without the service (rehearsal only:
    ``--submit`` refuses such a manifest).

``--dry-run``
    Sample the manifest's circuits on an ``AerSimulator`` carrying the saved
    calibration, with the same shots and the same options dict, through
    :func:`dgssp.runtime.submit_sampler_job` -- the function ``--submit``
    calls.  Writes Records (``executor="noise_model_prediction"``) to
    ``paper/data/hw_dryrun_<date>/`` and a fake-job file
    ``paper/hardware/jobs_dryrun_<date>.json`` that ``--collect`` accepts.
    Twirling and dynamical decoupling are server-side features: the local path
    ignores them with a warning, which is recorded.

``--probe`` / ``--probe-report``
    A two-PUB timing job sent before the main one: the manifest's shortest and
    longest circuits, same shots, same options, capped server-side at
    :data:`PROBE_CAP_SECONDS`.  ``--probe`` submits and exits (writing
    ``paper/hardware/probe_<date>.json``); ``--probe-report`` fetches it and
    measures the per-shot time (execution spans) and the fixed overhead
    (``job.usage()`` minus span time), says whether the server accepted
    twirling and DD, and prints the ``--plan`` flags that put those numbers
    into the main job's estimate.  It is a second job: about 10 s of QPU.

``--submit``
    Refuses unless ``paper/hardware/CHECKLIST_<date>.md`` exists with every box
    ticked and names the current, clean ``HEAD``.  Submits the manifest as ONE
    job (one PUB per circuit), writes ``paper/hardware/jobs_<date>.json`` and
    the ``runs/jobs.jsonl`` ledger, prints the job id and exits.  It never
    waits for results and refuses to submit twice for one date.

``--collect --jobs paper/hardware/jobs_<date>.json``
    Fetch each job's result (or decode a dry run's saved result), map PUBs to
    circuits through the manifest, and write Records with every metric and
    bootstrap interval, the scalar ZNE fit of every model and both pointwise
    renormalisations, and Mitiq's extrapolation run offline on the same folded
    counts (the hardware Mitiq comparison; no Mitiq-folded circuits are sent),
    to ``paper/data/hw_<date>/``.  Records QPU usage into
    ``paper/hardware/usage.md``.  Idempotent: an existing identical record is
    left alone, a different one is an error.  On a dry-run jobs file it writes
    to ``paper/data/hw_dryrun_<date>_collect/`` and compares the result with
    the dry run's own records (the acceptance check).

Run::

    python paper/scripts/hardware_batch.py --setup-help
    python paper/scripts/hardware_batch.py --plan --backend ibm_xxx
    python paper/scripts/hardware_batch.py --dry-run
    python paper/scripts/hardware_batch.py --collect --jobs paper/hardware/jobs_dryrun_<date>.json
    python paper/scripts/hardware_batch.py --probe
    python paper/scripts/hardware_batch.py --probe-report
    python paper/scripts/hardware_batch.py --plan --force --backend ibm_xxx --per-shot-time ... --overhead ...
    python paper/scripts/hardware_batch.py --submit
    python paper/scripts/hardware_batch.py --collect --jobs paper/hardware/jobs_<date>.json

Anything marked ``VERIFY`` is an assumption about the platform that the pinned
``qiskit-ibm-runtime`` (0.49.0) cannot confirm offline; the manifest lists them
with the documentation URL to check.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import date as date_cls
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from qiskit import QuantumCircuit, qpy  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

from _common import (  # noqa: E402
    DATA_DIR,
    HARDWARE_DIR,
    INSTANCES_DIR,
    REPO_ROOT,
    Record,
    git_state,
    record_provenance,
    save_record,
    utc_timestamp,
)
from dgssp.backends import (  # noqa: E402
    calibration_payload,
    calibration_timestamp,
    is_heavy_hex,
    load_noise_model,
    processor_family,
    save_noise_model,
)
from dgssp.execution import exact_index_distribution  # noqa: E402
from dgssp.experiments import (  # noqa: E402
    InstanceResult,
    PreparedInstance,
    build_instance_result,
    prepare_from_records,
)
from dgssp.instance import SubsetSumInstance  # noqa: E402
from dgssp.metrics import (  # noqa: E402
    bootstrap_metric,
    counts_to_probs,
    distribution_solution_probability,
    solution_probability,
    total_variation,
)
from dgssp.mitigation import (  # noqa: E402
    ZNE_METHODS,
    ZNEItem,
    ZNEItemRun,
    ZNESamplingConfig,
    mitiq_is_available,
    plan_zne_batch,
)
from dgssp.runtime import (  # noqa: E402
    backend_name,
    counts_from_result,
    execution_mode,
    get_service,
    job_usage,
    preflight_sampler_job,
    submit_sampler_job,
)
from dgssp.solvers import DGConfig, DGSSPSolver, grover_success_probability  # noqa: E402
from dgssp.transpilation import transpiled_metrics, two_qubit_count  # noqa: E402
from qpu_budget import DEFAULT_REP_DELAY, device_limits, schedule_seconds  # noqa: E402
from sim_campaign import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    FOLD_SEED,
    SEED_MAX,
    SEED_MIN,
    SEED_SIMULATOR,
    ZNE_METHOD,
    ZNE_MODE,
    ZNE_RENORM,
    resolve_fake_backend,
    throttle,
)

# ---------------------------------------------------------------------------
# Frozen job parameters
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA = "dgssp/paper/hardware-manifest/1"
JOBS_SCHEMA = "dgssp/paper/hardware-jobs/1"
PROBE_SCHEMA = "dgssp/paper/hardware-probe/1"

#: Server-side QPU cap on the probe job.  Two PUBs at 4096 shots are expected
#: to cost about 10 s; the cap bounds a surprise at 10 % of the allowance.
PROBE_CAP_SECONDS = 60

#: Shots per circuit (per PUB).
SHOTS = 4096

#: Noise scales.  Odd integers: every gate folded uniformly, no partial layer.
SCALES: tuple[float, ...] = (1.0, 3.0, 5.0)

#: Environment variables the service is configured from.  The token is never
#: read from, or written to, a file by this script.
TOKEN_ENV = "QISKIT_IBM_TOKEN"
INSTANCE_ENV = "QISKIT_IBM_INSTANCE"

#: The ledger every submission is appended to (``dgssp.runtime.log_job``).
RUN_LOG = REPO_ROOT / "runs" / "jobs.jsonl"

#: QPU-time estimate: ``(shots x per_shot + circuits x overhead) x safety``.
#: ``per_shot`` default (``None`` = auto): the job's LONGEST circuit, scheduled
#: on the device's own durations, plus the device's default ``rep_delay``,
#: charged to every shot -- an upper bound on what any one shot costs.  IBM's
#: rule of thumb is 0.35 ms (250 us rep_delay + ~100 us circuit).
#: ``overhead`` default: IBM quotes about 2 s per *sub-job*; charging it per
#: circuit is the worst case of one sub-job per PUB.
DEFAULT_PER_SHOT_S: float | None = None
DEFAULT_OVERHEAD_S = 2.0
SAFETY_FACTOR = 2.0

#: The job may spend at most this fraction of the remaining allowance.
BUDGET_FRACTION = 0.5

#: Open-plan allowance, used only when neither ``--remaining-seconds`` nor the
#: service reports what is left.  Placeholder, flagged as unverified.
OPEN_PLAN_SECONDS = 600.0

#: Gate twirling: 32 randomizations x 128 shots = 4096 shots exactly, so the
#: server never rounds the shot count up.  ``enable_measure`` stays off.
TWIRL_RANDOMIZATIONS = 32
TWIRL_STRATEGY = "active-accum"

#: Dynamical decoupling is OFF by default (decided 2026-10-25): it is not
#: part of C2, the noise model has no counterpart for it, and folding changes
#: each scale's idle pattern, so DD would change the noise differently at
#: lambda = 1, 3, 5.  ``--dd on`` uses this fixed sequence.
DD_SEQUENCE = "XX"

#: Label of the hardware records' Mitiq block: Mitiq's extrapolation applied
#: offline to the dgssp-folded counts (no Mitiq-folded circuits are sent).
MITIQ_FOLD_SOURCE = "dgssp_isa_folds (Mitiq extrapolation only, offline)"
DD_SCHEDULING = "alap"

#: Where the assumptions marked VERIFY can be checked.
VERIFY_URLS: dict[str, str] = {
    "qpu_time_model": "https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time",
    "execution_modes": "https://quantum.cloud.ibm.com/docs/en/guides/execution-modes",
    "twirling_options": "https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-twirling-options",
    "dd_options": "https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-dynamical-decoupling-options",
    "error_suppression": "https://quantum.cloud.ibm.com/docs/en/guides/configure-error-suppression",
    "max_execution_time": "https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time",
}

#: A drift instance must have an ideal P(solution) at least this far above
#: P_uniform.  ``M = N/2`` instances sit at P_uniform both ideally and under
#: full depolarisation, so their P(solution) cannot show drift at all.
DRIFT_MIN_SIGNAL = 0.25

#: Check-boxes CHECKLIST_template.md carries; a filled checklist must keep all.
CHECKLIST_MIN_BOXES = 10

SETUP_INSTRUCTIONS = f"""
IBM Quantum setup -- do this once per PowerShell window, before --plan or --submit
-------------------------------------------------------------------------------
1. Sign in at https://quantum.cloud.ibm.com/ with the account that holds the
   Open plan.
2. Create an API key: top-right avatar -> "API keys" (or the dashboard's
   "Create API key" box) -> name it e.g. dgssp-wp6 -> copy it once.  IBM shows
   it only once; keep it in your password manager, NOT in the repository.
3. Copy the instance CRN: dashboard -> "Instances" -> your Open-plan instance
   -> the CRN string starting with "crn:v1:bluemix:public:quantum-computing:".
4. In the PowerShell window you will run the script from (session only, nothing
   written to disk):

       $env:{TOKEN_ENV} = "<paste API key>"
       $env:{INSTANCE_ENV} = "<paste instance CRN>"

   Check with:  if ($env:{TOKEN_ENV}) {{ "token set" }}
5. Do not run QiskitRuntimeService.save_account and do not put the key in a
   .env or JSON file: this script reads it from the environment only, and every
   results file strips it.
6. Fill paper/hardware/account_facts.md (python paper/scripts/describe_account.py
   prints most of it; that script reads a *saved* account, so for it alone you
   may save one and delete it afterwards with QiskitRuntimeService.delete_account).
   Pick the primary device there and pass that name to --backend.
7. Nothing is ever submitted except by --submit, and --submit refuses to run
   until paper/hardware/CHECKLIST_<date>.md is filled in and every box ticked.
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    """``path`` relative to the repository, POSIX-style, when it is inside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve(path_text: str) -> Path:
    """A path from a manifest or jobs file: repository-relative or absolute."""
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON atomically (tmp file + rename); overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(partial, path)
    return path


def json_plain(value: Any) -> Any:
    """``value`` reduced to plain JSON types (datetimes become strings)."""
    return json.loads(json.dumps(value, default=str))


class Paths:
    """Every file one job date touches."""

    def __init__(self, date: str, hardware_dir: Path, data_dir: Path) -> None:
        self.date = date
        self.hardware_dir = hardware_dir
        self.data_dir = data_dir
        self.manifest = hardware_dir / f"manifest_{date}.json"
        self.circuits = hardware_dir / f"circuits_{date}.qpy"
        self.calibration = hardware_dir / f"calibration_{date}.json"
        self.jobs = hardware_dir / f"jobs_{date}.json"
        self.jobs_dryrun = hardware_dir / f"jobs_dryrun_{date}.json"
        self.checklist = hardware_dir / f"CHECKLIST_{date}.md"
        self.probe = hardware_dir / f"probe_{date}.json"
        self.probe_dir = data_dir / f"hw_probe_{date}"
        self.usage_json = hardware_dir / "usage.json"
        self.usage_md = hardware_dir / "usage.md"
        self.dryrun_dir = data_dir / f"hw_dryrun_{date}"
        self.dryrun_collect_dir = data_dir / f"hw_dryrun_{date}_collect"
        self.hardware_out_dir = data_dir / f"hw_{date}"


def latest_manifest_date(hardware_dir: Path) -> str | None:
    """Date of the newest ``manifest_<date>.json``, or ``None``."""
    dates = sorted(
        p.stem.removeprefix("manifest_") for p in hardware_dir.glob("manifest_*.json")
    )
    return dates[-1] if dates else None


# ---------------------------------------------------------------------------
# Settings shared by every mode
# ---------------------------------------------------------------------------


def analysis_config(shots: int, scales: Sequence[float], seed_max: int) -> ZNESamplingConfig:
    """
    The ZNE configuration of the job: the sim campaign's estimator settings.

    Parameters
    ----------
    shots:
        Shots per circuit.
    scales:
        Nominal scales.
    seed_max:
        End of the SABRE seed range searched once per instance.

    Returns
    -------
    ZNESamplingConfig
        Reported estimator exponential / scalar / clip with the ``p_uniform``
        asymptote, as in ``sim_campaign.run_zne_stage``.
    """
    return ZNESamplingConfig(
        scales=list(scales),
        shots_per_scale=shots,
        method=ZNE_METHOD,
        mode=ZNE_MODE,
        renorm=ZNE_RENORM,
        asymptote="p_uniform",
        fold_seed=FOLD_SEED,
        n_bootstrap=BOOTSTRAP_B,
        bootstrap_seed=BOOTSTRAP_SEED,
        seed_min=SEED_MIN,
        seed_max=seed_max,
        optimization_level=3,
        seed_simulator=SEED_SIMULATOR,
    )


def config_from_manifest(manifest: Mapping[str, Any]) -> ZNESamplingConfig:
    """:func:`analysis_config` with the settings a manifest recorded."""
    settings = manifest["settings"]
    return analysis_config(settings["shots"], settings["scales"], settings["seed_max"])


def twirling_support() -> dict[str, Any]:
    """
    What the installed client can express about gate twirling.

    The client is the only thing that can be inspected offline.  Whether the
    *plan* accepts twirled Sampler jobs is decided by the server when the job
    arrives; the pinned ``qiskit-ibm-runtime`` has no capability query for it.

    Returns
    -------
    dict
        ``{"client_supports_gate_twirling", "client_version",
        "plan_support"}``.
    """
    import qiskit_ibm_runtime
    from qiskit_ibm_runtime.options import SamplerOptions, TwirlingOptions

    sampler_fields = {f.name for f in fields(SamplerOptions)}
    twirl_fields = {f.name for f in fields(TwirlingOptions)}
    return {
        "client_supports_gate_twirling": "twirling" in sampler_fields
        and "enable_gates" in twirl_fields,
        "client_version": qiskit_ibm_runtime.__version__,
        # VERIFY: no client-side query exists; the server accepts or rejects.
        "plan_support": "unverified: decided server-side at submission",
    }


def sampler_options(
    shots: int,
    *,
    twirling: bool,
    dd: bool,
    date: str,
    max_execution_time: int | None = None,
) -> dict[str, Any]:
    """
    The fixed ``SamplerOptions`` of the job, as a nested dict.

    The same dict goes to ``--dry-run`` and ``--submit``; it is written to the
    manifest verbatim.

    Parameters
    ----------
    shots:
        Shots per PUB; must be a multiple of the randomization count when
        twirling is on.
    twirling:
        Enable two-qubit gate twirling.
    dd:
        Enable dynamical decoupling.
    date:
        Job date, used as a job tag.
    max_execution_time:
        Hard cap on the job's QPU usage, in seconds: the service cancels the
        job past it (https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time).
        ``--plan`` sets it to the budget cap, so a wrong estimate costs at
        most the cap.  VERIFY: whether usage before a cancellation is charged.

    Returns
    -------
    dict
        Options for ``SamplerV2(options=...)``.

    Raises
    ------
    ValueError
        If twirling is on and ``shots`` is not a multiple of
        :data:`TWIRL_RANDOMIZATIONS` (the server would round the shots up).
    """
    options: dict[str, Any] = {
        "environment": {"job_tags": ["dgssp", "paper-a", "wp6", date]},
    }
    if max_execution_time is not None:
        options["max_execution_time"] = int(max_execution_time)
    if twirling:
        if shots % TWIRL_RANDOMIZATIONS:
            raise ValueError(
                f"shots={shots} is not a multiple of {TWIRL_RANDOMIZATIONS} "
                "randomizations; twirled shots would be rounded up."
            )
        options["twirling"] = {
            "enable_gates": True,
            "enable_measure": False,
            "num_randomizations": TWIRL_RANDOMIZATIONS,
            "shots_per_randomization": shots // TWIRL_RANDOMIZATIONS,
            "strategy": TWIRL_STRATEGY,
        }
    else:
        options["twirling"] = {"enable_gates": False, "enable_measure": False}
    options["dynamical_decoupling"] = (
        {"enable": True, "sequence_type": DD_SEQUENCE, "scheduling_method": DD_SCHEDULING}
        if dd
        else {"enable": False}
    )
    return options


SAMPLER_NOTES = (
    "No measurement mitigation: SamplerV2 has no readout-error mitigation option "
    "(resilience options exist only for EstimatorV2), and measurement twirling "
    "(twirling.enable_measure) is kept off so readout noise keeps the form the "
    "calibration-derived noise model assumes. rep_delay and init_qubits are left "
    "at the device defaults. Error suppression is gate twirling only: it is "
    "noise tailoring for ZNE (folded noise grows predictably once coherent errors "
    "are randomised). Dynamical decoupling is off by design: it is not part of C2, "
    "the noise model has no counterpart, and it would change each fold's idle "
    "noise differently. The local simulator path ignores twirling "
    "(qiskit-ibm-runtime warns); under the Aer noise model, which is already "
    "stochastic, twirling does not change the expected distribution."
)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def connect(name: str, *, offline: bool, instance: str | None) -> tuple[Any, Any, str]:
    """
    The live device, or a bundled fake one for a rehearsal.

    Parameters
    ----------
    name:
        Device name (``--backend``) or fake name (``--offline-backend``).
    offline:
        Resolve a bundled fake device instead of calling the service.
    instance:
        Instance CRN; ``None`` reads :data:`INSTANCE_ENV`.

    Returns
    -------
    tuple
        ``(service_or_None, backend, source)``, ``source`` = ``"live"`` /
        ``"fake"``.

    Raises
    ------
    SystemExit
        If the token is not in the environment.
    """
    if offline:
        return None, resolve_fake_backend(name), "fake"
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(SETUP_INSTRUCTIONS)
        raise SystemExit(f"{TOKEN_ENV} is not set in this shell; see the steps above.")
    service = get_service(token=token, instance=instance or os.environ.get(INSTANCE_ENV))
    return service, service.backend(name), "live"


def backend_status(backend: Any) -> dict[str, Any]:
    """``operational``, ``pending_jobs`` and ``status_msg``, or ``None`` each."""
    try:
        status = backend.status()
    except Exception as exc:  # pragma: no cover - provider dependent
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "operational": getattr(status, "operational", None),
        "pending_jobs": getattr(status, "pending_jobs", None),
        "status_msg": getattr(status, "status_msg", None),
    }


def fresh_calibration_stamp(backend: Any) -> str | None:
    """The device's calibration stamp, re-read from the service when possible."""
    props_fn = getattr(backend, "properties", None)
    if callable(props_fn):
        try:
            stamp = getattr(props_fn(refresh=True), "last_update_date", None)
        except TypeError:
            return calibration_timestamp(backend)
        except Exception:  # pragma: no cover - network
            return calibration_timestamp(backend)
        if stamp is not None:
            iso = getattr(stamp, "isoformat", None)
            return iso() if callable(iso) else str(stamp)
    return calibration_timestamp(backend)


def raw_properties(backend: Any) -> dict[str, Any] | None:
    """``backend.properties().to_dict()`` as plain JSON, or ``None``."""
    try:
        props = backend.properties()
        return json_plain(props.to_dict()) if props is not None else None
    except Exception:  # pragma: no cover - provider dependent
        return None


def remaining_allowance(
    service: Any, override: float | None
) -> tuple[float, str, Any]:
    """
    Seconds of QPU time left this month, and where the number came from.

    Parameters
    ----------
    service:
        The live service, or ``None``.
    override:
        ``--remaining-seconds``; always wins.

    Returns
    -------
    tuple
        ``(seconds, source, raw_usage_payload)``.
    """
    if override is not None:
        return float(override), "--remaining-seconds", None
    raw: Any = None
    if service is not None:
        try:
            raw = json_plain(service.usage())
        except Exception as exc:  # pragma: no cover - network / plan dependent
            raw = {"error": f"{type(exc).__name__}: {exc}"}
        # VERIFY: the payload shape is the Platform API's, not Qiskit's; these
        # keys are what it is documented to carry, not something the client pins.
        if isinstance(raw, dict):
            if isinstance(raw.get("usage_remaining_seconds"), (int, float)):
                return float(raw["usage_remaining_seconds"]), "service.usage()", raw
            limit, used = raw.get("usage_limit_seconds"), raw.get("usage_consumed_seconds")
            if isinstance(limit, (int, float)) and isinstance(used, (int, float)):
                return float(limit - used), "service.usage() limit - consumed", raw
    return OPEN_PLAN_SECONDS, "ASSUMED full Open-plan allowance (unverified)", raw


def account_facts_status(path: Path, name: str) -> dict[str, Any]:
    """Whether ``account_facts.md`` is filled and names the chosen device."""
    if not path.exists():
        return {"file": rel(path), "exists": False, "filled": False, "backend_listed": False}
    text = path.read_text(encoding="utf-8")
    return {
        "file": rel(path),
        "exists": True,
        "filled": "Status: UNFILLED" not in text,
        "backend_listed": name in text,
    }


# ---------------------------------------------------------------------------
# --plan
# ---------------------------------------------------------------------------


def sweep_variant(
    base: PreparedInstance, r: int, backend: Any
) -> PreparedInstance:
    """
    One iteration-sweep variant on the base circuit's physical layout.

    The layout is fixed to the one the base instance's seed search chose, and
    the same transpiler seed is used; nothing is searched again.

    Parameters
    ----------
    base:
        The instance as prepared for the subset (its ``r_opt`` circuit).
    r:
        Grover iterations of this variant.
    backend:
        The device.

    Returns
    -------
    PreparedInstance
        With ``seed_source="sweep_fixed_layout"``.

    Raises
    ------
    RuntimeError
        If the transpiler did not keep the requested initial layout.
    """
    layout = base.transpiled.layout.initial_index_layout(filter_ancillas=True)
    circuit = DGSSPSolver(DGConfig(iterations=int(r))).build_circuit(base.instance)
    tqc = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        seed_transpiler=base.best_seed,
        initial_layout=list(layout),
    ).run(circuit)
    got = tqc.layout.initial_index_layout(filter_ancillas=True)
    if list(got) != list(layout):
        raise RuntimeError(
            f"{base.name} r={r}: transpiler moved the layout {list(layout)} -> {list(got)}."
        )
    return PreparedInstance(
        instance=base.instance,
        solution_bitstrings=list(base.solution_bitstrings),
        iterations=int(r),
        circuit=circuit,
        transpiled=tqc,
        best_seed=base.best_seed,
        seed_source="sweep_fixed_layout",
        scales=list(base.scales),
        p_uniform=base.p_uniform,
    )


def build_job(
    backend: Any,
    subset_rows: Sequence[Mapping[str, Any]],
    sweep_rows: Sequence[Mapping[str, Any]],
    cfg: ZNESamplingConfig,
    *,
    n_drift: int = 2,
) -> tuple[list[QuantumCircuit], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Every circuit of the job, in submission order, plus the manifest rows.

    Order: the drift instances first (so their scale-1 circuit is among the
    first PUBs), the rest of the subset in file order, the sweep variants, and
    finally the drift repeats (the last PUBs).  The drift instances are the
    smallest (fewest items, then fewest two-qubit gates on this device) whose
    ideal ``P(solution)`` exceeds ``P_uniform`` by :data:`DRIFT_MIN_SIGNAL`.

    Parameters
    ----------
    backend:
        The device to transpile for.
    subset_rows:
        ``hardware_subset.json`` ``selected`` rows.
    sweep_rows:
        ``sweep.json`` ``instances`` rows.
    cfg:
        Seed range, optimization level, scales and folding seed.
    n_drift:
        How many instances get a repeated scale-1 circuit.

    Returns
    -------
    tuple
        ``(circuits, record_rows, circuit_rows)``.
    """
    prepared: dict[str, PreparedInstance] = {}
    rows_by_name: dict[str, Mapping[str, Any]] = {}
    for row in subset_rows:
        print(f"  transpiling {row['name']} (SABRE seeds {cfg.seed_min}..{cfg.seed_max - 1})")
        prepared[row["name"]] = prepare_from_records(
            row, None, backend, cfg, scales=list(cfg.scales), use_recorded_seed=False
        )
        rows_by_name[row["name"]] = row

    eligible = [
        p for p in prepared.values()
        if grover_success_probability(
            p.instance.n_items, len(p.solution_bitstrings), p.iterations
        ) - p.p_uniform >= DRIFT_MIN_SIGNAL
    ]
    drift = sorted(
        eligible,
        key=lambda p: (p.instance.n_items, two_qubit_count(p.transpiled), p.name),
    )[:n_drift]
    drift_names = [p.name for p in drift]
    ordered = drift_names + [n for n in prepared if n not in drift_names]

    entries: list[tuple[str, str, PreparedInstance, int]] = [
        (name, "subset", prepared[name], int(rows_by_name[name]["r_opt"])) for name in ordered
    ]
    for srow in sweep_rows:
        name = srow["name"]
        r_opt = int(srow["r_opt"])
        base = prepared.get(name)
        if base is None:
            base = prepare_from_records(
                srow, None, backend, cfg, scales=list(cfg.scales), use_recorded_seed=False
            )
            entries.append((name, "sweep", base, r_opt))
        for r in srow["iterations"]:
            if int(r) == r_opt:
                continue  # the r_opt circuit is the instance's own; never sent twice
            print(f"  sweep {name} r={r} on the r_opt layout")
            entries.append((f"{name}_r{r}", "sweep", sweep_variant(base, int(r), backend), r_opt))

    items = [
        ZNEItem(
            name=key,
            circuit=item.transpiled,
            scales=list(item.scales),
            solution_bitstrings=list(item.solution_bitstrings),
        )
        for key, _, item, _ in entries
    ]
    circuits, plans = plan_zne_batch(items, backend, cfg)

    record_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    for (key, role, item, r_opt), plan in zip(entries, plans, strict=True):
        layout = item.transpiled.layout.initial_index_layout(filter_ancillas=True)
        record_rows.append(
            {
                "key": key,
                "kind": "zne",
                "instance": item.name,
                "role": role,
                "items": list(item.instance.items),
                "target": item.instance.target,
                "n_items": item.instance.n_items,
                "M": len(item.solution_bitstrings),
                "solution_bitstrings": list(item.solution_bitstrings),
                "p_uniform": item.p_uniform,
                "iterations": item.iterations,
                "r_opt": r_opt,
                "best_seed": item.best_seed,
                "seed_source": item.seed_source,
                "initial_layout": [int(q) for q in layout],
                "nominal_scales": [float(s) for s in plan["nominal"]],
                "pub_indices": list(plan["dgssp_idx"]),
                "drift_check": key in drift_names,
            }
        )
        for fold, idx, one_q in zip(
            plan["folds"], plan["dgssp_idx"], plan["one_q_realised"], strict=True
        ):
            circuit_rows.append(
                circuit_row(
                    idx, circuits[idx], key=key, instance=item.name, r=item.iterations,
                    role=role, nominal=fold.nominal_scale,
                    realised=(fold.realised_scale.numerator, fold.realised_scale.denominator),
                    realised_1q=one_q, layout=layout,
                )
            )

    for key in drift_names:
        first = next(r for r in record_rows if r["key"] == key)
        first_pub = first["pub_indices"][first["nominal_scales"].index(1.0)]
        repeat_pub = len(circuits)
        circuits.append(circuits[first_pub].copy())
        record_rows.append(
            {
                "key": f"{key}_drift",
                "kind": "drift",
                "instance": first["instance"],
                "role": "drift_repeat",
                **{k: first[k] for k in (
                    "items", "target", "n_items", "M", "solution_bitstrings", "p_uniform",
                    "iterations", "r_opt", "best_seed", "seed_source", "initial_layout",
                )},
                "nominal_scales": [1.0],
                "pub_indices": [repeat_pub],
                "repeat_of_pub": first_pub,
                "drift_check": True,
            }
        )
        base_row = next(c for c in circuit_rows if c["pub_index"] == first_pub)
        circuit_rows.append({**base_row, "pub_index": repeat_pub, "key": f"{key}_drift",
                             "role": "drift_repeat", "repeat_of_pub": first_pub})

    circuit_rows.sort(key=lambda c: c["pub_index"])
    return circuits, record_rows, circuit_rows


def circuit_row(
    index: int,
    circuit: QuantumCircuit,
    *,
    key: str,
    instance: str,
    r: int,
    role: str,
    nominal: float,
    realised: tuple[int, int],
    realised_1q: float | None,
    layout: Sequence[int],
) -> dict[str, Any]:
    """One manifest row: what a PUB is, and how big the circuit is."""
    sizes = transpiled_metrics(circuit)
    return {
        "pub_index": index,
        "key": key,
        "instance": instance,
        "r": int(r),
        "role": role,
        "nominal_scale": float(nominal),
        "realised_scale": realised[0] / realised[1],
        "realised_scale_exact": [int(realised[0]), int(realised[1])],
        "realised_scale_1q": realised_1q,
        "two_q_count": sizes["two_q_count"],
        "two_q_depth": sizes["two_q_depth"],
        "depth": sizes["depth"],
        "physical_qubits": sizes["physical_qubits"],
        "initial_layout": [int(q) for q in layout],
        "repeat_of_pub": None,
    }


def estimate_qpu_time(
    circuit_rows: Sequence[dict[str, Any]],
    circuits: Sequence[QuantumCircuit],
    backend: Any,
    *,
    shots: int,
    per_shot: float | None,
    overhead: float,
    remaining: float,
    remaining_source: str,
) -> dict[str, Any]:
    """
    The QPU-time estimate and the schedule-based floor it is checked against.

    ``estimate = (total_shots x per_shot + n_circuits x overhead) x 2``, where
    ``per_shot=None`` means the longest circuit's duration + ``rep_delay``.  The
    floor, ``sum shots x (scheduled duration + rep_delay)``, is the
    ``qpu_budget.py`` model: execution only, no overhead.  Each row gains
    ``duration_s``.

    Returns
    -------
    dict
        Inputs, the estimate in seconds and minutes, the floor, the budget
        line and whether the job fits.
    """
    limits = device_limits(backend)
    rep_delay = limits["default_rep_delay"] or DEFAULT_REP_DELAY
    durations: list[float | None] = []
    for row in circuit_rows:
        try:
            duration: float | None = schedule_seconds(circuits[row["pub_index"]], backend)
        except Exception:  # pragma: no cover - uncalibrated durations
            duration = None
        row["duration_s"] = duration
        durations.append(duration)

    known = [d for d in durations if d is not None]
    total_shots = shots * len(circuit_rows)
    floor = sum(shots * (d + rep_delay) for d in known) if known else None
    longest_per_shot = (max(known) + rep_delay) if known else None
    if per_shot is None:
        if longest_per_shot is None or len(known) < len(durations):
            raise SystemExit("Some circuit has no calibrated duration; pass --per-shot-time.")
        per_shot, per_shot_source = longest_per_shot, "auto: longest circuit + rep_delay"
    else:
        per_shot_source = "--per-shot-time"
    estimate = (total_shots * per_shot + len(circuit_rows) * overhead) * SAFETY_FACTOR
    cap = BUDGET_FRACTION * remaining
    return {
        "formula": "(total_shots * per_shot_time_s + n_circuits * per_circuit_overhead_s) * safety_factor",
        "per_shot_time_s": per_shot,
        "per_shot_time_source": per_shot_source,
        "per_circuit_overhead_s": overhead,
        "safety_factor": SAFETY_FACTOR,
        "total_shots": total_shots,
        "n_circuits": len(circuit_rows),
        "estimated_seconds": estimate,
        "estimated_minutes": estimate / 60.0,
        "floor_seconds": floor,
        "floor_model": "sum shots * (ASAP-scheduled duration + default rep_delay), execution only",
        "rep_delay_s": rep_delay,
        "longest_circuit_s": max(known) if known else None,
        "longest_per_shot_s": longest_per_shot,
        "per_shot_below_longest_circuit": (
            longest_per_shot is not None and per_shot < longest_per_shot
        ),
        "remaining_allowance_seconds": remaining,
        "remaining_allowance_source": remaining_source,
        "budget_fraction": BUDGET_FRACTION,
        "budget_cap_seconds": cap,
        "fraction_of_remaining": estimate / remaining if remaining else None,
        "within_budget": estimate <= cap,
        "verify": (
            "Measure per_shot and overhead from job.usage() of a previous small job "
            "(paper/hardware/usage.md once one exists), or from "
            + VERIFY_URLS["qpu_time_model"]
            + " (quick rule there: 2 s + 0.00035 s per execution)."
        ),
        "device_limits": limits,
    }


def run_plan(args: argparse.Namespace, paths: Paths) -> int:
    """``--plan``: build, freeze, validate and cost the job.  Submits nothing."""
    name = args.offline_backend or args.backend
    if not name:
        print("--plan needs --backend <device> (or --offline-backend <fake> to rehearse).")
        return 2
    for existing in (paths.jobs,):
        if existing.exists():
            print(f"REFUSED: {rel(existing)} exists -- a job was already submitted from "
                  "this date's manifest. Use another --date.")
            return 1
    if paths.manifest.exists() and not args.force:
        print(f"REFUSED: {rel(paths.manifest)} exists. Pass --force to replace this date's "
              "plan, or use another --date.")
        return 1
    # Without a manifest, a circuits or calibration file is left over from a
    # plan that failed part-way: it was never part of a plan, so it is replaced.
    orphans = [p for p in (paths.circuits, paths.calibration)
               if p.exists() and not paths.manifest.exists()]
    for orphan in orphans:
        print(f"note: replacing {rel(orphan)}, left over from an unfinished plan.")

    print(SETUP_INSTRUCTIONS if not args.offline_backend else
          "OFFLINE REHEARSAL: fake device, no service; --submit will refuse this manifest.")

    service, backend, source = connect(name, offline=bool(args.offline_backend),
                                       instance=args.instance)
    facts = account_facts_status(args.hardware_dir / "account_facts.md", name)
    if source == "live" and not facts["filled"]:
        print(f"WARNING: {facts['file']} still says 'Status: UNFILLED' near the top; change "
              "that line to 'Status: filled YYYY-MM-DD'. The checklist requires it.")
    if source == "live" and not facts["backend_listed"]:
        print(f"WARNING: {facts['file']} does not name {name}. The checklist requires its "
              "primary device to be the one planned.")
    status = backend_status(backend)
    stamp_before = fresh_calibration_stamp(backend)
    print(f"backend {backend_name(backend)} ({source}), {getattr(backend, 'num_qubits', '?')} "
          f"qubits, heavy-hex={is_heavy_hex(backend)}, calibration {stamp_before}, "
          f"status {status}")
    try:
        calibration_payload(backend)  # the snapshot must be possible before any work
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 1

    subset = json.loads((args.instances_dir / "hardware_subset.json").read_text(encoding="utf-8"))
    sweep = json.loads((args.instances_dir / "sweep.json").read_text(encoding="utf-8"))
    subset_rows = list(subset["selected"])
    sweep_rows = list(sweep["instances"])
    if args.only:
        subset_rows = [r for r in subset_rows if r["name"] in args.only]
        sweep_rows = [r for r in sweep_rows if r["name"] in args.only]
    if args.no_sweep:
        sweep_rows = []

    cfg = analysis_config(args.shots, args.scales, args.seed_max)
    circuits, record_rows, circuit_rows = build_job(backend, subset_rows, sweep_rows, cfg)
    for row in circuit_rows:
        row["shots"] = args.shots

    remaining, remaining_source, usage_raw = remaining_allowance(service, args.remaining_seconds)
    options = sampler_options(args.shots, twirling=args.twirling == "on",
                              dd=args.dd == "on", date=paths.date,
                              max_execution_time=math.floor(BUDGET_FRACTION * remaining))
    preflight = preflight_sampler_job(backend, circuits, shots=args.shots, options=options)

    # Freeze the circuits, then the calibration they were transpiled against.
    # Both or neither: a failure removes whatever this run wrote.
    from qiskit_aer.noise import NoiseModel

    buffer = io.BytesIO()
    qpy.dump(circuits, buffer)
    try:
        paths.circuits.parent.mkdir(parents=True, exist_ok=True)
        paths.circuits.write_bytes(buffer.getvalue())
        save_noise_model(
            paths.calibration,
            NoiseModel.from_backend(backend),
            backend,
            overwrite=True,
            raw_properties=raw_properties(backend),
        )
        snapshot = load_noise_model(paths.calibration)
    except BaseException:
        for written in (paths.circuits, paths.calibration):
            written.unlink(missing_ok=True)
        raise
    snapshot_matches = json_plain(calibration_payload(snapshot.backend)) == json_plain(
        calibration_payload(backend)
    )
    snapshot_preflight = preflight_sampler_job(
        snapshot.backend, circuits, shots=args.shots, options=options
    )
    stamp_after = fresh_calibration_stamp(backend)
    calibration_stable = stamp_before == stamp_after

    estimate = estimate_qpu_time(
        circuit_rows, circuits, backend, shots=args.shots, per_shot=args.per_shot_time,
        overhead=args.overhead, remaining=remaining, remaining_source=remaining_source,
    )

    limits = estimate["device_limits"]
    limit_problems = []
    if limits.get("max_circuits") and len(circuits) > limits["max_circuits"]:
        limit_problems.append(f"{len(circuits)} PUBs > max_circuits={limits['max_circuits']}")
    if limits.get("max_shots") and args.shots > limits["max_shots"]:
        limit_problems.append(f"shots {args.shots} > max_shots={limits['max_shots']}")

    by_role: dict[str, int] = {}
    for row in circuit_rows:
        by_role[row["role"]] = by_role.get(row["role"], 0) + 1

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "date": paths.date,
        "created_utc": utc_timestamp(),
        "backend": {
            "name": backend_name(backend),
            "source": source,
            "num_qubits": getattr(backend, "num_qubits", None),
            "processor_family": processor_family(backend),
            "heavy_hex": is_heavy_hex(backend),
            "status_at_plan": status,
            "calibration_timestamp": stamp_before,
            "calibration_timestamp_after_plan": stamp_after,
            "calibration_stable_during_plan": calibration_stable,
            "calibration_file": rel(paths.calibration),
            "calibration_sha256": sha256_file(paths.calibration),
            "snapshot_matches_live_target": snapshot_matches,
        },
        "account_facts": facts,
        "circuits_file": rel(paths.circuits),
        "circuits_sha256": sha256_file(paths.circuits),
        "settings": {
            "shots": args.shots,
            "scales": [float(s) for s in args.scales],
            "seed_min": SEED_MIN,
            "seed_max": args.seed_max,
            "optimization_level": 3,
            "layout_method": cfg.layout_method,
            "fold_seed": FOLD_SEED,
            "fold": "dgssp.mitigation.fold_isa (= fold_transpiled) + rebase_to_backend, "
                    "via plan_zne_batch; submitted as-is (no further transpilation)",
            "zne": {
                "method": ZNE_METHOD,
                "mode": ZNE_MODE,
                "renorm": ZNE_RENORM,
                "asymptote": "p_uniform",
                "methods_fitted": list(ZNE_METHODS),
                "n_bootstrap": BOOTSTRAP_B,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "seed_simulator": SEED_SIMULATOR,
            "drift_check": "scale-1 repeat, as the last PUBs, of the two smallest instances "
                           f"whose ideal P_sol exceeds P_uniform by >= {DRIFT_MIN_SIGNAL}",
            "instances_files": {
                "hardware_subset": rel(args.instances_dir / "hardware_subset.json"),
                "hardware_subset_sha256": sha256_file(args.instances_dir / "hardware_subset.json"),
                "sweep": rel(args.instances_dir / "sweep.json"),
                "sweep_sha256": sha256_file(args.instances_dir / "sweep.json"),
            },
        },
        "sampler": {
            "options": options,
            "execution_mode_default": "job",
            "twirling": {**twirling_support(), "enabled": args.twirling == "on"},
            "dynamical_decoupling": {"enabled": args.dd == "on", "sequence_type": DD_SEQUENCE
                                     if args.dd == "on" else None},
            "measurement_mitigation": "none",
            "notes": SAMPLER_NOTES,
        },
        "records": record_rows,
        "circuits": circuit_rows,
        "totals": {
            "circuits": len(circuits),
            "shots": args.shots * len(circuits),
            "by_role": by_role,
            "records": len(record_rows),
        },
        "estimate": estimate,
        "service_usage_raw": usage_raw,
        "preflight": {
            "live_target": preflight,
            "snapshot_target": {k: v for k, v in snapshot_preflight.items()
                                if k != "program_inputs"},
            "device_limits_problems": limit_problems,
            "passed": bool(preflight["passed"] and snapshot_preflight["passed"]
                           and not limit_problems and snapshot_matches),
            "not_covered": "plan entitlements and server-side option support "
                           "(twirling, DD) are only checked when the job arrives",
        },
        "verify": VERIFY_URLS,
        "provenance": record_provenance(backend),
    }
    write_json(paths.manifest, manifest)
    print_plan_summary(manifest)

    ok = manifest["preflight"]["passed"] and calibration_stable and estimate["within_budget"]
    print(f"\nwrote {rel(paths.manifest)}\n      {rel(paths.circuits)}\n      {rel(paths.calibration)}")
    if not calibration_stable:
        print("FAIL: the device recalibrated while the plan ran. Re-run --plan --force.")
    print("\nNext: review the manifest, then run --dry-run." if ok else
          "\nPlan written, but it is NOT ready to submit (see FAIL lines above).")
    return 0 if ok else 1


def print_plan_summary(manifest: Mapping[str, Any]) -> None:
    """The manifest, as a table plus the budget line."""
    print(f"\n{'pub':>3}  {'record':<22} {'r':>2} {'lam':>4} {'2q':>6} {'2q-dep':>6} "
          f"{'dur us':>7}  physical qubits")
    for row in manifest["circuits"]:
        duration = row.get("duration_s")
        print(f"{row['pub_index']:>3}  {row['key']:<22} {row['r']:>2} "
              f"{row['nominal_scale']:>4g} {row['two_q_count']:>6} {row['two_q_depth']:>6} "
              f"{(duration or float('nan')) * 1e6:>7.1f}  {row['physical_qubits']}")

    totals, est = manifest["totals"], manifest["estimate"]
    print(f"\n{totals['circuits']} circuits ({totals['by_role']}), "
          f"{totals['shots']} shots, {totals['records']} records")
    print(f"sampler options: {json.dumps(manifest['sampler']['options'])}")
    print(f"twirling: {manifest['sampler']['twirling']}")
    print("measurement mitigation: none (SamplerV2 has none)")

    print("\npre-flight (no submission):")
    for target, block in (("live", manifest["preflight"]["live_target"]),
                          ("snapshot", manifest["preflight"]["snapshot_target"])):
        for check, outcome in block["checks"].items():
            print(f"  [{target}] {check:<26} {outcome}")
    for problem in manifest["preflight"]["device_limits_problems"]:
        print(f"  FAIL device limit: {problem}")
    print(f"  snapshot target == live target: {manifest['backend']['snapshot_matches_live_target']}")

    floor = est["floor_seconds"]
    print(f"\nQPU floor (scheduled execution only): "
          f"{floor:.1f} s" if floor is not None else "\nQPU floor: unavailable")
    if est["longest_per_shot_s"] is not None:
        print(f"longest circuit {est['longest_circuit_s'] * 1e6:.1f} us + rep_delay "
              f"{est['rep_delay_s'] * 1e6:.0f} us = {est['longest_per_shot_s'] * 1e6:.1f} us/shot")
    if est["per_shot_below_longest_circuit"]:
        print("WARNING: --per-shot-time is below the longest circuit's per-shot time; "
              "it is not a conservative bound. Raise it.")
    print(f"\nESTIMATED QPU TIME = ({est['total_shots']} shots x {est['per_shot_time_s']:.6g} s "
          f"+ {est['n_circuits']} circuits x {est['per_circuit_overhead_s']:g} s) x "
          f"{est['safety_factor']:g} = {est['estimated_seconds']:.1f} s "
          f"= {est['estimated_minutes']:.2f} min")
    print(f"remaining allowance {est['remaining_allowance_seconds']:.0f} s "
          f"[{est['remaining_allowance_source']}]; cap {est['budget_fraction']:.0%} = "
          f"{est['budget_cap_seconds']:.0f} s -> "
          + ("WITHIN BUDGET" if est["within_budget"] else "FAIL: OVER BUDGET"))
    if not est["within_budget"]:
        print("levers: a measured --per-shot-time / --overhead (see below), --shots 2048, "
              "--no-sweep, or fewer --scales. Each changes the experiment; decide, then re-plan.")
    print(f"verify per-shot time and overhead: {est['verify']}")


# ---------------------------------------------------------------------------
# Manifest loading, shared by --dry-run / --submit / --collect
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> tuple[dict[str, Any], str, list[QuantumCircuit]]:
    """
    A manifest, its hash, and its circuits -- after checking every file hash.

    Raises
    ------
    SystemExit
        If the circuits or calibration file no longer match the manifest.
    """
    if not path.exists():
        raise SystemExit(f"{rel(path)} not found; run --plan first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{rel(path)} is not a {MANIFEST_SCHEMA} file.")
    circuits_path = resolve(manifest["circuits_file"])
    calibration_path = resolve(manifest["backend"]["calibration_file"])
    for file_path, expected in ((circuits_path, manifest["circuits_sha256"]),
                                (calibration_path, manifest["backend"]["calibration_sha256"])):
        if sha256_file(file_path) != expected:
            raise SystemExit(f"{rel(file_path)} changed since --plan wrote the manifest. "
                             "Re-run --plan.")
    with circuits_path.open("rb") as handle:
        circuits = list(qpy.load(handle))
    if len(circuits) != manifest["totals"]["circuits"]:
        raise SystemExit("circuit count in the QPY file disagrees with the manifest.")
    return manifest, sha256_file(path), circuits


def job_inputs(
    manifest: Mapping[str, Any], circuits: Sequence[QuantumCircuit]
) -> tuple[list[QuantumCircuit], int, dict[str, Any]]:
    """
    The exact ``(circuits, shots, options)`` both --dry-run and --submit send.

    Returns
    -------
    tuple
        Circuits in PUB order, shots per PUB, and the options dict.
    """
    return list(circuits), int(manifest["settings"]["shots"]), dict(manifest["sampler"]["options"])


# ---------------------------------------------------------------------------
# Records (shared by --dry-run and --collect)
# ---------------------------------------------------------------------------


def hardware_block(
    manifest: Mapping[str, Any],
    manifest_sha: str,
    jobs_doc: Mapping[str, Any],
    row: Mapping[str, Any],
    job_id: str | None,
) -> dict[str, Any]:
    """The ``metrics["hardware"]`` block: where a record's counts came from."""
    return {
        "date": manifest["date"],
        "manifest": f"manifest_{manifest['date']}.json",
        "manifest_sha256": manifest_sha,
        "circuits_sha256": manifest["circuits_sha256"],
        "calibration_file": manifest["backend"]["calibration_file"],
        "source": jobs_doc["source"],
        "job_id": job_id,
        "execution_mode": jobs_doc.get("execution_mode"),
        "record_key": row["key"],
        "role": row["role"],
        "r_opt": row["r_opt"],
        "pub_indices": list(row["pub_indices"]),
        "initial_layout": list(row["initial_layout"]),
        "sampler_options": manifest["sampler"]["options"],
        "ignored_options": jobs_doc.get("ignored_options", []),
    }


def prepared_from_row(
    row: Mapping[str, Any], transpiled: QuantumCircuit
) -> PreparedInstance:
    """Rebuild the submitted instance from its manifest row and ISA circuit."""
    instance = SubsetSumInstance(items=list(row["items"]), target=row["target"],
                                 name=row["instance"])
    return PreparedInstance(
        instance=instance,
        solution_bitstrings=list(row["solution_bitstrings"]),
        iterations=int(row["iterations"]),
        circuit=DGSSPSolver(DGConfig(iterations=int(row["iterations"]))).build_circuit(instance),
        transpiled=transpiled,
        best_seed=int(row["best_seed"]),
        seed_source=row["seed_source"],
        scales=list(row["nominal_scales"]),
        p_uniform=float(row["p_uniform"]),
    )


def zne_run_from_rows(
    key: str,
    circuit_rows: Sequence[Mapping[str, Any]],
    counts: Sequence[dict[str, int]],
    fold_seed: int,
    *,
    with_mitiq: bool = False,
) -> ZNEItemRun:
    """
    A :class:`ZNEItemRun` from manifest rows and the counts of their PUBs.

    With ``with_mitiq`` the same counts are also handed to Mitiq, keyed by the
    realised scales, so ``analyse_zne_item`` runs Mitiq's extrapolation on the
    dgssp folds: an offline comparison of the two extrapolation codes on
    identical data, which costs no QPU time.  It is not the simulations'
    Mitiq baseline, which folds with Mitiq's own ``fold_gates_at_random``.
    """
    realised = [float(c["realised_scale"]) for c in circuit_rows]
    counts_per_scale = {
        float(c["realised_scale"]): dict(counts[c["pub_index"]]) for c in circuit_rows
    }
    return ZNEItemRun(
        name=key,
        nominal_scales=[float(c["nominal_scale"]) for c in circuit_rows],
        realised_scales=realised,
        realised_scales_exact=[
            (int(c["realised_scale_exact"][0]), int(c["realised_scale_exact"][1]))
            for c in circuit_rows
        ],
        realised_scales_1q=[c["realised_scale_1q"] for c in circuit_rows],
        counts_per_scale=counts_per_scale,
        mitiq_scales=list(realised) if with_mitiq else [],
        mitiq_realised_scales=list(realised) if with_mitiq else [],
        mitiq_counts_per_scale=dict(counts_per_scale) if with_mitiq else {},
        fold_seed=fold_seed,
        pub_indices={"dgssp": [int(c["pub_index"]) for c in circuit_rows], "mitiq": []},
    )


def drift_metrics(
    first: Mapping[str, int], repeat: Mapping[str, int], solutions: Sequence[str]
) -> dict[str, Any]:
    """
    Did the device drift between the first and last PUB?

    A two-proportion z-test on ``P(solution)`` and the total-variation distance
    between the two histograms.

    Returns
    -------
    dict
        ``{"p_first", "p_repeat", "diff", "z", "p_value", "tv_between"}``.
    """
    n1, n2 = sum(first.values()), sum(repeat.values())
    x1 = sum(first.get(s, 0) for s in solutions)
    x2 = sum(repeat.get(s, 0) for s in solutions)
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return {
        "p_first": p1,
        "p_repeat": p2,
        "diff": p2 - p1,
        "z": z,
        "p_value": math.erfc(abs(z) / math.sqrt(2)),
        "tv_between": total_variation(first, repeat),
    }


def build_records(
    manifest: Mapping[str, Any],
    manifest_sha: str,
    circuits: Sequence[QuantumCircuit],
    counts: Sequence[dict[str, int]],
    pub_job_ids: Sequence[str | None],
    jobs_doc: Mapping[str, Any],
    device: Any,
    *,
    executor: str,
) -> list[tuple[str, Record, dict[str, Any]]]:
    """
    Every Record of the job, from per-PUB counts.  Pure offline analysis.

    Parameters
    ----------
    manifest, manifest_sha, circuits:
        From :func:`load_manifest`.
    counts:
        Counts per PUB, in PUB order.
    pub_job_ids:
        The job each PUB ran in.
    jobs_doc:
        The jobs file (hardware or dry run).
    device:
        The calibration snapshot backend (error metrics, provenance).
    executor:
        ``"hardware"`` or ``"noise_model_prediction"``.

    Returns
    -------
    list
        ``(record_key, Record, summary_row)`` per record.
    """
    cfg = config_from_manifest(manifest)
    shots = int(manifest["settings"]["shots"])
    rows_by_pub = {c["pub_index"]: c for c in manifest["circuits"]}
    out: list[tuple[str, Record, dict[str, Any]]] = []
    with_mitiq = mitiq_is_available()

    for row in manifest["records"]:
        pubs = list(row["pub_indices"])
        crow = [rows_by_pub[i] for i in pubs]
        prepared = prepared_from_row(row, circuits[pubs[0]])
        reference = exact_index_distribution(prepared.instance, prepared.iterations)
        sols = list(prepared.solution_bitstrings)
        job_id = pub_job_ids[pubs[0]]
        extra = {"hardware": hardware_block(manifest, manifest_sha, jobs_doc, row, job_id)}

        if row["kind"] == "zne":
            run = zne_run_from_rows(row["key"], crow, counts,
                                    int(manifest["settings"]["fold_seed"]),
                                    with_mitiq=with_mitiq)
            result = build_instance_result(
                prepared, run, device, cfg,
                reference=reference,
                bootstrap_samples=BOOTSTRAP_B,
                bootstrap_seed=BOOTSTRAP_SEED,
                methods=ZNE_METHODS,
                job_id=job_id,
                calibration=manifest["backend"]["calibration_timestamp"],
                elapsed_s=None,
            )
            if result.mitiq is not None:
                result.mitiq["fold_source"] = MITIQ_FOLD_SOURCE
            extra["hardware"]["mitiq"] = MITIQ_FOLD_SOURCE if with_mitiq else "mitiq not installed"
        else:
            first = counts[row["repeat_of_pub"]]
            repeat = counts[pubs[0]]
            extra["hardware"]["drift"] = {
                "first_pub": row["repeat_of_pub"],
                "repeat_pub": pubs[0],
                **drift_metrics(first, repeat, sols),
            }
            result = drift_result(prepared, repeat, reference, device, manifest, job_id)

        record = Record.from_instance_result(
            result, executor=executor, shots=shots, backend=device, extra_metrics=extra
        )
        summary = {
            "key": row["key"],
            "kind": row["kind"],
            "M": len(sols),
            "r": prepared.iterations,
            "p_ideal": distribution_solution_probability(reference, sols),
            "p_sol": result.solution_probability,
            "p_sol_ci": result.metrics_ci.get("solution_probability"),
            "zne": result.zne_fit,
            "mitiq": result.mitiq_solution_probability,
            "drift": extra["hardware"].get("drift"),
        }
        out.append((row["key"], record, summary))
    return out


def drift_result(
    prepared: PreparedInstance,
    counts: Mapping[str, int],
    reference: Mapping[str, float],
    device: Any,
    manifest: Mapping[str, Any],
    job_id: str | None,
) -> InstanceResult:
    """The unmitigated result of a drift-repeat PUB."""
    sols = list(prepared.solution_bitstrings)
    p_sol, lo, hi = bootstrap_metric(
        counts, lambda c: solution_probability(c, sols), BOOTSTRAP_B, BOOTSTRAP_SEED
    )
    sizes = transpiled_metrics(prepared.transpiled)
    return InstanceResult(
        instance=prepared.instance,
        n_qubits=prepared.circuit.num_qubits,
        iterations=prepared.iterations,
        solution_bitstrings=sols,
        num_solutions=len(sols),
        counts=dict(counts),
        distribution=counts_to_probs(counts),
        solution_probability=p_sol,
        backend_name=backend_name(device),
        depth=sizes["depth"],  # type: ignore[arg-type]
        two_qubit_depth=sizes["two_q_depth"],  # type: ignore[arg-type]
        two_qubit_count=sizes["two_q_count"],  # type: ignore[arg-type]
        physical_qubits=sizes["physical_qubits"],  # type: ignore[arg-type]
        best_seed=prepared.best_seed,
        seed_source=prepared.seed_source,
        calibration_timestamp=manifest["backend"]["calibration_timestamp"],
        job_id=job_id,
        p_uniform=prepared.p_uniform,
        reference_distribution=dict(reference),
        metrics_ci={
            "solution_probability": [lo, hi],
            "tv_to_reference": {"point": total_variation(counts, reference)},
        },
        mitigation_mode="none",
    )


def comparable(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A record without its provenance block (the part re-runs may change)."""
    return {k: v for k, v in payload.items() if k != "provenance"}


def write_records(
    directory: Path,
    records: Sequence[tuple[str, Record, dict[str, Any]]],
    *,
    force: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Write records idempotently.

    A missing file is written; an existing file whose content (provenance
    aside) is identical is left untouched; a different one is reported and
    kept unless ``force``.

    Returns
    -------
    tuple
        ``(written, unchanged, conflicts)``.
    """
    written = unchanged = 0
    conflicts: list[str] = []
    for key, record, _ in records:
        path = directory / f"{key}.json"
        new = record.to_dict()
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if comparable(old) == comparable(json_plain(new)):
                unchanged += 1
                continue
            if not force:
                conflicts.append(rel(path))
                continue
            write_json(path, new)
            written += 1
            continue
        save_record(path, record)
        written += 1
    return written, unchanged, conflicts


def _fmt(value: Any) -> str:
    """A probability for the summary table; ``nan`` for missing values."""
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "  nan"
    return f"{value:.3f}"


def print_records_summary(summaries: Sequence[Mapping[str, Any]], label: str) -> None:
    """Per record: ideal vs scale-1 vs ZNE-scalar P(solution); drift lines."""
    print(f"\n{label}")
    print(f"{'record':<22} {'M':>2} {'r':>2} {'ideal':>6}  {'P_sol lam=1 [95% CI]':<22}  "
          f"{'ZNE exp [95% CI]':<22}  resolved  ident.  Mitiq exp")
    for s in summaries:
        ci = s["p_sol_ci"] or [None, None]
        cell = f"{_fmt(s['p_sol'])} [{_fmt(ci[0])}, {_fmt(ci[1])}]"
        zne_cell, resolved, ident = "-", "-", "-"
        if s["zne"] is not None:
            z = s["zne"]
            zci = z.get("ci") or [None, None]
            zne_cell = (f"{_fmt(z.get('value'))} [{_fmt(zci[0])}, {_fmt(zci[1])}]"
                        if z.get("identifiable") else
                        f"n/i, reports {_fmt(z.get('reported_value'))}")
            resolved, ident = str(z.get("resolved")), str(z.get("identifiable"))
        print(f"{s['key']:<22} {s['M']:>2} {s['r']:>2} {_fmt(s['p_ideal']):>6}  {cell:<22}  "
              f"{zne_cell:<22}  {resolved:<8}  {ident:<6}  "
              f"{_fmt(s['mitiq']) if s.get('mitiq') is not None else '-'}")
    for s in summaries:
        if s["drift"]:
            d = s["drift"]
            print(f"drift {s['key']}: P_sol first {d['p_first']:.4f} vs last {d['p_repeat']:.4f} "
                  f"(z = {d['z']:+.2f}, p = {d['p_value']:.3f}), TV between = {d['tv_between']:.4f}")


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def run_dry_run(args: argparse.Namespace, paths: Paths) -> int:
    """``--dry-run``: the manifest's circuits on the saved noise model."""
    from qiskit_ibm_runtime.json import RuntimeEncoder

    manifest, manifest_sha, circuits = load_manifest(paths.manifest)
    snapshot = load_noise_model(resolve(manifest["backend"]["calibration_file"]))
    circuits, shots, options = job_inputs(manifest, circuits)

    simulator = throttle(snapshot.simulator(), args.max_threads)
    print(f"dry run: {len(circuits)} circuits x {shots} shots on AerSimulator with the "
          f"{snapshot.backend_name} calibration of {snapshot.calibration_timestamp}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        job = submit_sampler_job(
            simulator, circuits, shots=shots, options=options,
            seed_simulator=SEED_SIMULATOR, run_log=None,
            meta={"stage": "wp6_dryrun", "manifest": rel(paths.manifest)},
        )
        result = job.result()
    ignored = sorted({str(w.message) for w in caught if "local testing mode" in str(w.message)})
    for message in ignored:
        print(f"  note: {message}")

    job_id = f"dryrun-{paths.date}-{manifest_sha[:12]}"
    raw_path = paths.dryrun_dir / "raw" / f"result_{job_id}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(result, cls=RuntimeEncoder), encoding="utf-8")

    jobs_doc = {
        "schema": JOBS_SCHEMA,
        "date": paths.date,
        "source": "dryrun",
        "manifest": rel(paths.manifest),
        "manifest_sha256": manifest_sha,
        "backend": manifest["backend"]["name"],
        "executor_backend": f"AerSimulator.from_backend(calibration snapshot of "
                            f"{snapshot.backend_name})",
        "execution_mode": "local",
        "seed_simulator": SEED_SIMULATOR,
        "ignored_options": ignored,
        "jobs": [{
            "job_id": job_id,
            "pub_start": 0,
            "pub_stop": len(circuits),
            "raw_result": rel(raw_path),
            "raw_result_sha256": sha256_file(raw_path),
        }],
        "created_utc": utc_timestamp(),
    }
    write_json(paths.jobs_dryrun, jobs_doc)

    counts = counts_from_result(result)
    records = build_records(manifest, manifest_sha, circuits, counts, [job_id] * len(counts),
                            jobs_doc, snapshot.backend, executor="noise_model_prediction")
    written, unchanged, conflicts = write_records(paths.dryrun_dir, records, force=args.force)
    print_records_summary([s for _, _, s in records],
                          "noise-model prediction (scale 1 = unmitigated):")
    print(f"\n{written} record(s) written, {unchanged} unchanged -> {rel(paths.dryrun_dir)}")
    print(f"fake jobs file: {rel(paths.jobs_dryrun)}")
    if conflicts:
        print("CONFLICT (kept the old files; --force to replace):\n  " + "\n  ".join(conflicts))
        return 1
    print(f"\nAcceptance check: python paper/scripts/hardware_batch.py --collect "
          f"--jobs {rel(paths.jobs_dryrun)}")
    return 0


# ---------------------------------------------------------------------------
# --submit
# ---------------------------------------------------------------------------

_BOX = re.compile(r"^\s*[-*]\s*\[(?P<mark>[ xX])\]\s*(?P<text>.*)$")
_PLACEHOLDER = re.compile(r"<[a-z][a-z0-9 _-]*>|_{3,}")
_HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check_checklist(
    path: Path, *, head: str | None, dirty: bool | None, estimate_seconds: float
) -> list[str]:
    """
    Every reason the checklist does not allow a submission (empty = allowed).

    Rules: the file exists; it keeps at least :data:`CHECKLIST_MIN_BOXES`
    boxes and every one is ticked; no template placeholder is left; the
    pytest line names a prefix of ``HEAD`` and the tree is clean; the budget
    line carries two numbers, the estimate (not below the manifest's) and the
    remaining allowance, with the estimate at most half of it.

    Parameters
    ----------
    path:
        ``CHECKLIST_<date>.md``.
    head:
        Full ``HEAD`` commit.
    dirty:
        Whether the working tree is dirty.
    estimate_seconds:
        The manifest's estimated QPU seconds.

    Returns
    -------
    list[str]
        Problems found.
    """
    if not path.exists():
        return [f"{rel(path)} does not exist (copy CHECKLIST_template.md and fill it)."]
    problems: list[str] = []
    boxes = [m for line in path.read_text(encoding="utf-8").splitlines()
             if (m := _BOX.match(line))]
    if len(boxes) < CHECKLIST_MIN_BOXES:
        problems.append(f"only {len(boxes)} check-boxes; the template has {CHECKLIST_MIN_BOXES}.")
    for box in boxes:
        text = box["text"]
        if box["mark"] == " ":
            problems.append(f"not ticked: {text}")
        if _PLACEHOLDER.search(text):
            problems.append(f"placeholder left: {text}")
        lower = text.lower()
        if lower.startswith("pytest"):
            hashes = _HASH.findall(text)
            if not hashes or head is None or not any(head.startswith(h) for h in hashes):
                problems.append(f"pytest line does not name HEAD ({(head or '?')[:12]}): {text}")
            if dirty:
                problems.append("working tree is dirty; commit first so the job is traceable.")
        if lower.startswith("estimated qpu time"):
            numbers = [float(n) for n in _NUMBER.findall(text.replace("50%", ""))]
            if len(numbers) < 2:
                problems.append(f"budget line needs the estimate and the remaining seconds: {text}")
            else:
                estimate, remaining = numbers[0], numbers[1]
                if estimate + 1 < estimate_seconds:
                    problems.append(f"budget line estimate {estimate} s is below the "
                                    f"manifest's {estimate_seconds:.0f} s.")
                if estimate > BUDGET_FRACTION * remaining:
                    problems.append(f"{estimate} s exceeds {BUDGET_FRACTION:.0%} of {remaining} s.")
    return problems


def run_submit(args: argparse.Namespace, paths: Paths) -> int:
    """``--submit``: one job, ids written, exit.  Never waits for results."""
    manifest, manifest_sha, circuits = load_manifest(paths.manifest)
    if paths.jobs.exists():
        print(f"REFUSED: {rel(paths.jobs)} exists -- this manifest was already submitted. "
              "Collect it instead.")
        return 1
    if manifest["backend"]["source"] != "live":
        print("REFUSED: this manifest was planned against a fake device (rehearsal).")
        return 1
    if not manifest["preflight"]["passed"]:
        print("REFUSED: the manifest's pre-flight failed; re-run --plan.")
        return 1
    if not manifest["estimate"]["within_budget"]:
        print("REFUSED: the manifest's estimate is over budget.")
        return 1
    dry_sha = (json.loads(paths.jobs_dryrun.read_text(encoding="utf-8")).get("manifest_sha256")
               if paths.jobs_dryrun.exists() else None)
    if dry_sha != manifest_sha:
        print(f"REFUSED: no dry run of this manifest ({rel(paths.jobs_dryrun)} "
              f"{'is for an older plan' if dry_sha else 'missing'}); run --dry-run first.")
        return 1

    git = git_state(refresh=True)
    problems = check_checklist(paths.checklist, head=git["commit"], dirty=git["dirty"],
                               estimate_seconds=manifest["estimate"]["estimated_seconds"])
    if problems:
        print("REFUSED by the checklist:\n  " + "\n  ".join(problems))
        return 1

    service, backend, _ = connect(manifest["backend"]["name"], offline=False,
                                  instance=args.instance)
    status = backend_status(backend)
    if status.get("operational") is False:
        print(f"REFUSED: {backend_name(backend)} is not operational: {status}")
        return 1
    stamp = fresh_calibration_stamp(backend)
    if stamp != manifest["backend"]["calibration_timestamp"]:
        print(f"note: the device recalibrated since --plan ({manifest['backend']['calibration_timestamp']}"
              f" -> {stamp}). The circuits stay valid if the check below passes; the dry-run "
              "prediction used the older calibration.")
    circuits, shots, options = job_inputs(manifest, circuits)
    check = preflight_sampler_job(backend, circuits, shots=shots, options=options)
    if not check["passed"]:
        print(f"REFUSED: pre-flight against the live target now fails: {check['checks']}")
        return 1

    mode = "backend" if args.mode == "job" else "batch"
    with execution_mode(backend, mode=mode) as exec_mode:  # type: ignore[arg-type]
        job = submit_sampler_job(
            exec_mode, circuits, shots=shots, options=options, run_log=str(RUN_LOG),
            meta={"stage": "wp6_hardware", "manifest": rel(paths.manifest),
                  "manifest_sha256": manifest_sha, "date": paths.date},
        )
    job_id = job.job_id()
    try:
        estimation = job.usage_estimation
    except Exception as exc:  # pragma: no cover - network
        estimation = {"error": f"{type(exc).__name__}: {exc}"}

    jobs_doc = {
        "schema": JOBS_SCHEMA,
        "date": paths.date,
        "source": "hardware",
        "manifest": rel(paths.manifest),
        "manifest_sha256": manifest_sha,
        "backend": backend_name(backend),
        "execution_mode": args.mode,
        "calibration_at_submit": stamp,
        "status_at_submit": status,
        "jobs": [{
            "job_id": job_id,
            "pub_start": 0,
            "pub_stop": len(circuits),
            "submitted_utc": utc_timestamp(),
            "usage_estimation": json_plain(estimation),
        }],
        "run_log": rel(RUN_LOG),
        "created_utc": utc_timestamp(),
        "provenance": record_provenance(),
    }
    write_json(paths.jobs, jobs_doc)
    print(f"SUBMITTED job {job_id} ({len(circuits)} PUBs x {shots} shots) to "
          f"{backend_name(backend)} in {args.mode} mode.")
    print(f"usage estimation: {estimation}")
    print(f"wrote {rel(paths.jobs)} and appended {rel(RUN_LOG)}")
    print(f"\nLater: python paper/scripts/hardware_batch.py --collect --jobs {rel(paths.jobs)}")
    return 0


# ---------------------------------------------------------------------------
# --probe / --probe-report
# ---------------------------------------------------------------------------


def probe_pubs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    The two manifest circuits a probe sends: the shortest and the longest.

    The longest circuit is the one the auto per-shot estimate charges to every
    shot, so the probe measures exactly the number the estimate assumes; the
    shortest (the smallest instance at scale 1) is a hardware sanity point
    against the dry run.

    Returns
    -------
    list[dict]
        Two manifest circuit rows, shortest first.

    Raises
    ------
    SystemExit
        If the manifest carries no scheduled durations.
    """
    rows = [r for r in manifest["circuits"] if r.get("duration_s") is not None
            and r["role"] != "drift_repeat"]
    if not rows:
        raise SystemExit("The manifest has no scheduled durations; re-run --plan.")
    shortest = min(rows, key=lambda r: (r["duration_s"], r["pub_index"]))
    longest = max(rows, key=lambda r: (r["duration_s"], -r["pub_index"]))
    return [shortest, longest]


def span_seconds_per_pub(spans: Any, n_pubs: int) -> list[float] | None:
    """
    QPU seconds attributable to each PUB, from the result's execution spans.

    A span covering several PUBs is split by each PUB's share of its shots.
    Spans are documented as possibly loose (they may overlap), so these are
    upper bounds on execution time.

    Parameters
    ----------
    spans:
        ``result.metadata["execution"]["execution_spans"]``, or ``None``.
    n_pubs:
        PUBs in the job.

    Returns
    -------
    list[float] | None
        Seconds per PUB, or ``None`` when the result carries no spans
        (local simulators, or a platform that stops sending them).
    """
    if spans is None:
        return None
    seconds = [0.0] * n_pubs
    seen = False
    for span in spans:
        total = int(span.size)
        if total <= 0:
            continue
        for idx in span.pub_idxs:
            if 0 <= idx < n_pubs:
                seconds[idx] += span.duration * int(span.filter_by_pub(idx).size) / total
                seen = True
    return seconds if seen else None


def spans_union_seconds(spans: Any) -> float | None:
    """Wall time covered by the spans, overlaps counted once."""
    if spans is None:
        return None
    windows = sorted((s.start, s.stop) for s in spans)
    if not windows:
        return None
    total, (cur_start, cur_stop) = 0.0, windows[0]
    for start, stop in windows[1:]:
        if start <= cur_stop:
            cur_stop = max(cur_stop, stop)
        else:
            total += (cur_stop - cur_start).total_seconds()
            cur_start, cur_stop = start, stop
    return total + (cur_stop - cur_start).total_seconds()


def measure_probe(
    pubs: Sequence[Mapping[str, Any]],
    *,
    shots: int,
    usage_seconds: float | None,
    spans: Any,
    rep_delay: float,
    manifest_estimate: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Per-shot time and fixed overhead from one probe job, and what they imply.

    Model: ``usage = fixed + sum_i shots * per_shot_i``.  ``per_shot_i`` comes
    from the execution spans; ``fixed = usage - span time``.  The fixed part is
    charged per circuit (``fixed / n_pubs``), which over-counts it for a larger
    job if most of it is per job -- the conservative side.  Without spans, the
    per-shot time falls back to the schedule model and is flagged.

    Parameters
    ----------
    pubs:
        The probe's PUB rows (``duration_s`` each).
    shots:
        Shots per PUB.
    usage_seconds:
        ``job.usage()``.
    spans:
        The result's execution spans, or ``None``.
    rep_delay:
        The device's default ``rep_delay``.
    manifest_estimate:
        The manifest's ``estimate`` block (totals and budget cap).

    Returns
    -------
    dict
        Measured numbers, the recommended ``--per-shot-time`` / ``--overhead``,
        and the main job's estimate under them.
    """
    n = len(pubs)
    per_pub = span_seconds_per_pub(spans, n)
    model_per_shot = [float(p["duration_s"]) + rep_delay for p in pubs]
    if per_pub is not None:
        per_shot = [s / shots for s in per_pub]
        executed = spans_union_seconds(spans) or sum(per_pub)
        source = "execution spans"
    else:
        per_shot = model_per_shot
        executed = shots * sum(model_per_shot)
        source = "schedule model (no execution spans in the result)"

    out: dict[str, Any] = {
        "usage_seconds": usage_seconds,
        "per_shot_source": source,
        "per_shot_seconds": per_shot,
        "model_per_shot_seconds": model_per_shot,
        "measured_over_model": [m / s for m, s in zip(per_shot, model_per_shot, strict=True)],
        "executed_seconds": executed,
    }
    if not isinstance(usage_seconds, (int, float)) or usage_seconds <= 0:
        out["recommendation"] = None
        out["note"] = "job.usage() not final yet; run --probe-report again later."
        return out

    fixed = max(0.0, float(usage_seconds) - executed)
    rec_per_shot = max(per_shot)
    rec_overhead = fixed / n
    total_shots = int(manifest_estimate["total_shots"])
    n_circuits = int(manifest_estimate["n_circuits"])
    raw = total_shots * rec_per_shot + n_circuits * rec_overhead
    estimate = raw * SAFETY_FACTOR
    cap = float(manifest_estimate["budget_cap_seconds"])
    out.update({
        "fixed_seconds": fixed,
        "recommendation": {
            "per_shot_time_s": rec_per_shot,
            "overhead_s": rec_overhead,
            "rule": "per_shot = the slower PUB's measured per-shot time; "
                    "overhead = all fixed time / probe PUBs",
        },
        "main_job": {
            "expected_seconds": raw,
            "estimated_seconds": estimate,
            "budget_cap_seconds": cap,
            "within_budget": estimate <= cap,
        },
    })
    return out


def run_probe(args: argparse.Namespace, paths: Paths) -> int:
    """``--probe``: submit the two-PUB timing job and exit."""
    manifest, manifest_sha, circuits = load_manifest(paths.manifest)
    if paths.probe.exists():
        print(f"REFUSED: {rel(paths.probe)} exists -- the probe was already submitted. "
              "Run --probe-report.")
        return 1
    if manifest["backend"]["source"] != "live":
        print("REFUSED: this manifest was planned against a fake device (rehearsal).")
        return 1
    if not manifest["preflight"]["passed"]:
        print("REFUSED: the manifest's pre-flight failed; re-run --plan.")
        return 1

    pubs = probe_pubs(manifest)
    probe_circuits = [circuits[p["pub_index"]] for p in pubs]
    shots = int(manifest["settings"]["shots"])
    options = copy.deepcopy(dict(manifest["sampler"]["options"]))
    options["max_execution_time"] = PROBE_CAP_SECONDS
    options.setdefault("environment", {})["job_tags"] = [
        *options.get("environment", {}).get("job_tags", []), "probe"]

    service, backend, _ = connect(manifest["backend"]["name"], offline=False,
                                  instance=args.instance)
    status = backend_status(backend)
    if status.get("operational") is False:
        print(f"REFUSED: {backend_name(backend)} is not operational: {status}")
        return 1
    check = preflight_sampler_job(backend, probe_circuits, shots=shots, options=options)
    if not check["passed"]:
        print(f"REFUSED: pre-flight against the live target fails: {check['checks']}")
        return 1

    job = submit_sampler_job(
        backend, probe_circuits, shots=shots, options=options, run_log=str(RUN_LOG),
        meta={"stage": "wp6_probe", "manifest": rel(paths.manifest),
              "manifest_sha256": manifest_sha, "date": paths.date},
    )
    doc = {
        "schema": PROBE_SCHEMA,
        "date": paths.date,
        "manifest": rel(paths.manifest),
        "manifest_sha256": manifest_sha,
        "backend": backend_name(backend),
        "job_id": job.job_id(),
        "shots": shots,
        "options": options,
        "rep_delay_s": manifest["estimate"]["rep_delay_s"],
        "pubs": [{"probe_index": i, "manifest_pub": p["pub_index"], "key": p["key"],
                  "nominal_scale": p["nominal_scale"], "two_q_count": p["two_q_count"],
                  "duration_s": p["duration_s"]} for i, p in enumerate(pubs)],
        "manifest_estimate": manifest["estimate"],
        "calibration_at_submit": fresh_calibration_stamp(backend),
        "status_at_submit": status,
        "submitted_utc": utc_timestamp(),
        "provenance": record_provenance(),
    }
    write_json(paths.probe, doc)
    print(f"SUBMITTED probe {doc['job_id']}: 2 PUBs x {shots} shots "
          f"({pubs[0]['key']} lam={pubs[0]['nominal_scale']:g}, "
          f"{pubs[1]['key']} lam={pubs[1]['nominal_scale']:g}), capped at "
          f"{PROBE_CAP_SECONDS} s of QPU time.")
    print(f"wrote {rel(paths.probe)}\n\nLater: python paper/scripts/hardware_batch.py "
          f"--probe-report --date {paths.date}")
    return 0


def run_probe_report(args: argparse.Namespace, paths: Paths) -> int:
    """``--probe-report``: fetch the probe, measure, recommend.  Safe to re-run."""
    from qiskit_ibm_runtime.json import RuntimeEncoder

    if not paths.probe.exists():
        print(f"{rel(paths.probe)} not found; run --probe first.")
        return 2
    doc = json.loads(paths.probe.read_text(encoding="utf-8"))
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(SETUP_INSTRUCTIONS)
        print(f"{TOKEN_ENV} is not set.")
        return 2
    service = get_service(token=token, instance=args.instance or os.environ.get(INSTANCE_ENV))
    job = service.job(doc["job_id"])
    status = job.status()
    status_name = getattr(status, "name", str(status))
    twirling_on = bool(doc["options"].get("twirling", {}).get("enable_gates"))

    if status_name in ("ERROR", "CANCELLED"):
        try:
            message = job.error_message()
        except Exception as exc:  # pragma: no cover - network
            message = f"{type(exc).__name__}: {exc}"
        doc["measured"] = {"status": status_name, "error": message,
                           "options_accepted": False, **json_plain(job_usage(job))}
        write_json(paths.probe, doc)
        print(f"probe {doc['job_id']}: {status_name}: {message}")
        print("If the message names twirling or dynamical decoupling, re-plan with "
              "--twirling off (or --dd off) and probe again; otherwise send me the message.")
        return 1
    if status_name != "DONE":
        print(f"probe {doc['job_id']}: {status_name}. Not finished; run --probe-report again later.")
        return 3

    result = job.result()
    raw_path = paths.probe_dir / "raw" / f"result_{doc['job_id']}.json"
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(result, cls=RuntimeEncoder), encoding="utf-8")
    counts = counts_from_result(result)
    # VERIFY: the documented location of execution spans in 0.49.0.
    spans = (getattr(result, "metadata", None) or {}).get("execution", {}).get("execution_spans")
    usage = json_plain(job_usage(job))
    measured = measure_probe(
        doc["pubs"], shots=int(doc["shots"]), usage_seconds=usage.get("usage_seconds"),
        spans=spans, rep_delay=float(doc["rep_delay_s"]),
        manifest_estimate=doc["manifest_estimate"],
    )
    measured.update({
        "status": status_name,
        "options_accepted": True,
        "twirling_accepted": twirling_on,
        "counts_total": [sum(c.values()) for c in counts],
        "usage": usage,
        "reported_utc": utc_timestamp(),
    })
    doc["measured"] = measured
    write_json(paths.probe, doc)
    update_usage_store(paths, {doc["job_id"]: {
        **usage, "date": doc["date"], "backend": doc["backend"], "label": "probe",
        "n_pubs": len(doc["pubs"]), "shots": doc["shots"],
        "manifest_estimate_seconds": None, "manifest_floor_seconds": None,
    }})

    print(f"probe {doc['job_id']}: DONE on {doc['backend']}; options accepted "
          f"(gate twirling {'ON' if twirling_on else 'off'}, DD "
          f"{'ON' if doc['options'].get('dynamical_decoupling', {}).get('enable') else 'off'}).")
    print(f"job.usage() = {measured['usage_seconds']} s; per-shot source: "
          f"{measured['per_shot_source']}")
    for pub, ps, model in zip(doc["pubs"], measured["per_shot_seconds"],
                              measured["model_per_shot_seconds"], strict=True):
        print(f"  {pub['key']:<22} lam={pub['nominal_scale']:g}: {ps * 1e6:7.1f} us/shot "
              f"(schedule model {model * 1e6:.1f} us)")
    rec = measured.get("recommendation")
    if rec is None:
        print(measured.get("note"))
        return 3
    main = measured["main_job"]
    print(f"fixed (non-shot) time: {measured['fixed_seconds']:.1f} s for {len(doc['pubs'])} PUBs")
    print(f"\nmain job with these numbers: expected {main['expected_seconds']:.1f} s, "
          f"estimate x{SAFETY_FACTOR:g} = {main['estimated_seconds']:.1f} s vs cap "
          f"{main['budget_cap_seconds']:.0f} s -> "
          + ("WITHIN BUDGET" if main["within_budget"] else "STILL OVER BUDGET"))
    print("\nRe-plan with the measured numbers (subtract this probe's usage from the "
          "remaining allowance on the dashboard):")
    print(f"  python paper/scripts/hardware_batch.py --plan --force --backend {doc['backend']} "
          f"--per-shot-time {rec['per_shot_time_s']:.6g} --overhead {rec['overhead_s']:.4g} "
          "--remaining-seconds <dashboard>")
    dry = paths.dryrun_dir / f"{doc['pubs'][0]['key']}.json"
    if dry.exists() and doc["pubs"][0]["nominal_scale"] == 1.0:
        record = json.loads(dry.read_text(encoding="utf-8"))
        sols = record["metrics"]["solution_bitstrings"]
        if sols:
            print(f"\nsanity: {doc['pubs'][0]['key']} lam=1 P_sol on hardware "
                  f"{solution_probability(counts[0], sols):.3f} vs noise-model prediction "
                  f"{record['metrics']['solution_probability']:.3f}")
    return 0


def update_usage_store(paths: Paths, rows: Mapping[str, Any]) -> None:
    """Merge rows into ``usage.json`` and regenerate ``usage.md``."""
    store = (json.loads(paths.usage_json.read_text(encoding="utf-8"))
             if paths.usage_json.exists() else {})
    store.update(rows)
    write_json(paths.usage_json, store)
    paths.usage_md.write_text(render_usage_md(store), encoding="utf-8")


# ---------------------------------------------------------------------------
# --collect
# ---------------------------------------------------------------------------


def decode_result_file(path: Path) -> Any:
    """A ``PrimitiveResult`` saved with ``RuntimeEncoder``."""
    from qiskit_ibm_runtime.json import RuntimeDecoder

    return json.loads(path.read_text(encoding="utf-8"), cls=RuntimeDecoder)


def render_usage_md(usage: Mapping[str, Any]) -> str:
    """``usage.md`` from the usage store (regenerated whole, so idempotent)."""
    lines = [
        "# QPU usage of the paper's hardware jobs",
        "",
        "Generated by `paper/scripts/hardware_batch.py --collect` from `usage.json`; "
        "do not edit by hand.",
        "",
        "`usage_seconds` is `job.usage()`, which in qiskit-ibm-runtime 0.49.0 returns "
        "`metrics()['usage']['qpu_charge_time_seconds']`. What the platform counts in it: "
        f"{VERIFY_URLS['qpu_time_model']}",
        "",
        "| date | job | job id | backend | status | usage (s) | estimated by IBM (s) | "
        "estimated by manifest (s) | actual / manifest |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for job_id, row in sorted(usage.items(), key=lambda kv: (kv[1].get("date", ""), kv[0])):
        used = row.get("usage_seconds")
        planned = row.get("manifest_estimate_seconds")
        ratio = (f"{used / planned:.2f}" if isinstance(used, (int, float))
                 and isinstance(planned, (int, float)) and planned else "-")
        ibm = row.get("usage_estimation")
        ibm_s = ibm.get("quantum_seconds") if isinstance(ibm, dict) else None
        lines.append(f"| {row.get('date')} | {row.get('label', 'main')} | `{job_id}` | "
                     f"{row.get('backend')} | "
                     f"{row.get('status')} | {used} | {ibm_s} | "
                     f"{planned if planned is None else round(planned, 1)} | {ratio} |")
    lines.append("")
    for job_id, row in sorted(usage.items()):
        lines += [f"## `{job_id}`", "", "```json",
                  json.dumps(row.get("metrics"), indent=2, default=str), "```", ""]
    return "\n".join(lines)


def run_collect(args: argparse.Namespace, paths: Paths) -> int:
    """``--collect``: fetch, decode, analyse, write.  Safe to re-run."""
    from qiskit_ibm_runtime.json import RuntimeEncoder

    jobs_path = Path(args.jobs)
    jobs_doc = json.loads(jobs_path.read_text(encoding="utf-8"))
    if jobs_doc.get("schema") != JOBS_SCHEMA:
        print(f"{jobs_path} is not a {JOBS_SCHEMA} file.")
        return 2
    manifest, manifest_sha, circuits = load_manifest(resolve(jobs_doc["manifest"]))
    if manifest_sha != jobs_doc["manifest_sha256"]:
        print("REFUSED: the manifest changed after these jobs were created.")
        return 1
    paths = Paths(manifest["date"], paths.hardware_dir, paths.data_dir)
    dry = jobs_doc["source"] == "dryrun"
    out_dir = Path(args.out_dir) if args.out_dir else (
        paths.dryrun_collect_dir if dry else paths.hardware_out_dir)
    snapshot = load_noise_model(resolve(manifest["backend"]["calibration_file"]))

    service = None
    if not dry:
        token = os.environ.get(TOKEN_ENV)
        if not token:
            print(SETUP_INSTRUCTIONS)
            print(f"{TOKEN_ENV} is not set; collect needs the service for hardware jobs.")
            return 2
        service = get_service(token=token, instance=args.instance or os.environ.get(INSTANCE_ENV))

    n_pubs = manifest["totals"]["circuits"]
    counts: list[dict[str, int] | None] = [None] * n_pubs
    pub_job_ids: list[str | None] = [None] * n_pubs
    usage_rows: dict[str, Any] = {}
    for entry in jobs_doc["jobs"]:
        job_id = entry["job_id"]
        if dry or service is None:
            result = decode_result_file(resolve(entry["raw_result"]))
        else:
            job = service.job(job_id)
            status = job.status()
            status_name = getattr(status, "name", str(status))
            if status_name not in ("DONE",):
                print(f"job {job_id}: {status_name}. Not finished; run --collect again later.")
                if status_name in ("ERROR", "CANCELLED"):
                    print(f"  error: {getattr(job, 'error_message', lambda: None)()}")
                    return 1
                return 3
            result = job.result()
            raw_path = out_dir / "raw" / f"result_{job_id}.json"
            if not raw_path.exists():
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps(result, cls=RuntimeEncoder), encoding="utf-8")
            usage_rows[job_id] = {
                **json_plain(job_usage(job)),
                "date": manifest["date"],
                "backend": jobs_doc["backend"],
                "n_pubs": entry["pub_stop"] - entry["pub_start"],
                "shots": manifest["settings"]["shots"],
                "manifest_estimate_seconds": manifest["estimate"]["estimated_seconds"],
                "manifest_floor_seconds": manifest["estimate"]["floor_seconds"],
            }
        job_counts = counts_from_result(result)
        span = range(entry["pub_start"], entry["pub_stop"])
        if len(job_counts) != len(span):
            print(f"job {job_id}: {len(job_counts)} PUB results, expected {len(span)}.")
            return 1
        for index, pub_counts in zip(span, job_counts, strict=True):
            counts[index] = pub_counts
            pub_job_ids[index] = job_id

    missing = [i for i, c in enumerate(counts) if c is None]
    if missing:
        print(f"PUBs without results: {missing}")
        return 1
    shots = manifest["settings"]["shots"]
    short = [i for i, c in enumerate(counts) if sum((c or {}).values()) != shots]
    if short:
        print(f"PUBs whose counts do not sum to {shots}: {short}")
        return 1

    records = build_records(
        manifest, manifest_sha, circuits, [c for c in counts if c is not None], pub_job_ids,
        jobs_doc, snapshot.backend,
        executor="noise_model_prediction" if dry else "hardware",
    )
    written, unchanged, conflicts = write_records(out_dir, records, force=args.force)
    print_records_summary([s for _, _, s in records],
                          "dry-run jobs, re-collected:" if dry else "hardware results:")
    print(f"\n{written} record(s) written, {unchanged} unchanged -> {rel(out_dir)}")

    if usage_rows:
        update_usage_store(paths, usage_rows)
        for job_id, row in usage_rows.items():
            print(f"usage {job_id}: {row.get('usage_seconds')} s "
                  f"(manifest estimate {row['manifest_estimate_seconds']:.0f} s) "
                  f"-> {rel(paths.usage_md)}")

    if conflicts:
        print("CONFLICT (kept the old files; --force to replace):\n  " + "\n  ".join(conflicts))
        return 1
    if dry:
        return compare_dirs(paths.dryrun_dir, out_dir, [k for k, _, _ in records])
    return 0


def compare_dirs(expected_dir: Path, actual_dir: Path, keys: Sequence[str]) -> int:
    """Acceptance check: the re-collected records equal the dry run's own."""
    differ = []
    for key in keys:
        a = expected_dir / f"{key}.json"
        b = actual_dir / f"{key}.json"
        if not a.exists() or not b.exists():
            differ.append(f"{key}: missing")
            continue
        if comparable(json.loads(a.read_text(encoding="utf-8"))) != comparable(
            json.loads(b.read_text(encoding="utf-8"))
        ):
            differ.append(key)
    if differ:
        print(f"\nACCEPTANCE FAIL: {len(differ)} record(s) differ from {rel(expected_dir)}: "
              f"{differ}")
        return 1
    print(f"\nACCEPTANCE PASS: all {len(keys)} records identical to {rel(expected_dir)} "
          "(every field except provenance).")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="WP6: the single monthly hardware job.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--setup-help", action="store_true",
                        help="Print the IBM account setup steps and exit.")
    action.add_argument("--plan", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--probe", action="store_true",
                        help="Submit the two-PUB timing job (about 10 s of QPU) and exit.")
    action.add_argument("--probe-report", action="store_true",
                        help="Fetch the probe; print measured per-shot time and overhead.")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--collect", action="store_true")

    parser.add_argument("--backend", default=None,
                        help="Live device name (the primary in account_facts.md).")
    parser.add_argument("--offline-backend", default=None,
                        help="Rehearse --plan on a bundled fake device (no service).")
    parser.add_argument("--instance", default=None,
                        help=f"Instance CRN; default ${INSTANCE_ENV}.")
    parser.add_argument("--date", default=None,
                        help="Job date YYYY-MM-DD. --plan: today; otherwise the newest manifest.")
    parser.add_argument("--jobs", default=None, help="Jobs file for --collect.")
    parser.add_argument("--mode", choices=("job", "batch"), default="job",
                        help="Execution mode for --submit. One job needs no Batch; sessions "
                             "are not offered (the Open plan cannot run them).")
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--scales", type=float, nargs="+", default=list(SCALES))
    parser.add_argument("--seed-max", type=int, default=SEED_MAX,
                        help=f"SABRE seeds searched once per instance: {SEED_MIN}..seed_max-1.")
    parser.add_argument("--only", nargs="+", default=None, help="Restrict to these instances.")
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--twirling", choices=("on", "off"), default="on")
    parser.add_argument("--dd", choices=("on", "off"), default="off",
                        help="Dynamical decoupling. Off by design (see DD_SEQUENCE).")
    parser.add_argument("--per-shot-time", type=float, default=DEFAULT_PER_SHOT_S,
                        help="Seconds of QPU time per shot, for the estimate. Default: the "
                             "longest circuit's scheduled duration + rep_delay.")
    parser.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD_S,
                        help="Seconds of QPU overhead per circuit, for the estimate.")
    parser.add_argument("--remaining-seconds", type=float, default=None,
                        help="QPU seconds left this month (from the dashboard).")
    parser.add_argument("--max-threads", type=int, default=None)
    parser.add_argument("--out-dir", default=None, help="Override --collect's output directory.")
    parser.add_argument("--force", action="store_true",
                        help="Replace this date's plan, or differing records.")
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the chosen mode."""
    args = parse_args(argv)
    if args.setup_help:
        print(SETUP_INSTRUCTIONS)
        return 0
    if args.collect and not args.jobs:
        print("--collect needs --jobs paper/hardware/jobs_<date>.json")
        return 2

    date = args.date
    if date is None:
        date = (date_cls.today().isoformat() if args.plan
                else latest_manifest_date(args.hardware_dir))
    if date is None and not args.collect:
        print(f"No manifest in {args.hardware_dir}; run --plan first.")
        return 2
    paths = Paths(date or "", args.hardware_dir, args.data_dir)

    if args.plan:
        return run_plan(args, paths)
    if args.dry_run:
        return run_dry_run(args, paths)
    if args.probe:
        return run_probe(args, paths)
    if args.probe_report:
        return run_probe_report(args, paths)
    if args.submit:
        return run_submit(args, paths)
    return run_collect(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
